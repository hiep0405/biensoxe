import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import Levenshtein
import re

# --- Cài đặt chung ---
CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHAR_LIST) + 1  # +1 cho blank
IMG_WIDTH = 256
IMG_HEIGHT = 64
MAX_TEXT_LEN = 12
BATCH_SIZE = 32
EPOCHS =100
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "datasetkhung/prepare để train/paddleocr_lp_dataset"
TRAIN_LABEL = os.path.join(DATA_ROOT, "train_rec_labels.txt")
VAL_LABEL = os.path.join(DATA_ROOT, "val_rec_labels.txt")
PRETRAINED_MODEL = "crnn.pth"
SAVE_PATH = "best_crnn.pth"

# --- Dataset ---
class CRNNTextDataset(Dataset):
    def __init__(self, label_file, transform=None):
        self.samples = []
        self.transform = transform
        self.char_to_int = {c: i for i, c in enumerate(CHAR_LIST)}
        self.int_to_char = {i: c for i, c in enumerate(CHAR_LIST)}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                img_path, label = parts
                label = re.sub(r'[^0-9 A-Z]', '', label.upper())
                if 1 <= len(label) <= MAX_TEXT_LEN and all(c in CHAR_LIST for c in label):
                    self.samples.append((os.path.join(DATA_ROOT, img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        try:
            img = Image.open(path).convert("L")
        except:
            return None, None, None
        if self.transform:
            img = self.transform(img)
        label_int = [self.char_to_int[c] for c in label_str]
        label_tensor = torch.full((MAX_TEXT_LEN,), -1, dtype=torch.long)
        label_tensor[:len(label_int)] = torch.tensor(label_int)
        return img, label_tensor, torch.tensor(len(label_int))

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    images, labels, lengths = zip(*batch)
    return torch.stack(images), torch.cat([l[l != -1] for l in labels]), torch.stack(lengths)

# --- Transform ---
transform_train = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ColorJitter(0.1, 0.1),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

# --- Mô hình CRNN ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.linear_in_rnn = nn.Linear((IMG_HEIGHT // 8) * 256, 64)
        self.rnn = nn.LSTM(64, 128, bidirectional=True, num_layers=2, dropout=0.25)
        self.text_fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(3, 0, 2, 1).reshape(x.size(3), x.size(0), -1)
        x = self.linear_in_rnn(x)
        rnn_out, _ = self.rnn(x)
        ctc_out = nn.functional.log_softmax(self.text_fc(rnn_out), dim=2)
        return ctc_out

# --- Decode ---
def _decode(seq, blank_idx, int_to_char):
    res = []
    for i in range(len(seq)):
        idx = seq[i].item()
        if idx == blank_idx: continue
        if i > 0 and seq[i] == seq[i-1]: continue
        res.append(int_to_char[idx])
    return "".join(res)

# --- Train ---
def train():
    train_loader = DataLoader(CRNNTextDataset(TRAIN_LABEL, transform_train), BATCH_SIZE, True, collate_fn=collate_fn)
    val_loader = DataLoader(CRNNTextDataset(VAL_LABEL, transform_val), BATCH_SIZE, False, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    if os.path.exists(SAVE_PATH):
        print("\n🔁 Resume training from saved model...")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE), strict=False)
    elif os.path.exists(PRETRAINED_MODEL):
        print("\n⚡️ Loading pretrained base model...")
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=DEVICE), strict=False)

    criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    int_to_char = {i: c for i, c in enumerate(CHAR_LIST)}
    best_cer = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels, lengths in train_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            input_lens = torch.full((imgs.size(0),), out.size(0), dtype=torch.long).to(DEVICE)
            loss = criterion(out, labels.to(DEVICE), input_lens, lengths.to(DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")

        # --- Eval ---
        model.eval()
        errs, total = 0, 0
        with torch.no_grad():
            for imgs, labels, lengths in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds = out.permute(1, 0, 2).argmax(2)
                start = 0
                for i, l in enumerate(lengths):
                    gt = labels[start:start+l]
                    pr = _decode(preds[i], NUM_CLASSES-1, int_to_char)
                    gt_text = ''.join([int_to_char[c.item()] for c in gt])
                    errs += Levenshtein.distance(pr, gt_text)
                    total += len(gt_text)
                    start += l
        cer = errs / total if total > 0 else 0
        print(f"--> Val CER: {cer:.4f}")
        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Saved best model")

if __name__ == '__main__':
    train()