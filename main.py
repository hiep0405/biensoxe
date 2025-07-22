import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import re
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHAR_LIST) + 1
IMG_WIDTH = 256
IMG_HEIGHT = 64
MAX_TEXT_LEN = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_MODEL_PATH = "runs_seg_plate/yolo_seg_model2/weights/best.pt"
CRNN_MODEL_PATH = "best_crnn.pth"

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

class ResizeWithPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        tw, th = self.size[1], self.size[0]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        new_img = Image.new("L", (tw, th), 0)
        new_img.paste(img_resized, ((tw - nw) // 2, (th - nh) // 2))
        return new_img

transform = transforms.Compose([
    ResizeWithPad((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

def _decode(seq, blank_idx, int_to_char):
    res = []
    for i in range(len(seq)):
        idx = seq[i].item()
        if idx == blank_idx: continue
        if i > 0 and seq[i] == seq[i - 1]: continue
        res.append(int_to_char[idx])
    return "".join(res)

def show_debug_images(cropped, transformed_tensor):
    plt.figure(figsize=(10, 3))

    # Ảnh sau khi crop từ YOLO
    plt.subplot(1, 2, 1)
    plt.imshow(cropped, cmap='gray')
    plt.title("Sau khi crop từ YOLO")

    # Ảnh sau khi resize + pad
    plt.subplot(1, 2, 2)
    img_trans = transformed_tensor.squeeze().cpu().numpy()
    plt.imshow(img_trans, cmap='gray')
    plt.title("Sau khi resize + pad (vào CRNN)")

    plt.tight_layout()
    plt.show()

def predict_from_image(image_path):
    yolo = YOLO(YOLO_MODEL_PATH)
    crnn = CRNN(NUM_CLASSES).to(DEVICE)
    crnn.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=DEVICE), strict=False)
    crnn.eval()

    int_to_char = {i: c for i, c in enumerate(CHAR_LIST)}

    image = cv2.imread(image_path)
    results = yolo.predict(image, conf=0.25, save=False, show=False)
    if not results:
        print("YOLO không phát hiện được biển số.")
        return

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            roi = image[y1:y2, x1:x2]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            # Chuyển thành ảnh xám cho CRNN
            pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("L")
            input_tensor = transform(pil).unsqueeze(0).to(DEVICE)

            # DEBUG: hiển thị ảnh để kiểm tra
            show_debug_images(pil, input_tensor)

            with torch.no_grad():
                out = crnn(input_tensor)
                pred = out.permute(1, 0, 2).argmax(2)[0]
                text = _decode(pred, NUM_CLASSES - 1, int_to_char)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Text: {text}")

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog
    tk.Tk().withdraw()
    path = filedialog.askopenfilename()
    if path:
        predict_from_image(path)
