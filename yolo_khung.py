from ultralytics import YOLO
import os
import yaml
import random
from tkinter import Tk, filedialog
import cv2

import numpy as np
def get_random_image_from_dataset(dataset_yaml_path, split="train"):
    try:
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        base_dir = os.path.dirname(dataset_yaml_path)

        if split == "train":
            image_dir_key = 'train'
        elif split == "val":
            image_dir_key = 'val'
        else:
            print(f"Split '{split}' không hợp lệ. Chỉ hỗ trợ 'train' hoặc 'val'.")
            return None

        image_folder_rel = dataset_config.get(image_dir_key)
        if not image_folder_rel:
            print(f"Không tìm thấy khóa '{image_dir_key}' trong file {dataset_yaml_path}.")
            return None

        image_folder = os.path.join(base_dir, image_folder_rel)
        image_folder = os.path.abspath(image_folder)

        if not os.path.isdir(image_folder):
            print(f"Thư mục ảnh không tồn tại: {image_folder}")
            return None

        image_files = [f for f in os.listdir(image_folder) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            print(f"Không tìm thấy file ảnh nào trong thư mục: {image_folder}")
            return None

        random_image_name = random.choice(image_files)
        return os.path.join(image_folder, random_image_name)

    except Exception as e:
        print(f" Lỗi khi đọc dataset.yaml hoặc lấy ảnh: {e}")
        return None


def train_yolo_segmentation():
    model_checkpoint_path = "runs_seg_plate/yolo_seg_model2/weights/last.pt"
    run_dir = "runs_seg_plate/yolo_seg_model2"
    args_file_path = os.path.join(run_dir, "args.yaml")
    dataset_yaml_path = "datasetkhung/dataset.yaml"


    if os.path.exists(model_checkpoint_path):
        model = YOLO(model_checkpoint_path)
        print(f" Tải mô hình từ {model_checkpoint_path} để tiếp tục huấn luyện.")

        if os.path.exists(args_file_path):
            try:
                with open(args_file_path, 'r') as f:
                    args = yaml.safe_load(f)
                    previous_total_epochs = args.get('epochs', 0)
                    print(f" Mô hình này đã được huấn luyện tổng cộng **{previous_total_epochs} epoch** trước đó.")
                    print("   Nó sẽ tiếp tục huấn luyện thêm các epoch mới bạn đặt.")
            except Exception as e:
                print(f" Không thể đọc file args.yaml để lấy số epoch trước đó: {e}")
        else:
            print("Không tìm thấy file args.yaml. Không thể xác định số epoch đã train trước đó.")
    else:
        model = YOLO("yolov8n-seg.pt")
        print(" Khởi tạo mô hình YOLOv8n segmentation mới.")

    # --- Tham số huấn luyện---
    epochs_to_train = 1
    hyp_config = {
        'degrees': 5.0,  #  góc xoay
        'translate': 0.1,  #  dịch chuyển
        'scale': 0.5,  #  mức độ thay đổi kích thước
        'shear': 0.0,  #  biến dạng cắt
        'perspective': 0.000,   #biến dạng phối cảnh
        'flipud': 0.0,  # lật dọc
        'fliplr': 0.5,  # lật ngang
        'mosaic': 1.0,  # Duy trì Mosaic
        'mixup': 0.0,  # Mixup
        'hsv_h': 0.015,
        'hsv_s': 0.7,  #  độ bão hòa
        'hsv_v': 0.4,  #  giá trị (độ sáng)
        'erasing': 0.4  # Thêm ngẫu nhiên các vùng bị xóa
    }


    print("\n🚀 Bắt đầu quá trình huấn luyện mô hình YOLOv8 segmentation...")
    model.train(
        data=dataset_yaml_path,
        epochs=epochs_to_train,
        imgsz=512,
        batch=64,
        device=0,
        project="runs_seg_plate",
        name="yolo_seg_model2",
        exist_ok=True,
        workers=0,
        task="segment",
        verbose=True,
        # Thêm các tham số augmentation vào đây nếu bạn muốn tinh chỉnh chúng
        # Các tham số này sẽ ghi đè các giá trị mặc định của YOLOv8
        # Các tham số mặc định của YOLOv8 đã khá tốt cho đa số trường hợp
        # hsv_h=hyp_config['hsv_h'], hsv_s=hyp_config['hsv_s'], hsv_v=hyp_config['hsv_v'],
        # degrees=hyp_config['degrees'], translate=hyp_config['translate'], scale=hyp_config['scale'],
        # shear=hyp_config['shear'], perspective=hyp_config['perspective'],
        # flipud=hyp_config['flipud'], fliplr=hyp_config['fliplr'],
        # mosaic=hyp_config['mosaic'], mixup=hyp_config['mixup'],
        # erasing=hyp_config['erasing'] # Đây là một loại augmentation mới hơn trong các phiên bản YOLOv8 gần đây
    )
    print("✅ Huấn luyện hoàn tất!")

    # --- Phần TEST mô hình sau huấn luyện ---
    print("\n--- TEST MÔ HÌNH SAU HUẤN LUYỆN ---")
    print("🖼️ Bạn muốn chọn ảnh test từ đâu?")
    print("1. Một ảnh ngẫu nhiên từ tập Train")
    print("2. Một ảnh ngẫu nhiên từ tập Val (Test)")
    print("3. Chọn một file ảnh thủ công từ máy tính của bạn")

    choice = input("Nhập lựa chọn của bạn (1, 2 hoặc 3): ")

    test_image_path = None
    if choice == '1':
        test_image_path = get_random_image_from_dataset(dataset_yaml_path, split="train")
        if test_image_path:
            print(f"Đã chọn ngẫu nhiên ảnh từ tập Train: {test_image_path}")
    elif choice == '2':
        test_image_path = get_random_image_from_dataset(dataset_yaml_path, split="val")
        if test_image_path:
            print(f"Đã chọn ngẫu nhiên ảnh từ tập Val: {test_image_path}")
    elif choice == '3':
        print("\n🖼️ Vui lòng chọn một file ảnh để test mô hình...")
        root = Tk()
        root.withdraw()
        test_image_path = filedialog.askopenfilename(
            title="Chọn ảnh để kiểm tra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if test_image_path:
            print(f"Đã chọn ảnh thủ công: {test_image_path}")
    else:
        print(" Lựa chọn không hợp lệ. Bỏ qua bước test.")

    if test_image_path and os.path.exists(test_image_path):
        print(f"\n Đang test mô hình với ảnh: {os.path.basename(test_image_path)}...")

        # Tải mô hình tốt nhất sau khi huấn luyện
        best_model_path = os.path.join(run_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            test_model = YOLO(best_model_path)
            print(f" Đã tải mô hình tốt nhất '{best_model_path}' để test.")
        else:
            test_model = model
            print(f" Không tìm thấy '{best_model_path}'. Sử dụng mô hình cuối cùng để test.")

        results = test_model.predict(
            source=test_image_path,
            save=True,
            show=False,
            conf=0.25  # Ngưỡng độ tin cậy để hiển thị phát hiện
        )

        print(f"✅ Ảnh kết quả dự đoán đã được lưu tại thư mục: {test_model.predictor.save_dir}")
        predicted_image_name = os.path.basename(test_image_path)
        output_image_path = os.path.join(test_model.predictor.save_dir, predicted_image_name)

        if os.path.exists(output_image_path):
            img_result = cv2.imread(output_image_path)
            if img_result is not None:
                cv2.imshow("Ket qua nhan dien YOLOv8", img_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f" Không thể đọc ảnh kết quả tại {output_image_path}")
        else:
            print(f" Không tìm thấy ảnh kết quả tại {output_image_path}. Có thể YOLO không lưu ảnh.")

    elif test_image_path:
        print(f" File ảnh không tồn tại tại đường dẫn: {test_image_path}. Bỏ qua bước test.")
    else:
        pass

if __name__ == "__main__":
    print(" Chuẩn bị huấn luyện và test mô hình YOLOv8 segmentation...")
    train_yolo_segmentation()