# TinyNav - Xe tự hành bám làn sử dụng vi điều khiển ESP32-S3 và Custom CNN

Dự án phát triển hệ thống xe robot tự hành bám làn (Lane Following) khép kín thời gian thực. Hệ thống sử dụng camera trên mạch ESP32-S3 để truyền hình ảnh (stream) về máy tính qua kết nối UDP. Máy tính sử dụng mô hình mạng nơ-ron tích chập (CNN) tự thiết kế để dự đoán góc lái và gửi lệnh điều khiển ngược lại cho xe.

## 🛠️ Cấu trúc hệ thống

### 1. Phần cứng (Hardware)
*   **Vi điều khiển**: ESP32-S3 (Tích hợp PSRAM để xử lý camera mượt mà).
*   **Camera**: Mô-đun OV2640 (Cấu hình QVGA 320x240).
*   **Động cơ**: Khung xe 2 bánh chủ động (2WD) kết hợp mạch cầu H (L298N hoặc tương đương).
*   **Nguồn cấp**: Sử dụng nguồn riêng cho ESP32 và động cơ để tránh sụt áp gây reset mạch.

### 2. Sơ đồ hoạt động (Data Flow)
```
[ESP32-S3 Camera] ---> (UDP Stream Ảnh JPEG) ---> [Laptop (AI Inference)]
        ^                                                   |
        |                                                   v
[Động cơ di chuyển] <--- (Lệnh lái: Hướng + Tốc độ) <--- [Bộ xử lý bẻ lái]
```

---

## 📂 Cấu trúc thư mục dự án

```text
├── esp32_firmware/          # Mã nguồn Arduino C++ nạp cho ESP32-S3
├── models/                  # Chứa cấu trúc baseline, mô hình lượng tử hóa TFLite và model_data.h
├── notebooks/               # Các script xử lý dữ liệu và huấn luyện mô hình
├── Colab_Training_TinyNav.ipynb # File Jupyter Notebook để huấn luyện trên Google Colab
├── collect_data.py          # Script lái tay thu thập hình ảnh mẫu
├── autonomous_drive.py      # Script chạy chế độ tự hành (Auto-Pilot)
├── clean_dataset.py         # Script kiểm tra và lọc bỏ ảnh hỏng truyền dẫn
└── README.md                # Hướng dẫn dự án
```

---

## 🚀 Hướng dẫn cài đặt và vận hành

### 1. Nạp chương trình cho ESP32-S3
*   Mở thư mục `esp32_firmware` bằng Arduino IDE.
*   Cài đặt thư viện `esp32` và cấu hình bo mạch là **ESP32S3 Dev Module** (Bật tính năng **PSRAM**).
*   Thay đổi thông tin WiFi (`ssid` và `password`) trong file `esp32_firmware.ino`.
*   Tiến hành biên dịch và nạp code lên mạch.

### 2. Thu thập dữ liệu lái tay (Data Collection)
Chạy script trên máy tính để kết nối với xe và lái tay (sử dụng các phím `W`, `A`, `S`, `D`):
```bash
python collect_data.py
```
*   Nhấn phím **`R`** để bắt đầu ghi hình tự động khi xe di chuyển.
*   Lái xe chạy khoảng 4-5 vòng sa bàn để thu thập ảnh (khoảng 1000 - 1500 ảnh).
*   Chạy script dọn dẹp để phân loại ảnh lỗi:
    ```bash
    python clean_dataset.py
    ```

### 3. Huấn luyện mô hình (Training Model)
#### Cách 1: Huấn luyện trên Google Colab (Khuyên dùng)
*   Chạy script tiền xử lý dữ liệu trên máy tính để đóng gói ảnh:
    ```bash
    python notebooks/preprocess_dataset.py
    ```
*   Tải file `Colab_Training_TinyNav.ipynb` lên Google Colab.
*   Tải lên file `preprocessed_data.npz` (trong thư mục `dataset`) và `model_baseline.keras` (trong thư mục `models`).
*   Bật GPU T4 và chạy huấn luyện để nhận file `best_model.keras` tải về máy.

#### Cách 2: Huấn luyện cục bộ trên CPU
```bash
python notebooks/train_model.py
```

### 4. Lượng tử hóa mô hình (Quantization)
Chuyển đổi mô hình `.keras` sang dạng TFLite 8-bit siêu nhẹ để tối ưu hóa tốc độ chạy:
```bash
python notebooks/quantize_model.py
```

### 5. Chạy tự hành (Autonomous Driving)
Đặt xe lên sa hình và chạy lệnh tự hành trên máy tính:
```bash
python autonomous_drive.py
```
*   Nhấp chọn cửa sổ camera và nhấn **Phím cách (Space)** để kích hoạt chế độ **`AUTO-PILOT ACTIVE`**.
*   Sử dụng phím **`+`** hoặc **`-`** để điều chỉnh tốc độ chạy thực tế cho mượt mà nhất.

---

## 💡 Các kỹ thuật tối ưu hóa trong dự án
*   **ROI Cropping (Cắt vùng quan tâm)**: Tự động loại bỏ 40% phần trên bức ảnh (chứa không gian phòng, nội thất) để chống hiện tượng học vẹt (overfitting), ép mô hình chỉ học vệt làn đường ngay trước mũi xe.
*   **Auto-Discovery**: Tự động phát hiện IP của máy tính qua cổng UDP Broadcast, giúp xe tự tìm thấy Laptop mà không cần phải cấu hình cứng địa chỉ IP tĩnh.
*   **Kick-start**: Xung kích khởi hành thông minh khi xe đi từ trạng thái dừng yên để thắng lực ma sát tĩnh ban đầu của động cơ ở dải tốc độ thấp.
*   **Pivot Turning**: Thuật toán khóa bánh trong cua và tăng tốc bánh ngoài giúp xe xoay ngoặt tại chỗ sắc nét, hạn chế tối đa việc lấn làn.
