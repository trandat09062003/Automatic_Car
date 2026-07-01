# TinyNav - Hệ thống xe tự hành bám làn sử dụng ESP32-S3 và mạng Custom CNN

Dự án này phát triển hệ thống điều khiển xe robot tự hành bám làn (Lane Following) khép kín thời gian thực. Hệ thống thu nhận hình ảnh từ camera trên mạch ESP32-S3, truyền stream qua giao thức UDP về máy tính để chạy mô hình học sâu CNN dự đoán góc lái phù hợp, sau đó phản hồi lệnh điều khiển động cơ ngược lại cho xe.

---

## 🛠️ Cấu trúc hệ thống

### 1. Phần cứng
*   **Mạch điều khiển**: ESP32-S3 Dev Module (Yêu cầu bật tính năng PSRAM để xử lý ảnh camera).
*   **Camera**: OV2640 (Cấu hình luồng truyền ảnh JPEG kích thước QVGA 320x240).
*   **Động cơ & Mạch lái**: Khung xe robot 2WD kết hợp mạch cầu H L298N (hoặc tương đương) điều khiển qua chân PWM.
*   **Nguồn điện**: Sử dụng nguồn độc lập cho vi điều khiển và động cơ để hạn chế nhiễu sụt áp.

### 2. Sơ đồ luồng dữ liệu (Data Flow)
```
[ESP32-S3 Camera] ---> (UDP Stream JPEG) ---> [Laptop (AI Inference)]
        ^                                              |
        |                                              v
[Điều khiển động cơ] <--- (Lệnh điều khiển) <--- [Bộ tính toán hướng lái]
```

---

## 📂 Các tệp nguồn chính

*   `esp32_firmware/`: Mã nguồn Arduino C++ nạp cho ESP32-S3.
*   `models/`: Chứa các mô hình huấn luyện (`model_baseline.keras`, `best_model.keras`), mô hình lượng tử hóa (`tiny_nav_quantized.tflite`) và mảng nhị phân xuất cho firmware (`model_data.h`).
*   `notebooks/`: Các script Python phục vụ việc thiết kế, tiền xử lý và huấn luyện mô hình.
*   `Colab_Training_TinyNav.ipynb`: Notebook Jupyter được tối ưu cấu hình để huấn luyện mô hình trên Google Colab GPU.
*   `collect_data.py`: Script điều khiển lái xe thủ công qua bàn phím và ghi hình ảnh làm bộ dữ liệu.
*   `autonomous_drive.py`: Script chính để chạy chế độ tự hành (Auto-Pilot).
*   `clean_dataset.py`: Công cụ lọc và phân loại các ảnh bị lỗi truyền dẫn.

---

## 🚀 Hướng dẫn vận hành hệ thống

### 1. Cài đặt cho ESP32-S3
1. Mở mã nguồn `esp32_firmware/esp32_firmware.ino` bằng Arduino IDE.
2. Cài đặt thư viện bo mạch `esp32` (phiên bản khuyến nghị từ 2.0.x trở lên).
3. Cấu hình bo mạch: **ESP32S3 Dev Module**, kích hoạt **PSRAM** ở chế độ thích hợp (OPI/QSPI).
4. Khai báo thông tin mạng WiFi (`ssid` và `password`).
5. Nạp chương trình lên mạch ESP32-S3.

### 2. Thu thập dữ liệu
Chạy lệnh dưới đây để kết nối với xe và lái tay bằng các phím `W`, `A`, `S`, `D` để lấy mẫu hành trình:
```bash
python collect_data.py
```
*   Nhấn phím `R` trên giao diện camera để bật/tắt ghi dữ liệu tự động khi xe chạy.
*   Cố gắng lái xe chạy mượt mà ở giữa vệt làn xám khoảng 4 - 5 vòng sa hình.
*   Sau khi thu thập xong, chạy script để lọc ảnh hỏng và phân loại sang thư mục `loi`:
    ```bash
    python clean_dataset.py
    ```

### 3. Huấn luyện mô hình
Để huấn luyện mô hình nhanh nhất bằng GPU, hãy đóng gói dữ liệu trên máy tính:
```bash
python notebooks/preprocess_dataset.py
```
Sau đó, import file `Colab_Training_TinyNav.ipynb` lên Google Colab, tải lên file nén dữ liệu `preprocessed_data.npz` và file baseline `model_baseline.keras` để tiến hành huấn luyện. Tải file `best_model.keras` kết quả về máy và đặt vào thư mục `models/`.

### 4. Lượng tử hóa mô hình sang dạng TFLite 8-bit
Chạy script lượng tử hóa để tối ưu hóa mô hình phục vụ suy luận thời gian thực:
```bash
python notebooks/quantize_model.py
```

### 5. Chạy xe tự hành
Đặt xe vào sa hình, chạy lệnh tự lái trên máy tính:
```bash
python autonomous_drive.py
```
Nhấp chọn màn hình camera HUD và nhấn **Phím cách (Space)** để bắt đầu tự lái. Sử dụng phím `+` hoặc `-` để tinh chỉnh tốc độ thẳng cơ bản phù hợp với ma sát mặt đường.

---

## 💡 Các điểm cải tiến kỹ thuật trong dự án
*   **ROI Crop**: Tự động cắt bỏ 40% phần trên của ảnh chụp (loại bỏ không gian phòng) để tránh hiện tượng học vẹt đặc trưng ngoại cảnh, ép mô hình CNN chỉ học thông tin vệt đường đi phía trước.
*   **UDP Auto-Discovery**: Sử dụng cơ chế gửi gói tin quảng bá (UDP Broadcast) để xe tự nhận biết IP của máy tính khi đổi mạng WiFi mà không cần sửa code nạp lại mạch.
*   **Proportional Steering Control**: Áp dụng bộ điều khiển tỷ lệ tuyến tính dựa trên góc bẻ lái dự đoán của AI để điều tiết tốc độ hai bánh, giúp xe ôm cua mượt mà và tránh dao động giật cục.
*   **Auto Kick-start**: Tự động phát xung lực đẩy mạnh khi bắt đầu khởi hành từ trạng thái đứng yên để khắc phục lực ma sát tĩnh ở dải tốc độ thấp.
