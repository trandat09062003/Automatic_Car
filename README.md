# TinyNav - Hệ thống xe tự hành bám làn thời gian thực sử dụng ESP32-S3 & CNN

Dự án này tập trung thiết kế và triển khai hệ thống điều khiển xe robot tự hành bám làn (Lane Following) khép kín trong thời gian thực. ESP32-S3 đóng vai trò thu nhận hình ảnh từ camera OV2640, truyền stream JPEG qua giao thức UDP về máy tính. Tại máy tính, mô hình học sâu CNN (chạy trực tiếp Keras hoặc tối ưu qua TFLite 8-bit) sẽ dự đoán góc lái và phản hồi lệnh điều khiển động cơ qua UDP ngược lại cho xe để giữ xe luôn chạy đúng làn đường.

---

## 🛠️ Kiến trúc hệ thống & Đấu nối phần cứng

### 1. Sơ đồ luồng dữ liệu (Data Flow)
```
[ESP32-S3 Camera] ---> (Truyền JPEG qua UDP Stream) ---> [Laptop (Suy luận AI)]
        ^                                                         |
        |                                                         v
[Điều khiển động cơ] <--- (Lệnh điều khiển dạng chuỗi) <--- [Bộ điều khiển góc lái]
```

### 2. Thiết bị phần cứng
*   **Mạch điều khiển chính**: ESP32-S3 Dev Module hoặc ESP32-S3 Eye (Yêu cầu mạch có tích hợp **PSRAM** để xử lý và lưu trữ bộ đệm ảnh camera).
*   **Camera**: OV2640 (Cấu hình truyền ảnh JPEG, kích thước QVGA 320x240, tần số truyền quét ~30 FPS).
*   **Động cơ & Mạch lái**: Khung xe robot 2 bánh chủ động (2WD) + 1 bánh dẫn hướng phía trước. Sử dụng mạch cầu H L298N (hoặc dòng Driver tương đương) để điều khiển động cơ DC bằng tín hiệu xung PWM.
*   **Nguồn điện**: Sử dụng pin sạc 18650 (khuyến nghị dùng nguồn riêng cho vi điều khiển và nguồn riêng cho động cơ để tránh hiện tượng sụt áp gây nhiễu và tự khởi động lại ESP32).

### 3. Sơ đồ nối dây (Pinout)
Dựa theo cấu hình mặc định trong mã nguồn C++ (`esp32_firmware.ino`):

| Chân trên ESP32-S3 | Chân trên mạch cầu H L298N / Động cơ | Ghi chú |
| :--- | :--- | :--- |
| **GPIO 42** | Chân PWM điều khiển Motor A (Trái) | Điều khiển tốc độ bánh bên trái |
| **GPIO 41** | Chân PWM điều khiển Motor B (Phải) | Điều khiển tốc độ bánh bên phải |
| **GND** | GND nguồn chung | Phải nối chung đất giữa ESP32 và Driver L298N |
| **5V / VIN** | Nguồn cấp cho vi điều khiển | Đảm bảo dòng cấp ổn định |

*Lưu ý:* Mã nguồn sử dụng phương pháp điều khiển tốc độ vi sai thông qua 2 kênh PWM độc lập để chuyển hướng (rẽ trái/phải bằng cách giảm tốc độ của bánh phía rẽ hoặc đảo tốc độ tương đối giữa hai bánh).

---

## 📂 Danh mục mã nguồn dự án

*   `esp32_firmware/`: Mã nguồn Arduino C++ nạp cho ESP32-S3, xử lý đa nhân FreeRTOS (Core 1 chụp & truyền stream ảnh; Core 0 nhận gói tin UDP điều khiển động cơ).
*   `dataset/`: Thư mục lưu trữ bộ ảnh thô và tệp dữ liệu sau khi tiền xử lý (`preprocessed_data.npz`).
*   `models/`: Chứa mô hình cơ sở (`model_baseline.keras`), mô hình huấn luyện tối ưu (`best_model.keras`), mô hình lượng tử hóa (`tiny_nav_quantized.tflite`) và file header C (`model_data.h`) để nạp trực tiếp vào code C++ nếu chạy suy luận onboard.
*   `notebooks/`:
    *   `design_model.py`: Thiết kế cấu trúc mô hình CNN.
    *   `preprocess_dataset.py`: Tiền xử lý, ghép sliding window 16 frame liên tục và phân bổ tập Train/Val/Test.
    *   `train_model.py`: Huấn luyện mô hình cục bộ tích hợp các thuật toán tăng cường dữ liệu (Augmentation).
    *   `quantize_model.py`: Lượng tử hóa mô hình sang INT8 và xuất ra file nhị phân cho C/Rust.
*   `collect_data.py`: Script điều khiển lái xe thủ công qua bàn phím máy tính để thu thập dữ liệu ảnh chạy thực tế.
*   `clean_dataset.py`: Quét lọc và di chuyển các ảnh bị lỗi truyền nhận UDP (thiếu byte, hỏng định dạng JPEG) sang thư mục `loi`.
*   `autonomous_drive.py`: Chương trình điều khiển xe tự hành thời gian thực chạy trên máy tính.

---

## 🚀 Hướng dẫn cài đặt và vận hành hệ thống

### 1. Chuẩn bị môi trường phần mềm trên PC
Cài đặt Python (khuyến nghị phiên bản 3.10) và cài các thư viện bổ trợ bằng lệnh:
```bash
pip install opencv-python tensorflow numpy scikit-learn pillow
```

### 2. Cấu hình và nạp code cho ESP32-S3
1. Mở file `esp32_firmware/esp32_firmware.ino` bằng phần mềm Arduino IDE.
2. Cài đặt board hỗ trợ `esp32` (phiên bản 2.0.x trở lên).
3. Trong phần thiết lập board, cấu hình:
   *   Board: **ESP32S3 Dev Module** (hoặc ESP32S3 Eye).
   *   **PSRAM**: **Enabled** (chọn chế độ thích hợp OPI hoặc QSPI tùy theo linh kiện mạch của bạn).
4. Thay đổi thông tin mạng WiFi trong code:
   ```cpp
   const char *ssid = "Tên_WiFi_Của_Bạn";
   const char *password = "Mật_Khẩu_WiFi";
   ```
5. Tiến hành biên dịch và nạp code lên mạch. Mở Serial Monitor với tốc độ **115200 baud** để kiểm tra trạng thái kết nối mạng và lấy địa chỉ IP của ESP32.

### 3. Thu thập dữ liệu chạy thử (Lái tay)
Đặt xe lên sa hình, khởi động và kết nối máy tính vào cùng mạng WiFi với ESP32. Chạy script để lái xe thủ công:
```bash
python collect_data.py
```
*   **W** (hoặc mũi tên lên): Đi thẳng.
*   **A** (hoặc mũi tên trái): Rẽ trái (Bánh ngoài xoay mạnh, bánh trong dừng hẳn để cua xoáy mượt).
*   **D** (hoặc mũi tên phải): Rẽ phải.
*   **S** (hoặc phím cách): Dừng xe khẩn cấp.
*   **R**: Bật/Tắt chế độ tự động lưu ảnh khi xe di chuyển.
*   **Q**: Thoát chương trình điều khiển.

*Kinh nghiệm thu thập:* Hãy lái xe đi mượt mà ở giữa vệt làn đường khoảng 4 - 5 vòng sa hình để có bộ dữ liệu chuẩn chỉnh nhất. Tránh việc đâm đụng hoặc lệch làn quá nhiều khi đang bật ghi dữ liệu.

### 4. Tiền xử lý dữ liệu và lọc lỗi truyền dẫn
Sau khi thu thập xong, trong thư mục `dataset/` sẽ chứa rất nhiều ảnh. Do ảnh được truyền không dây qua UDP nên có thể xảy ra tình trạng mất gói, ảnh bị lỗi byte.

1. **Lọc ảnh hỏng**:
   ```bash
   python clean_dataset.py
   ```
   Script sẽ tự động quét thư mục `dataset/`, dùng thư viện PIL và OpenCV để kiểm tra tính toàn vẹn của từng ảnh. Ảnh lỗi sẽ tự động được gom sang thư mục `loi/` độc lập để tránh gây crash khi train.
   
2. **Đóng gói dữ liệu chuỗi**:
   ```bash
   python notebooks/preprocess_dataset.py
   ```
   Script này sẽ phân tích mốc thời gian của từng ảnh để gom nhóm thành các đợt chạy liên tục (session), sau đó tạo một **sliding window có kích thước 16 frame liên tục** (tương ứng tensor đầu vào có chiều sâu 16 kênh xám 128x128). Dữ liệu sau đó được chia theo tỷ lệ **70% Train - 20% Val - 10% Test** rồi nén thành file `dataset/preprocessed_data.npz`.

### 5. Huấn luyện mô hình CNN
Mô hình CNN được thiết kế nhận đầu vào dạng chuỗi 16 frame để học cả động học chuyển động của xe (tránh hiện tượng mất dấu làn tạm thời).

*   **Cách 1 (Chạy trên máy cá nhân)**:
    ```bash
    python notebooks/train_model.py
    ```
    Script này cấu hình mặc định ép chạy trên CPU (`CUDA_VISIBLE_DEVICES = -1`) để chạy ổn định trên các máy không có GPU rời.
    
*   **Cách 2 (Huấn luyện qua Google Colab - Khuyến nghị)**:
    Tải file notebook `Colab_Training_TinyNav.ipynb` lên Google Colab, chuyển Runtime sang **T4 GPU**, sau đó tải lên 2 file `preprocessed_data.npz` (trong `dataset/`) và `model_baseline.keras` (trong `models/`). Chạy toàn bộ notebook để huấn luyện với tốc độ cao.
    
Sau khi hoàn tất huấn luyện, tải file `best_model.keras` kết quả về và lưu vào thư mục `models/`.

### 6. Lượng tử hóa mô hình sang dạng TFLite 8-bit
Để mô hình suy luận nhẹ hơn và mượt mà hơn trên máy tính (hoặc nạp xuống vi điều khiển), thực hiện lượng tử hóa tĩnh (Post-Training Quantization):
```bash
python notebooks/quantize_model.py
```
Script sẽ sinh ra mô hình lượng tử hóa `models/tiny_nav_quantized.tflite` (dung lượng giảm đáng kể), đồng thời tự động xuất ra file header C (`models/model_data.h`) và Rust slice (`esp32-rust/src/model_data.rs`) phục vụ mục đích nhúng.

### 7. Chạy xe tự hành (Auto-Pilot)
Đặt xe ở vạch xuất phát trên sa hình, khởi động xe và chạy script trên máy tính:
```bash
python autonomous_drive.py
```
*   Giao diện hiển thị HUD của camera xe sẽ xuất hiện trên màn hình máy tính.
*   Click chuột vào màn hình HUD và nhấn **Phím cách (Space)** để kích hoạt chế độ **Auto-Pilot**.
*   Nhấn phím `+` hoặc `-` trên bàn phím để tinh chỉnh tăng/giảm tốc độ nền (`BASE_SPEED`) của xe cho khớp với ma sát thực tế của mặt đường.
*   Nhấn phím **S** hoặc **Space** bất kỳ lúc nào để chuyển lại về chế độ điều khiển bằng tay (hoặc phanh khẩn cấp dừng xe).

---

## 💡 Các giải pháp kỹ thuật tối ưu hóa trong dự án

*   **ROI Crop (Region of Interest)**: Hệ thống tự động cắt bỏ 40% phần trên của ảnh thu được từ camera (bỏ qua bối cảnh trần nhà, người đi lại, thiết bị xung quanh). Điều này giúp mô hình CNN tránh hiện tượng học vẹt bối cảnh ngoại cảnh mà chỉ tập trung học hình dáng của vệt đường đi phía trước.
*   **UDP Auto-Discovery (Dò tìm IP tự động)**: Xe gửi gói tin quảng bá (UDP Broadcast) chứa nội dung `"HB"` (Heartbeat) ra toàn mạng. Máy tính khi nhận được gói tin này sẽ tự động biết được IP của xe và gửi phản hồi lệnh điều khiển về đúng IP đó. Cơ chế này giúp vận hành xe linh hoạt ở các mạng WiFi khác nhau mà không cần sửa code nạp lại firmware.
*   **Proportional Steering Control (Bộ điều khiển lái tỷ lệ)**: Xe điều chỉnh tốc độ bánh xe vi sai tỉ lệ tuyến tính trực tiếp theo độ lớn góc lái dự đoán của AI. Khi xe lệch làn ít, chênh lệch tốc độ 2 bánh nhỏ giúp ôm cua mượt; khi cua gắt, chênh lệch lớn giúp xe bám cua nhanh, giải quyết triệt để hiện tượng vẫy đuôi giật cục của động cơ DC.
*   **Auto Kick-start (Xung kích đề ba)**: Để thắng lực ma sát tĩnh ban đầu khi xe xuất phát từ trạng thái dừng hẳn ở mức ga thấp (PWM thấp dễ gây kẹt động cơ), script điều khiển tự động phát xung lực mạnh (PWM 130 trong 120ms) để đẩy xe di chuyển trước, sau đó mới đưa về tốc độ nền đã cấu hình.
*   **Đa nhiệm đa nhân trên ESP32**: Sử dụng hệ điều hành thời gian thực FreeRTOS, lập trình chia Task chạy song song: `SendImageTask` (chụp & gửi ảnh) ghim chạy trên Core 1; `RecvMsgTask` (nhận gói điều khiển động cơ) chạy trên Core 0. Điều này đảm bảo việc truyền nhận dữ liệu không bị nghẽn hay giật lag.
