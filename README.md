# Real-Time Skeleton-Based Fall Detection System

Hệ thống phát hiện hành vi té ngã theo thời gian thực dựa trên trích xuất khung xương người (**Skeleton-based Fall Detection**). Dự án tối ưu hóa hiệu năng bằng cách kết hợp mô hình thị giác **YOLOv8-Pose** để trích xuất đặc trưng động học và mạng hồi quy **LSTM** để phân loại chuỗi hành vi theo thời gian.

<p align="center">
<img width="854" height="480" alt="clideo_editor_ce9844098ea4427b865891ec40bc2605" src="https://github.com/user-attachments/assets/ff000220-d8a6-4f8d-9d22-b4f4cad3808d" />
</p>

[Video Demo](https://youtu.be/21NPpfityFI)

---

## 1. Mục tiêu dự án (Objective)

* **Xây dựng Pipeline AI hoàn chỉnh:** Có khả năng nhận diện hành vi té ngã (Fall Detection) từ luồng video trực tiếp (Camera Stream) với độ trễ cực thấp và độ chính xác cao.
* **Giải quyết bài toán thực tế:** Ứng dụng trong y tế, giám sát an toàn cho người cao tuổi hoặc bệnh nhân tại nhà và bệnh viện.
* **Bảo vệ quyền riêng tư (Privacy-preserving):** Thay vì truyền hoặc lưu trữ video gốc, hệ thống chỉ xử lý dữ liệu ma trận tọa độ khung xương (skeleton), tránh lộ thông tin cá nhân nhạy cảm và tối ưu hóa băng thông truyền tải trên các thiết bị Edge/Embedded.

---

## 2. Bộ dữ liệu huấn luyện (Dataset)

* **Nguồn gốc:** Bộ dữ liệu chuẩn hóa **IMVIA Le2i Fall Detection Dataset**.
* **Kênh khai thác:** Được khai thác qua phiên bản lưu trữ trên Kaggle: [FallDataset IMVIA](https://www.kaggle.com/).
* **Bối cảnh dữ liệu:** Video ghi hình các hành vi sinh hoạt thường ngày (ADL - Activities of Daily Living) và các pha té ngã mô phỏng tại nhiều không gian phòng khác nhau (`Coffee_room_01`, `Home_01`,...).

### Thống kê tập dữ liệu
| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
| :---: | :--- | :---: | :---: |
| **0** | ADL (Bình thường) | 7,628 | 75.6% |
| **1** | Fall (Té ngã) | 2,463 | 24.4% |
| **Tổng** | | **10,091** | **100%** |

> ⚠️ **Thách thức cốt lõi:** Sự chênh lệch **3:1** giữa hai lớp tạo ra bài toán **Imbalanced Classification** — đây là bài toán trung tâm cần giải quyết xuyên suốt pipeline xử lý.

---

## 3. Phương pháp thực hiện (Methodology)

### 3.1. Chuẩn hóa không gian (Spatial Normalization)
Tọa độ 17 keypoints từ YOLOv8-Pose được dịch chuyển gốc tọa độ về **tâm hông (hip center)**, sau đó chia cho chiều cao động của cơ thể trong frame. Kỹ thuật này giúp vector khung xương **bất biến với khoảng cách camera** — người đứng gần hay xa đều cho ra đặc trưng đồng nhất.

### 3.2. Trích xuất đặc trưng động học (Kinematic Feature Engineering)
Thay vì chỉ dùng tọa độ tĩnh $(x, y)$, hệ thống tính thêm vi phân bậc 1 và bậc 2 theo thời gian để nắm bắt hành vi động:

| Loại đặc trưng | Cách tính | Số lượng đặc trưng |
| :--- | :--- | :---: |
| **Vị trí** | Tọa độ gốc $(x, y)$ | 17 × 2 = **34** |
| **Vận tốc** | Sai phân bậc 1 (`diff` bậc 1) | 17 × 2 = **34** |
| **Gia tốc** | Sai phân bậc 2 (`diff` bậc 2) | 17 × 2 = **34** |
| **Tổng đầu vào** | | **102 đặc trưng / frame** |

*Ý nghĩa:* Giúp mô hình phân biệt được hành vi *"ngồi xuống nhanh"* (gia tốc đều, có kiểm soát) với *"ngã quỵ đột ngột"* (gia tốc đột biến, mất kiểm soát).

### 3.3. Xử lý mất cân bằng nhãn (Imbalanced Data Handling)
* **Focal Loss ($\alpha=0.75, \gamma=2.0$):** Thay thế cho Binary Cross Entropy truyền thống. Cơ chế tự động hạ trọng số các ca dễ (ADL chiếm đa số) và tập trung gradient vào các ca ngã khó, buộc model phải học đặc trưng té ngã thay vì đoán mò toàn bộ là bình thường.
* **Skeleton Jittering:** Bơm nhiễu Gaussian $\mathcal{N}(0, 0.02)$ ngẫu nhiên vào các mẫu `Fall` trong quá trình huấn luyện, tạo ra các biến thể tư thế đa dạng qua mỗi epoch, tăng khả năng tổng quát hóa.

### 3.4. Chiến lược lựa chọn mô hình (Model Selection Pipeline)
Quá trình thực nghiệm được chia làm **2 giai đoạn độc lập**:

* **Giai đoạn 1 — Lựa chọn kiến trúc (5-Fold Cross Validation):** Đánh giá khách quan 3 kiến trúc trên toàn bộ dữ liệu bằng phương pháp *Stratified K-Fold* (giữ nguyên tỷ lệ nhãn ở mỗi fold):
    * **LSTM:** Bộ nhớ dài hạn, phù hợp chuỗi thời gian có phụ thuộc xa.
    * **GRU:** Biến thể nhẹ hơn LSTM, tối ưu tốc độ hội tụ.
    * **CNN-1D:** Trích xuất đặc trưng cục bộ theo trục thời gian.
    * *Tiêu chí chọn:* Kiến trúc sở hữu **Mean Recall cao nhất** sẽ đi tiếp vào vòng trong.

* **Giai đoạn 2 — Tối ưu siêu tham số (Grid Search):** Thực hiện tối ưu hóa cấu hình trên kiến trúc chiến thắng thông qua 3 trục không gian tham số:
    * `hidden_size`: `[32, 64]`
    * `num_layers`: `[1, 2]`
    * `lr` (Learning Rate): `[0.001, 0.005]`
    * *Quy mô:* Gồm 8 tổ hợp, mỗi tổ hợp chạy tối đa 35 epoch kết hợp *Early Stopping*. Model cuối cùng sẽ được thử thách trên **Test set độc lập hoàn toàn** với ngưỡng phân loại (threshold) tối ưu xác định bởi **Youden Index** trên Validation set.

---

## 4. Kết quả thực nghiệm (Experimental Results)

### 4.1. So sánh hiệu năng 5-Fold Cross Validation
| Kiến trúc Mô hình | Mean Accuracy | Mean Recall | Mean F1-Score | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM (Optimized)** | **93.91%** | **85.71%** | **87.29%** | **Được chọn** |
| **GRU** | 93.44% | 84.69% | 86.29% | Loại |
| **CNN-1D (TCN)** | 89.24% | 72.96% | 76.73% | Loại |

### 4.2. Kết quả cấu hình tối ưu từ Grid Search
| Tham số | Giá trị tối ưu |
| :--- | :---: |
| **Hidden Size** | 64 |
| **Num Layers** | 1 |
| **Learning Rate** | 0.001 |

### 4.3. Đánh giá trên tập Test độc lập hoàn toàn
Sau khi cấu hình ngưỡng quyết định tối ưu bằng Youden Index tại **Threshold = 0.3415** (thấp hơn mức 0.5 mặc định để ưu tiên bắt ca Fall trong y tế), mô hình đạt kết quả:

| Chỉ số | Giá trị | Ý nghĩa thực tế |
| :--- | :---: | :--- |
| **Accuracy** | **90.03%** | 90/100 mẫu được phân loại chính xác |
| **Recall (Fall)** | **90.00%** | Bắt trúng đúng 9/10 ca té ngã thực tế |
| **F1-Score** | **81.52%** | Đạt trạng thái cân bằng tốt giữa Precision và Recall |

---

## 5. Phân tích sâu & Đánh giá (Analysis)

### 5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)

<p align="center">
  <img src="plots/lstm_normalized_confusion_matrix.png" alt="Confusion Matrix" width="600">
</p>

* Hai đường chéo đạt tỷ lệ đối xứng lý tưởng **(0.90 / 0.90)**, cho thấy mô hình không bị lệch hay thiên vị về bất kỳ lớp nào dù tập dữ liệu gốc mất cân bằng nghiêm trọng (3:1). Điều này chứng minh sự kết hợp giữa *Focal Loss*, *Skeleton Jittering* và kỹ thuật tối ưu ngưỡng dịch chuyển đã phát huy tác dụng triệt để.
* *Góc nhìn thực tế:* Tỷ lệ **False Negative là 10%** (bỏ sót 1/10 ca ngã). Trong bối cảnh y tế, đây là chỉ số cần tiếp tục tối ưu ở các phiên bản tiếp theo bằng cách bổ sung thêm đặc trưng về góc xoay của các khớp xương hoặc làm giàu thêm dữ liệu thực tế.

### 5.2. Biểu đồ lịch sử huấn luyện (Training History Analysis)

<p align="center">
  <img src="plots/lstm_training_history.png" alt="Training History Analysis" width="600">
</p>

#### a. Biểu đồ đường Loss
* **Độ hội tụ:** Quá trình diễn ra ổn định và đúng hướng. `Train Loss` giảm mượt mà từ ~0.076 xuống ~0.008 ở epoch 50, chứng tỏ mạng mạng học sâu trích xuất được các đặc trưng mang tính bản chất cao.
* **Hiện tượng Overfitting nhẹ:** Từ epoch 1-25, `Val Loss` giảm song song với Train Loss xuống ~0.034. Tuy nhiên từ epoch 25-50 bước vào giai đoạn dao động và nhích nhẹ lên ~0.055. Cơ chế **Early Stopping** đã can thiệp kịp thời giúp ngắt huấn luyện ở epoch 50 và khôi phục trọng số tốt nhất ở **epoch 38** (điểm cực trị của Val Score).
* *Lưu ý:* Khoảng cách cuối cùng giữa Train/Val Loss chênh lệch khoảng **5.25 lần** hoàn toàn giải thích được do tập Train được áp dụng kỹ thuật tăng cường dữ liệu (*Skeleton Jittering*) tạo nhiễu trong khi tập Val giữ nguyên dạng gốc sạch sẽ.

#### b. Biểu đồ độ chính xác (Accuracy)
* `Train Accuracy` tăng tiến đều đặn không có điểm gãy từ 83% lên **98.3%**.
* `Val Accuracy` bứt phá nhanh ở 10 epoch đầu (83% $\rightarrow$ 92%), sau đó dao động ổn định trong biên độ hẹp **92% - 94.5%**. Sự rung lắc nhẹ $\pm 1-2\%$ này đến từ việc cài đặt kích thước `batch_size` nhỏ (=8) trên một tập validation giới hạn.
* Điểm mấu chốt là `Val Accuracy` **không hề sụt giảm** khi bước qua epoch 25 (dù Val Loss nhích lên). Điều này chứng tỏ mô hình chỉ suy giảm nhẹ về độ tự tin của xác suất phân phối chứ hoàn toàn vững chắc về mặt phân loại nhãn quyết định, minh chứng trực quan là điểm Accuracy và Recall trên tập Test độc lập vẫn duy trì xuất sắc ở mức **90%**.
