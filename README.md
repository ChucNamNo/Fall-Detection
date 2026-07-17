# 🏃‍♂️ Real-Time Skeleton-Based Fall Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-v8.0-00FFFFFF?logo=ultralytics&logoColor=white" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/Architecture-BiGRU--Attention-darkorange" alt="Architecture">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

Hệ thống phát hiện hành vi té ngã theo thời gian thực dựa trên trích xuất khung xương người (**Skeleton-based Fall Detection**). Dự án tối ưu hóa hiệu năng bằng cách kết hợp mô hình thị giác **YOLOv8-Pose** để trích xuất đặc trưng động học và mạng hồi quy kết hợp cơ chế chú ý **BiGRU-Attention** để phân loại chuỗi hành vi theo thời gian.

<p align="center">
  <img width="536" height="344" alt="Fall Detection Demo Snapshot" src="https://github.com/user-attachments/assets/5c53e040-2c88-4d5f-bd3e-b22d6ba24824" />
</p>

<p align="center">
  <a href="https://youtu.be/JisHb_Q-2x8" target="_blank">
    <img src="https://img.shields.io/badge/🎥%20Video%20Demo-Watch%20on%20YouTube-red?style=for-the-badge&logo=youtube" alt="Video Demo">
  </a>
</p>

---

## Mục lục
- [1. Mục tiêu dự án (Objective)](#1-mục-tiêu-dự-án-objective)
- [2. Bộ dữ liệu huấn luyện (Dataset)](#2-bộ-dữ-liệu-huấn-luyện-dataset)
  - [2.1. Chiến lược phân chia dữ liệu (Data Splitting Strategy)](#21-chiến-lược-phân-chia-dữ-liệu-data-splitting-strategy)
- [3. Phương pháp thực hiện (Methodology)](#3-phương-pháp-thực-hiện-methodology)
  - [3.1. Chuẩn hóa không gian (Spatial Normalization)](#31-chuẩn-hóa-không-gian-spatial-normalization)
  - [3.2. Trích xuất đặc trưng động học (Kinematic Feature Engineering)](#32-trích-xuất-đặc-trưng-động-học-kinematic-feature-engineering)
  - [3.3. Xử lý mất cân bằng nhãn (Imbalanced Data Handling)](#33-xử-lý-mất-cân-bằng-nhãn-imbalanced-data-handling)
  - [3.4. Chiến lược lựa chọn mô hình (Model Selection Pipeline)](#34-chiến-lược-lựa-chọn-mô-hình-model-selection-pipeline)
- [4. Kết quả thực nghiệm (Experimental Results)](#4-kết-quả-thực-nghiệm-experimental-results)
  - [4.1. So sánh hiệu năng 5-Fold Cross Validation](#41-so-sánh-hiệu-năng-5-fold-cross-validation)
  - [4.2. Kết quả cấu hình tối ưu từ Grid Search](#42-kết-quả-cấu-hình-tối-ưu-từ-grid-search)
  - [4.3. Đánh giá trên tập Test độc lập hoàn toàn](#43-đánh-giá-trên-tập-test-độc-lập-hoàn-toàn)
- [5. Phân tích sâu & Đánh giá (Analysis)](#5-phân-tích-sâu--đánh-giá-analysis)
  - [5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)](#51-ma-trận-nhầm-lẫn-confusion-matrix-analysis)
  - [5.2. Biểu đồ lịch sử huấn luyện (Training History Analysis)](#52-biểu-đồ-lịch-sử-huấn-luyện-training-history-analysis)

---

## 1. Mục tiêu dự án (Objective)

* **Xây dựng Pipeline AI hoàn chỉnh:** Có khả năng nhận diện hành vi té ngã (Fall Detection) từ luồng video trực tiếp (Camera Stream) với độ trễ cực thấp và độ chính xác cao.
* **Giải quyết bài toán thực tế:** Ứng dụng trong y tế, giám sát an toàn cho người cao tuổi hoặc bệnh nhân tại nhà và bệnh viện.
* **Bảo vệ quyền riêng tư (Privacy-preserving):** Thay vì truyền hoặc lưu trữ video gốc, hệ thống chỉ xử lý dữ liệu ma trận tọa độ khung xương (skeleton), tránh lộ thông tin cá nhân nhạy cảm và tối ưu hóa băng thông truyền tải trên các thiết bị Edge/Embedded.

---

## 2. Bộ dữ liệu huấn luyện (Dataset)

* **Nguồn gốc:** Bộ dữ liệu chuẩn hóa **IMVIA Le2i Fall Detection Dataset**.
* **Kênh khai thác:** Được khai thác qua phiên bản lưu trữ trên Kaggle: [FallDataset IMVIA](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data).
* **Bối cảnh dữ liệu:** Video ghi hình các hành vi sinh hoạt thường ngày (ADL - Activities of Daily Living) và các pha té ngã mô phỏng tại nhiều không gian phòng khác nhau (`Coffee_room_01`, `Home_01`,...).

### Thống kê tập dữ liệu
| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
| :---: | :--- | :---: | :---: |
| **0** | ADL (Bình thường) | 7,628 | 75.6% |
| **1** | Fall (Té ngã) | 2,463 | 24.4% |
| **Tổng** | | **10,091** | **100%** |

> [!WARNING]
> **Thách thức cốt lõi:** Sự chênh lệch **3:1** giữa hai lớp tạo ra bài toán *Imbalanced Classification* — đây là bài toán trung tâm cần giải quyết xuyên suốt pipeline xử lý.

### 2.1. Chiến lược phân chia dữ liệu (Data Splitting Strategy)

Toàn bộ bộ dữ liệu gồm **10,091 mẫu** được phân chia theo tỷ lệ vàng **70 : 15 : 15** bằng phương pháp phân tầng (**Stratified Split**). Phương pháp này giúp giữ nguyên tỷ lệ lệch nhãn ~3:1 (ADL vs Fall) đồng đều trên cả 3 tập dữ liệu:

* **Tập Huấn luyện (Train Set - 70%):** Dùng để cập nhật trọng số cho mô hình BiGRU-Attention kết hợp kỹ thuật tăng cường dữ liệu *Skeleton Jittering*.
* **Tập Kiểm thử nội bộ (Validation Set - 15%):** Dùng để theo dõi quá trình hội tụ, kích hoạt cơ chế dừng sớm (*Early Stopping*) và tìm ngưỡng quyết định tối ưu (*Optimal Threshold*) bằng Youden Index.
* **Tập Kiểm thử độc lập (Test Set - 15%):** Hoàn toàn cô lập trong suốt quá trình huấn luyện và tối ưu tham số, chỉ được sử dụng một lần duy nhất để đánh giá hiệu năng thực tế cuối cùng của hệ thống.

#### Bảng thống kê chi tiết số lượng mẫu sau phân chia:

| Tập dữ liệu | Tỷ lệ phân chia | Số mẫu lớp ADL (Nhãn 0) | Số mẫu lớp Fall (Nhãn 1) | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 70.0% | 5,340 | 1,724 | **7,064** |
| **Validation Set** | 15.0% | 1,144 | 369 | **1,513** |
| **Test Set** | 15.0% | 1,144 | 370 | **1,514** |
| **Tổng cộng** | **100%** | **7,628** | **2,463** | **10,091** |

---

## 3. Phương pháp thực hiện (Methodology)

### 3.1. Chuẩn hóa không gian (Spatial Normalization)
Tọa độ 17 keypoints từ YOLOv8-Pose được dịch chuyển gốc tọa độ về **tâm hông (hip center)**, sau đó chia cho chiều cao động của cơ thể trong frame. Kỹ thuật này giúp vector khung xương **bất biến với khoảng cách camera** — người đứng gần hay xa đều cho ra đặc trưng đồng nhất.

### 3.2. Trích xuất đặc trưng động học (Kinematic Feature Engineering)
Thay vì chỉ sử dụng tọa độ tĩnh `(x, y)`, hệ thống tiến hành tính toán thêm vi phân bậc 1 (Vận tốc) và bậc 2 (Gia tốc) theo thời gian để nắm bắt toàn diện hành vi động của đối tượng qua từng khung hình.

> [!NOTE]
> **Cơ chế Đệm chuỗi (Padding Strategy)**
> Để xử lý các chuỗi video có độ dài ngắn hơn `max_len = 16`, hệ thống áp dụng kỹ thuật **Pre-padding bằng cách lặp lại frame đầu tiên (Replicate Padding)** thay vì đệm số 0 (Zero-padding) ở cuối chuỗi nhằm:
> 1. **Triệt tiêu nhiễu động học:** Loại bỏ hoàn toàn hiện tượng biến động đột biến của vận tốc và gia tốc tại vùng ranh giới đệm.
> 2. **Bảo toàn thông tin cuối:** Đảm bảo thông tin thực tế cuối cùng của chuỗi luôn nằm ở bước thời gian cuối cùng (`t = -1`) khi đưa vào mạng học sâu, giúp mô hình ra quyết định chính xác nhất.

#### Chi tiết cấu trúc dữ liệu đầu vào (Input Tensor Shape)

| Loại đặc trưng | Phương thức tính toán | Số lượng đặc trưng | Kích thước (Shape) |
| :--- | :--- | :---: | :---: |
| **Vị trí** | Tọa độ gốc `(x, y)` của 17 keypoints | 17 × 2 = **34** | `(16, 34)` |
| **Vận tốc** | Sai phân bậc 1 (`diff` bậc 1) theo trục thời gian | 17 × 2 = **34** | `(16, 34)` |
| **Gia tốc** | Sai phân bậc 2 (`diff` bậc 2) theo trục thời gian | 17 × 2 = **34** | `(16, 34)` |
| **Tổng đầu vào** | **Nối đặc trưng (Concatenation)** | **102 đặc trưng** | **`(16, 102)`** |

---

### 3.3. Xử lý mất cân bằng nhãn (Imbalanced Data Handling)

#### 1. Focal Loss chuẩn hóa ($\alpha = 0.75, \gamma = 2.0$)
Thay thế hoàn toàn cho hàm Binary Cross Entropy (BCE) truyền thống. Focal Loss tự động hạ thấp trọng số tổn thất của các mẫu dễ phân loại (ADL chiếm đa số) và tập trung phân phối Gradient vào các mẫu khó học (Fall).

$$\alpha_t = y \cdot \alpha + (1 - y) \cdot (1 - \alpha)$$

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

* Hệ số $\alpha = 0.75$ giúp tăng trọng số phạt khi bỏ sót ca ngã (lớp 1).
* Hệ số tập trung $\gamma = 2.0$ điều chỉnh tốc độ giảm trọng số của các mẫu dễ học.

#### 2. Nhiễu Skeleton Jittering (Data Augmentation)
Trong quá trình huấn luyện, với xác suất $50\%$, các chuỗi thuộc nhãn `Fall` (lớp 1) sẽ được bơm thêm nhiễu trắng Gaussian ngẫu nhiên nhằm giúp mô hình học được các biến thể tư thế ngã đa dạng hơn, cải thiện đáng kể khả năng tổng quát hóa:

$$\text{Noise} \sim \mathcal{N}(0, 0.02)$$

---

### 3.4. Chiến lược lựa chọn mô hình (Model Selection Pipeline)

* **Giai đoạn 1 — Lựa chọn kiến trúc (5-Fold Cross Validation):** Đánh giá khách quan 4 kiến trúc trên toàn bộ dữ liệu bằng phương pháp *Stratified K-Fold*: **BiGRU-Attention**, **LSTM**, **GRU**, **CNN-1D**. Kiến trúc sở hữu **Mean F1-Score cao nhất** sẽ được chọn.
* **Giai đoạn 2 — Tối ưu siêu tham số (Grid Search):** Thực hiện tối ưu hóa cấu hình trên kiến trúc chiến thắng qua không gian tham số: `hidden_size: [32, 64]`, `num_layers: [1, 2]`, `lr: [0.001, 0.005]`. Model tốt nhất được xác định ngưỡng phân loại (threshold) bằng **Youden Index** trên Validation set trước khi đánh giá trên Test set độc lập.

---

## 4. Kết quả thực nghiệm (Experimental Results)

### 4.1. So sánh hiệu năng 5-Fold Cross Validation
| Kiến trúc Mô hình | Mean Accuracy | Mean Recall | Mean F1-Score | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **BiGRU-Attention (Optimized)** | **92.80%** | **90.78%** | **86.02%** | **Được chọn** |
| **LSTM** | 91.07% | 93.34% | 83.63% | Loại |
| **GRU** | 90.80% | 92.49% | 83.13% | Loại |
| **CNN-1D** | 83.48% | 88.67% | 72.44% | Loại |

### 4.2. Kết quả cấu hình tối ưu từ Grid Search
| Tham số | Giá trị tối ưu |
| :--- | :---: |
| **Hidden Size** | 64 |
| **Num Layers** | 2 |
| **Learning Rate** | 0.001 |

### 4.3. Đánh giá trên tập Test độc lập hoàn toàn
Sau khi tối ưu hóa ngưỡng quyết định bằng Youden Index trên tập Validation, **ngưỡng tối ưu được xác định là 0.4577**. Mô hình đạt kết quả xuất sắc trên tập Test độc lập như sau:

| Chỉ số | Giá trị | Ý nghĩa thực tế |
| :--- | :---: | :--- |
| **Accuracy** | **92.47%** | Gần 92.5/100 mẫu được phân loại chính xác hoàn toàn. |
| **Recall (Fall)** | **91.62%** | Nhận diện chính xác hơn 91.6% các ca té ngã thực tế. |
| **F1-Score** | **85.61%** | Đạt trạng thái cân bằng rất cao giữa Precision và Recall. |

---

## 5. Phân tích sâu & Đánh giá (Analysis)

### 5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)

<p align="center">
  <img src="plots/Confusion Matrix Fall Detection.png" alt="Normalized Confusion Matrix" width="550">
</p>

Đường chéo chính của ma trận nhầm lẫn đạt tỷ lệ phân loại rất cao và đồng đều giữa hai lớp:
* **ADL (Bình thường):** Phân loại chính xác **93%**.
* **Fall (Té ngã):** Nhận diện chính xác **92%** (tương đương chỉ số Recall thực tế trên tập Test).

> [!TIP]
> **Các chỉ số lỗi quan trọng:**
> * **Tỷ lệ Báo động giả (False Alarm Rate - 7%):** Chỉ có 7% số ca sinh hoạt bình thường bị mô hình đoán nhầm thành té ngã.
> * **Tỷ lệ Bỏ sót ca ngã (False Negative Rate - 8%):** Mô hình chỉ bỏ sót khoảng 8% số ca ngã thực tế. Tỷ lệ này hoàn toàn nằm trong phạm vi an toàn chấp nhận được đối với một kiến trúc mạng gọn nhẹ hoạt động bảo mật quyền riêng tư.

### 5.2. Biểu đồ lịch sử huấn luyện (Training History Analysis)

<p align="center">
  <img src="plots/Training History Fall Detection.png" alt="Training History Analysis" width="800">
</p>

#### a. Biểu đồ đường Loss
* **Độ hội tụ vượt trội:** Đường `Train Loss` giảm mạnh mẽ và mượt mà từ **0.0460** xuống còn **0.0029** (Epoch 50), cho thấy mô hình học tập vô cùng nhanh chóng nhờ vào cơ chế chú ý thông minh.
* **Độ ổn định cao:** Đường `Val Loss` giảm sâu ở các epoch đầu và dao động cực kỳ ổn định trong vùng cực tiểu (`0.0216` đến `0.0240`) ở giai đoạn sau. Hiện tượng quá khớp (overfitting) hoàn toàn được kiểm soát tốt.
* **Điểm dừng tối ưu:** Trạng thái mô hình tốt nhất được tự động ghi nhận và khôi phục tại **Epoch 46** (thời điểm đạt đỉnh cao nhất về khả năng phân loại toàn diện với `Val F1 = 90.1%` và `Val Recall = 88.6%`).

#### b. Biểu đồ độ chính xác (Accuracy)
* **Tập huấn luyện (Train Set):** Đường `Train Accuracy` bứt phá bền vững từ **80.1%** lên tới **98.3%** ở epoch 50.
* **Tập kiểm thử nội bộ (Validation Set):** Đường `Val Accuracy` tăng tốc nhanh chóng ngay từ 10 epoch đầu tiên (vượt mốc 90%) và duy trì ổn định trong biên độ cao từ **92.5% - 95.2%** cho đến cuối, khẳng định sự vượt trội hoàn toàn của kiến trúc **BiGRU-Attention** mới.
