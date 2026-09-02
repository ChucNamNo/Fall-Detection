<h1 align="center"><b>Hệ Thống Phát Hiện Té Ngã</b></h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-v8.0-00FFFFFF?logo=ultralytics&logoColor=white" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/Architecture-BiGRU--Attention-darkorange" alt="Architecture">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

Hệ thống phát hiện hành vi té ngã theo thời gian thực dựa trên trích xuất khung xương người (**Skeleton-based Fall Detection**). Hệ thống ứng dụng mô hình thị giác **YOLOv8-Pose** để trích xuất đặc trưng động học không gian - thời gian và mạng hồi quy hai chiều tích hợp cơ chế chú ý **BiGRU-Attention** để phân loại chuỗi hành vi.

<p align="center">
  <img width="636" height="430" alt="Fall Detection Demo Snapshot" src="https://github.com/user-attachments/assets/e0e63c29-48d7-4b97-844a-76a12af4c81b" />
</p>

<p align="center">
  <a href="https://youtu.be/1Gddk5XIg-o" target="_blank">
    <img src="https://img.shields.io/badge/🎥%20Video%20Demo-Watch%20on%20YouTube-red?style=for-the-badge&logo=youtube" alt="Video Demo">
  </a>
</p>

---

## Mục lục
- [1. Mục tiêu dự án (Objective)](#1-mục-tiêu-dự-án-objective)
- [2. Bộ dữ liệu huấn luyện (Dataset)](#2-bộ-dữ-liệu-huấn-luyện-dataset)
  - [2.1. Chiến lược phân chia dữ liệu (Data Splitting Strategy)](#21-chiến-lược-phân-chia-dữ-liệu-data-splitting-strategy)
- [3. Phương pháp thực hiện (Methodology)](#3-phương-pháp-thực-hiện-methodology)
  - [3.1. Quy trình Tiền xử lý Dữ liệu & Chuẩn hóa Không gian (Data Pipeline & Spatial Normalization)](#31-quy-trình-tiền-xử-lý-dữ-liệu--chuẩn-hóa-không-gian-data-pipeline--spatial-normalization)
  - [3.2. Trích xuất đặc trưng động học (Kinematic Feature Engineering)](#32-trích-xuất-đặc-trưng-động-học-kinematic-feature-engineering)
  - [3.3. Xử lý mất cân bằng nhãn (Imbalanced Data Handling)](#33-xử-lý-mất-cân-bằng-nhãn-imbalanced-data-handling)
  - [3.4. Cơ chế Chú ý Chi tiết (Feed-Forward Soft Attention Mechanism)](#34-cơ-chế-chú-ý-chi-tiết-feed-forward-soft-attention-mechanism)
  - [3.5. Quy trình lựa chọn mô hình (Model Selection Pipeline)](#35-quy-trình-lựa-chọn-mô-hình-model-selection-pipeline)
- [4. Kết quả thực nghiệm & Phân tích (Experimental Results)](#4-kết-quả-thực-nghiệm--phân-tích-experimental-results)
  - [4.1. So sánh hiệu năng 5-Fold Cross Validation](#41-so-sánh-hiệu-năng-5-fold-cross-validation)
  - [4.2. Khảo sát Ma trận Siêu tham số Focal Loss (Alpha x Gamma)](#42-khảo-sát-ma-trận-siêu-tham-số-focal-loss-alpha-x-gamma)
  - [4.3. Nghiên cứu Cắt bỏ Thành phần (Ablation Study)](#43-nghiên-cứu-cắt-bỏ-thành-phần-ablation-study)
  - [4.4. Đánh giá Mô hình Đề xuất trên tập Test độc lập](#44-đánh-giá-mô-hình-đề-xuất-trên-tập-test-độc-lập)
- [5. Phân tích chi tiết & Đánh giá (Analysis)](#5-phân-tích-chi-tiết--đánh-giá-analysis)
  - [5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)](#51-ma-trận-nhầm-lẫn-confusion-matrix-analysis)
  - [5.2. Biểu đồ lịch sử huấn luyện (Training History Analysis)](#52-biểu-đồ-lịch-sử-huấn-luyện-training-history-analysis)

---

## 1. Mục tiêu dự án (Objective)

* **Xây dựng Pipeline AI thời gian thực:** Nhận diện hành vi té ngã (Fall Detection) từ luồng video trực tiếp với độ trễ thấp và độ tin cậy ổn định.
* **Ứng dụng giám sát an toàn:** Hỗ trợ cảnh báo tự động trong môi trường y tế, theo dõi người cao tuổi và bệnh nhân tại nhà hoặc cơ sở chăm sóc tập trung.
* **Bảo vệ quyền riêng tư (Privacy-preserving):** Thay vì xử lý trực tiếp hình ảnh thô, hệ thống chỉ lưu trữ và truyền tải dữ liệu ma trận tọa độ khung xương (skeleton keypoints), giảm thiểu nguy cơ lộ thông tin cá nhân nhạy cảm và tối ưu hóa băng thông trên các thiết bị biên (Edge Devices).

---

## 2. Bộ dữ liệu huấn luyện (Dataset)

* **Nguồn gốc:** Bộ dữ liệu chuẩn hóa **IMVIA Le2i Fall Detection Dataset**.
* **Kênh khai thác:** Tải về qua Kaggle: [FallDataset IMVIA](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data).
* **Bối cảnh dữ liệu:** Video ghi hình các hành vi sinh hoạt thường ngày (ADL - Activities of Daily Living) và các tình huống té ngã mô phỏng tại các không gian thực tế khác nhau (`Coffee_room_01`, `Home_01`,...).

### Thống kê tập dữ liệu sau xử lý
| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
| :---: | :--- | :---: | :---: |
| **0** | ADL (Bình thường) | 7,628 | 75.6% |
| **1** | Fall (Té ngã) | 2,463 | 24.4% |
| **Tổng** | | **10,091** | **100%** |

> [!WARNING]
> **Thách thức cốt lõi:** Tỷ lệ chênh lệch nhãn ~3:1 giữa ADL và Fall gây ra hiện tượng thiên lệch gradient về phía lớp đa số, đòi hỏi các giải pháp xử lý mất cân bằng dữ liệu chuyên biệt.

### 2.1. Chiến lược phân chia dữ liệu (Data Splitting Strategy)

Toàn bộ **10,091 mẫu** dữ liệu được phân chia theo tỷ lệ **70 : 15 : 15** bằng phương pháp phân tầng (**Stratified Split**), đảm bảo tỷ lệ mất cân bằng nhãn 3:1 được giữ nguyên trên cả 3 tập:

* **Tập Huấn luyện (Train Set - 70%):** Dùng để cập nhật trọng số cho mô hình kết hợp kỹ thuật tăng cường dữ liệu *Skeleton Jittering*.
* **Tập Kiểm thử nội bộ (Validation Set - 15%):** Dùng để theo dõi quá trình hội tụ, kích hoạt cơ chế dừng sớm (*Early Stopping*) và xác định ngưỡng quyết định tối ưu (*Optimal Threshold*) bằng Youden Index.
* **Tập Kiểm thử độc lập (Test Set - 15%):** Cô lập hoàn toàn trong quá trình tối ưu tham số, chỉ được sử dụng ở bước đánh giá hiệu năng cuối cùng.

#### Bảng thống kê chi tiết phân chia dữ liệu:

| Tập dữ liệu | Tỷ lệ phân chia | Số mẫu ADL (Nhãn 0) | Số mẫu Fall (Nhãn 1) | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 70.0% | 5,340 | 1,724 | **7,064** |
| **Validation Set** | 15.0% | 1,144 | 369 | **1,513** |
| **Test Set** | 15.0% | 1,144 | 370 | **1,514** |
| **Tổng cộng** | **100%** | **7,628** | **2,463** | **10,091** |

---

## 3. Phương pháp thực hiện (Methodology)

### 3.1. Quy trình Tiền xử lý Dữ liệu & Chuẩn hóa Không gian (Data Pipeline & Spatial Normalization)

Quy trình xử lý dữ liệu đầu vào từ video thô của bộ dữ liệu **IMVIA Le2i** đến tập dữ liệu chuỗi khung xương hoàn chỉnh được thực hiện khép kín qua 4 bước chính:

> **Pipeline:** `[Video .avi Thô]` ──> `[Bóc tách Annotation]` ──> `[Trích xuất Pose (YOLOv8)]` ──> `[Chuẩn hóa Khung xương]` ──> `[Sliding Window (16 frames)]`

#### 1. Bóc tách và Cấu trúc Nhãn Nâng cao (Advanced Annotation Parsing)
Mỗi video trong bộ dữ liệu IMVIA đi kèm với file cấu hình `.txt` ghi nhận nhãn hành vi. Hàm xử lý nhãn đọc và bóc tách tập hợp các frame té ngã (`fall_frames_set`) thông qua 2 cơ chế song song:
* **Dạng khoảng frame (Range format):** Đọc dải frame liên tục từ `start_fall` đến `end_fall` ghi ở các dòng đầu tiên của file annotation.
* **Dạng mã trạng thái (Status code format):** Quét từng dòng theo cấu trúc `(frame_idx, status_code)`. Các frame mang mã trạng thái thuộc tập $\{2, 4, 7, 8\}$ (đại diện cho các giai đoạn mất thăng bằng, va chạm và nằm trên sàn) sẽ được nạp trực tiếp vào tập nhãn té ngã.

#### 2. Trích xuất Khung xương với YOLOv8-Pose (Keypoint Extraction)
Video `.avi` được đọc theo từng khung hình (frame-by-frame) qua thư viện OpenCV:
* Mô hình thị giác **YOLOv8n-Pose** (`yolov8n-pose.pt`) tiến hành phát hiện người và trích xuất tọa độ không gian 2D của **17 điểm khớp chuẩn COCO** ($\mathbf{KP} \in \mathbb{R}^{17 \times 2}$).

#### 3. Chuẩn hóa Không gian Khung xương (Spatial Normalization)
Để giúp vector đặc trưng đạt **tính bất biến với vị trí, khoảng cách và góc nhìn của camera** đối với đối tượng, ma trận tọa độ $\mathbf{KP}$ của từng frame được chuẩn hóa theo công thức:

$$\mathbf{KP}_{\text{shifted}} = \mathbf{KP} - \mathbf{C}_{\text{hip}}$$

$$\mathbf{KP}_{\text{normalized}} = \frac{\mathbf{KP}_{\text{shifted}}}{H_{\text{body}}}$$

Trong đó:
* **Tâm hông ($\mathbf{C}_{\text{hip}}$):** Được xác định bằng trung điểm tọa độ của 2 khớp hông trái và phải (Keypoint 11 và 12):
  $$\mathbf{C}_{\text{hip}} = \frac{\mathbf{KP}[11] + \mathbf{KP}[12]}{2}$$
* **Chiều cao cơ thể ước tính ($H_{\text{body}}$):** Tính bằng khoảng cách theo trục thẳng đứng ($Y$) từ đỉnh đầu/mũi (Keypoint 0) đến vị trí thấp nhất của hai cổ chân (Keypoint 15 và 16):
  $$H_{\text{body}} = \max\left( \big| Y_{\text{foot}} - Y_{\text{head}} \big|, 1.0 \right)$$
  (Lưu ý: Mẫu số $$H_{\text{body}}$$ được giới hạn giá trị tối thiểu là 1.0 để phòng tránh lỗi chia cho 0).

#### 4. Kỹ thuật Cửa sổ Trượt & Gán nhãn Chuỗi (Sliding Window Strategy)
Dữ liệu chuỗi khung xương sau khi chuẩn hóa được đóng gói thành các mẫu chuỗi thời gian (time-series sequences) phục vụ cho mô hình học sâu:
* **Kích thước cửa sổ (Window Size):** $W = 16$ frames (tương đương $\approx 0.53 - 0.64$ giây quan sát động học).
* **Bước trượt (Step/Stride):** $S = 4$ frames. Việc chọn $S = 4$ tạo ra sự chồng lấp 75% (overlap) giữa các cửa sổ kề nhau, giúp tăng cường số lượng mẫu huấn luyện và bắt trọn mọi khoảnh khắc chuyển tiếp hành vi.
* **Quy tắc gán nhãn chuỗi (Sequence Labeling):**

$$\text{Label}_{\text{seq}} = \begin{cases} 1, & \text{nếu } \exists \, f_t \in \text{cửa sổ 16 frames} \text{ mà } f_t \in \text{fall}_{\text{frames}} \\ 0, & \text{ngược lại (ADL - Sinh hoạt bình thường)} \end{cases}$$

Toàn bộ dữ liệu sau khi đóng gói được xuất ra 2 file định dạng NumPy bao gồm `sequences.npy` (mảng chứa các chuỗi tọa độ $16 \times 17 \times 2$) và `labels.npy` (mảng nhãn $0$ hoặc $1$).

### 3.2. Trích xuất đặc trưng động học (Kinematic Feature Engineering)
Ngoại trừ tọa độ không gian tĩnh, hệ thống tính toán vi phân bậc 1 (Vận tốc) và bậc 2 (Gia tốc) theo trục thời gian để phản ánh biến động động học của chuỗi hành vi.

> [!NOTE]
> **Chiến lược đệm chuỗi (Padding Strategy):**
> Với các chuỗi ngắn hơn `max_len = 16`, hệ thống áp dụng kỹ thuật **Pre-padding bằng cách lặp lại frame đầu tiên (Replicate Padding)** thay vì đệm số 0 (Zero-padding) nhằm tránh hiện tượng xuất hiện biến động vận tốc và gia tốc ảo tại ranh giới đệm.

#### Chi tiết cấu trúc vector đầu vào (Input Tensor Shape)

| Loại đặc trưng | Phương thức tính toán | Số lượng đặc trưng | Kích thước (Shape) |
| :--- | :--- | :---: | :---: |
| **Vị trí** | Tọa độ gốc (x, y) của 17 keypoints | 17 × 2 = 34 | `(16, 34)` |
| **Vận tốc** | Sai phân bậc 1 theo trục thời gian | 17 × 2 = 34 | `(16, 34)` |
| **Gia tốc** | Sai phân bậc 2 theo trục thời gian | 17 × 2 = 34 | `(16, 34)` |
| **Tổng đầu vào** | **Nối đặc trưng (Concatenation)** | **102 đặc trưng** | **`(16, 102)`** |

---

### 3.3. Xử lý mất cân bằng nhãn (Imbalanced Data Handling)

#### 1. Focal Loss chuẩn hóa (α = 0.50, γ = 2.0)
Thay thế hàm tổn thất Binary Cross Entropy (BCE) truyền thống nhằm giảm trọng số đóng góp của các mẫu dễ phân loại và tập trung gradient vào các mẫu khó phân biệt:

* **α = 0.50**: Tăng hệ số phạt tổn thất khi dự đoán sai lớp té ngã (lớp 1).
* **γ = 2.0**: Điều chỉnh tốc độ giảm trọng số của các mẫu dễ phân loại.

#### 2. Nhiễu Skeleton Jittering (Data Augmentation)
Trong quá trình huấn luyện, các chuỗi thuộc nhãn `Fall` (lớp 1) được bổ sung ngẫu nhiên nhiễu trắng Gaussian N(0, 0.02) với xác suất 50% để nâng cao khả năng tổng quát hóa.

---

### 3.4. Cơ chế Chú ý Chi tiết (Feed-Forward Soft Attention Mechanism)

Cơ chế **Feed-Forward Soft Attention** được tích hợp nhằm tính toán trọng số đóng góp của từng bước thời gian *t* thay vì chỉ lấy trạng thái ẩn cuối cùng:

1. **Điểm số chú ý (Attention Score):** `e_t = W_a * h_t + b_a`
2. **Trọng số chuẩn hóa (Attention Weights):** `α_t = exp(e_t) / Σ exp(e_k)`
3. **Vector ngữ cảnh (Context Vector):** `c = Σ (α_t * h_t)`

Vector `c` tổng hợp thông tin từ các khung hình chứa biến động động học lớn (giai đoạn mất thăng bằng hoặc va chạm) để đưa vào lớp phân loại cuối cùng.

---

### 3.5. Quy trình lựa chọn mô hình (Model Selection Pipeline)

* **Giai đoạn 1 — So sánh kiến trúc (5-Fold Cross Validation):** Đánh giá 4 kiến trúc: **BiGRU-Attention**, **LSTM**, **GRU**, **CNN-1D**.
* **Giai đoạn 2 — Tối ưu siêu tham số:** Xác định cấu hình tối ưu qua Grid Search: `hidden_size = 64`, `num_layers = 2`, `lr = 0.001`.
* **Giai đoạn 3 — Thực nghiệm chứng minh:** 
  1. *Khảo sát ma trận Focal Loss (γ x α)* trên tập Validation.
  2. *Nghiên cứu cắt bỏ (Ablation Study)* trên tập Test để đánh giá đóng góp của từng thành phần.

---

## 4. Kết quả thực nghiệm & Phân tích (Experimental Results)

### 4.1. So sánh hiệu năng 5-Fold Cross Validation
Để đánh giá toàn diện các kiến trúc mạng học sâu, báo cáo không chỉ xem xét các chỉ số độ chính xác (Accuracy, Recall, F1-Score) mà còn phân tích mối quan hệ đánh đổi (trade-off) giữa hiệu năng nhận dạng và độ phức tạp tính toán (Số lượng tham số học được và chi phí FLOPs cho một chuỗi 16 frames đầu vào).

| Mô hình (Model) | Mean Accuracy | Mean Recall | Mean F1-Score | Parameters | FLOPs (16 frames) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiGRU-Attention** | **93.51%** | **89.89%** | **87.13%** | **139,266** | **4.51 MFLOPs** | **Được chọn** |
| **GRU** | 91.93% | 90.74% | 84.64% | 57,281 | 1.86 MFLOPs | Bị loại |
| **LSTM** | 90.73% | 92.57% | 83.02% | 76,353 | 2.47 MFLOPs | Bị loại |
| **CNN-1D** | 83.41% | 87.78% | 72.19% | 19,713 | 0.63 MFLOPs | Bị loại |

**Đánh giá:**
* **Hiệu năng nhận dạng tối ưu nhất — BiGRU-Attention**: Mô hình BiGRU-Attention đạt chỉ số tổng hợp tốt nhất với F1-Score = 87.13% và Accuracy = 93.51% (tăng tương ứng 1.11% F1-Score và 0.71% Accuracy so với kết quả thử nghiệm trước). Việc kết hợp cơ chế Attention cùng GRU hai chiều giúp mô hình lọc nhiễu tốt và tập trung vào các chuyển động quan trọng trong chuỗi dữ liệu xương (skeleton). Tuy có dung lượng tham số lớn nhất (139,266 parameters) và chi phí 4.51 MFLOPs, mức tài nguyên này vẫn đủ nhẹ để chạy mượt mà theo thời gian thực trên các thiết bị Edge AI thông thường.
* **Sự dịch chuyển thứ hạng giữa GRU và LSTM**: Nhờ tối ưu hóa quy trình, GRU vượt qua LSTM để vươn lên vị trí thứ 2 với F1-Score = 84.64% (cao hơn 1.62% so với LSTM). Xét về mặt chi phí tính toán, GRU thể hiện ưu thế vượt trội hơn hẳn LSTM.
* **Mô hình siêu nhẹ — CNN-1D**: Mặc dù CNN-1D có chi phí tính toán cực kỳ thấp (chỉ 19,713 parameters và 0.63 MFLOPs), hiệu năng nhận dạng của nó khá khiêm tốn (F1-Score = 72.19%). Điều này cho thấy các lớp tích chập 1 chiều thuần túy khó bắt trọn được sự phụ thuộc chuỗi theo thời gian (temporal dependency) dài bằng các kiến trúc học chuỗi chuyên dụng dạng RNN.

Mô hình BiGRU-Attention là lựa chọn tối ưu nhất cho bài toán phát hiện té ngã. Mặc dù chi phí FLOPs tăng gấp ~2.4 lần so với GRU đơn thuần, mức bù đắp +2.49% F1-Score mang lại giá trị rất lớn trong việc giảm thiểu tối đa các tình huống báo động giả (False Positives).
  
---

### 4.2. Khảo sát Ma trận Siêu tham số Focal Loss (Alpha x Gamma)

Thực nghiệm tiến hành quét ma trận giữa $\gamma$ trong dải [0.1, 0.2, 0.5, 1.0, 2.0] và $\alpha$ trong dải [0.10, 0.25, 0.50, 0.75, 0.90] trên tập **Validation Set** đối với 4 kiến trúc (cố định Hidden Size = 64, Layers = 2, LR = 0.001).

#### Kết quả Ma trận F1-Score (%) trên tập Validation:

##### a. Mô hình BiGRU-Attention
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 78.06% | 82.14% | 85.47% | 85.56% | 82.41% |
| **γ = 0.2** | 77.81% | 81.79% | 85.39% | 86.70% | 83.17% |
| **γ = 0.5** | 77.27% | 82.26% | 87.39% | 87.18% | 83.37% |
| **γ = 1.0** | 76.05% | 81.28% | 86.49% | 86.12% | 82.77% |
| **γ = 2.0** | 77.35% | 81.80% | **87.59%** | 85.24% | 83.40% |

##### b. Mô hình LSTM
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 75.54% | 81.39% | **85.17%** | 82.52% | 80.39% |
| **γ = 0.2** | 75.71% | 82.33% | 84.59% | 82.86% | 78.87% |
| **γ = 0.5** | 71.94% | 81.28% | 85.07% | 84.31% | 80.09% |
| **γ = 1.0** | 74.07% | 83.41% | 84.62% | 85.03% | 81.32% |
| **γ = 2.0** | 74.83% | 83.43% | 84.71% | 82.38% | 80.09% |

##### c. Mô hình GRU
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 73.56% | 81.29% | 83.48% | 83.24% | 78.52% |
| **γ = 0.2** | 74.04% | 79.33% | 84.64% | 83.45% | 79.09% |
| **γ = 0.5** | 75.66% | 80.83% | 82.72% | 84.95% | 77.84% |
| **γ = 1.0** | 75.04% | 80.95% | 84.15% | 84.31% | 77.92% |
| **γ = 2.0** | 73.37% | 81.37% | **85.26%** | 83.63% | 78.93% |

##### d. Mô hình CNN-1D
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 0.00% | 31.86% | 0.00% | 0.00% | 0.00% |
| **γ = 0.2** | 0.00% | 32.30% | 0.00% | 0.00% | 0.00% |
| **γ = 0.5** | 0.00% | 34.50% | 0.00% | 0.00% | 0.00% |
| **γ = 1.0** | 42.77% | 57.84% | 70.74% | **71.51%** | 66.67% |
| **γ = 2.0** | 42.53% | 57.84% | 69.69% | 70.44% | 66.26% |

> [!NOTE]
> **Ghi chú về số liệu thực nghiệm:**
> Số liệu trong Mục 4.2 được đánh giá trên **Validation Set** trong quá trình tìm kiếm không gian siêu tham số. Điểm F1-Score đạt giá trị cao nhất là **87.59%** trên mô hình đề xuất **BiGRU-Attention** tại cấu hình $(\alpha = 0.50, \gamma = 2.0)$.

---

#### 4.2.1. Kết luận & Lý do lựa chọn bộ siêu tham số Focal Loss (α = 0.50, γ = 2.0)

Dựa trên kết quả phân tích ma trận thực nghiệm 5x5 của mô hình đề xuất `BiGRU-Attention`, hệ thống chính thức lựa chọn bộ siêu tham số **α = 0.50 và γ = 2.0**. Lựa chọn này dựa trên các luận điểm khoa học về hiệu năng đỉnh, tính ổn định vùng và khả năng tổng quát hóa:

1. **Lý do lựa chọn α = 0.50 (Sự tối ưu hóa hài hòa giữa Precision và Recall):**
   * Kết quả thực nghiệm cho thấy dải **α = 0.50** liên tục tạo ra hiệu năng áp đảo trên hầu hết các mô hình dạng chuỗi (BiGRU-Attention đạt 87.59%, GRU đạt 85.26%, LSTM đạt 85.17%).
   * Việc thiết lập $\alpha = 0.50$ duy trì sự cân bằng lực kéo gradient giữa lớp dương tính (Fall) và âm tính (ADL), tránh hiện tượng phạt quá thiên lệch về một phía, từ đó đạt chỉ số F1-Score tối ưu hài hòa nhất trên toàn tập dữ liệu.

2. **Lý do lựa chọn γ = 2.0 (Ý nghĩa Focal Loss và Độ ổn định vùng cực trị):**
   * *Đạt hiệu năng cao nhất (Global Peak):* Cấu hình $(\alpha = 0.50, \gamma = 2.0)$ đạt đỉnh F1-Score **87.59%** trên mô hình BiGRU-Attention, cao nhất trong toàn bộ 100 cấu hình thực nghiệm của 4 kiến trúc.
   * *Ý nghĩa lý thuyết của Focal Loss:* Với $\gamma = 2.0$, thừa số $(1 - p_t)^2$ tạo ra lực triệt tiêu trọng số tổn thất đủ mạnh đối với các mẫu dễ phân loại ($p_t > 0.9$), ép gradient tập trung tối đa vào việc xử lý các chuỗi hành vi khó ở vùng ranh giới (ví dụ: cúi nhanh, ngồi xuống đột ngột dễ bị nhầm với té ngã).
   * *Tính ổn định vùng (Flat Minimum Region):* Tại cột $\alpha = 0.50$, hiệu năng của BiGRU-Attention duy trì mức rất cao và phẳng trên các giá trị $\gamma$ lớn ($\gamma = 0.5: 87.39\%$, $\gamma = 1.0: 86.49\%$, $\gamma = 2.0: 87.59\%$). Việc nằm trong một vùng cực trị phẳng đảm bảo mô hình có độ bền vững cao với nhiễu dữ liệu và đạt khả năng tổng quát hóa tối ưu khi đánh giá trên tập Test độc lập.

---

### 4.3. Nghiên cứu Cắt bỏ Thành phần (Ablation Study)

Thực nghiệm **Ablation Study** được thực hiện trên tập **Validation Set** nhằm khảo sát định lượng đóng góp của từng thành phần kỹ thuật (kiến trúc mạng BiGRU, cơ chế Attention, đặc trưng động học Kinematics, và hàm mất mát Focal Loss) đối với hiệu năng chung của mô hình:

---

### Bảng kết quả thực nghiệm

| STT | Biến thể Cấu hình Thử nghiệm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Parameters | FLOPs (M) | Tác động khi cắt bỏ / Thay thế |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | **PROPOSED (BiGRU + Attention + Kinematics)** | **94.24** | **92.19** | **83.42** | **87.59** | **139,266** | **4.5100** | **Mô hình đề xuất đầy đủ (Baseline)** |
| **2** | **BiGRU (Bỏ Attention)** | 93.18 | 90.27 | 80.71 | 85.22 | 139,137 | 4.5059 | F1 giảm **-2.37%**: Precision và Recall đều sụt giảm. |
| **3** | **LSTM + Attention** | 93.31 | 89.15 | 82.61 | 85.75 | 185,602 | 6.0009 | F1 giảm **-1.84%**: Tốn thêm ~33.3% FLOPs và Params. |
| **4** | **Vanilla LSTM (Bỏ Attention)** | 92.72 | 88.62 | 80.43 | 84.33 | 185,473 | 5.9968 | F1 giảm **-3.26%**: Tốn tài nguyên nhưng hiệu năng kém nhất. |
| **5** | **Bỏ Kinematics (Chỉ Pose thô 34 dims)** | 93.91 | 91.32 | 82.88 | 86.89 | 113,154 | 3.6744 | F1 giảm **-0.70%**: Tiết kiệm ~18.5% FLOPs nhưng suy giảm hiệu năng. |
| **6** | **Bỏ Focal Loss (Standard BCE)** | 93.84 | 91.54 | 82.34 | 86.70 | 139,266 | 4.5100 | F1 giảm **-0.89%**: Recall thấp hơn và tổng thể kém hiệu quả hơn Focal Loss. |

---

### Phân tích đóng góp & Độ phức tạp tính toán

- **Cơ chế Attention (Cấu hình 1 vs 2):**
  Loại bỏ khối Attention khỏi BiGRU khiến F1-Score giảm từ **87.59%** xuống **85.22%** (-2.37%), đồng thời sụt giảm ở cả Precision (từ **92.19%** xuống **90.27%**) và Recall (từ **83.42%** xuống **80.71%**). Đáng chú ý, cơ chế Attention chỉ bổ sung **129 parameters** (139,266 vs 139,137) và tiêu tốn thêm một lượng chi phí tính toán cực kỳ nhỏ (**0.0041 MFLOPs**). Điều này chứng minh Attention mang lại hiệu quả tập trung trọng số thời gian vượt trội với mức chi phí tài nguyên gần như không đáng kể.

- **So sánh BiGRU và LSTM (Cấu hình 1 vs 3):**
  Khi thay thế BiGRU bằng LSTM (+ Attention), F1-Score giảm xuống **85.75%** (-1.84%). Xét về mặt tài nguyên phần cứng, kiến trúc LSTM tốn nhiều hơn **33.27% số lượng Parameters** (185,602 vs 139,266) và **33.06% chi phí tính toán FLOPs** (6.0009M vs 4.5100M). Kết quả khẳng định BiGRU là sự lựa chọn tối ưu hơn hẳn LSTM cả về độ chính xác lẫn tính gọn nhẹ khi xử lý chuỗi thời gian ngắn.

- **Đặc trưng Động học Kinematics (Cấu hình 1 vs 5):**
  Khi cắt bỏ thông tin Vận tốc và Gia tốc (chuyển từ 102 chiều xuống 34 chiều pose thô), số lượng FLOPs giảm **18.53%** (từ 4.5100M xuống 3.6744M) và tham số giảm xuống còn 113,154. Tuy nhiên, F1-Score giảm từ **87.59%** xuống **86.89%** (-0.70%), với Precision giảm xuống **91.32%** và Recall giảm xuống **82.88%**. Việc đánh đổi thêm **0.8356 MFLOPs** để tích hợp Kinematics hoàn toàn đáng giá nhằm đảm bảo mô hình nắm bắt tốt các biến thiên chuyển động đột ngột đặc thù của hành vi té ngã.

- **Hàm mất mát Focal Loss (Cấu hình 1 vs 6):**
  Thay thế Focal Loss bằng Standard BCE duy trì nguyên vẹn tham số (139,266) và FLOPs (4.5100M). Tuy nhiên, F1-Score sụt giảm từ **87.59%** xuống **86.70%** (-0.89%), với Recall giảm xuống **82.34%** (so với 83.42% của mô hình đề xuất). Kết quả này khẳng định Focal Loss giúp mô hình xử lý tình trạng mất cân bằng dữ liệu hiệu quả hơn, duy trì khả năng nhận diện cao ở cả Precision và Recall.

---

### 4.4. Đánh giá Mô hình Đề xuất trên tập Test độc lập

Sau khi xác định cấu hình tối ưu từ ablation study, mô hình **BiGRU + Attention + Kinematics + Focal Loss** được đưa vào đánh giá chính thức trên Test Set hoàn toàn độc lập (chiếm 15% tổng dữ liệu, chưa từng xuất hiện trong quá trình huấn luyện hay tinh chỉnh tham số). Ngưỡng phân loại tối ưu theo chỉ số **Youden Index ($J = 0.4577$, làm tròn $\approx 0.46$)** được áp dụng nhằm cân bằng và tối đa hóa khả năng phát hiện sự cố té ngã.

#### Bảng chỉ số đánh giá trên tập Test độc lập

| Chỉ số đánh giá | Giá trị | Ý nghĩa ứng dụng trong giám sát té ngã |
| :--- | :---: | :--- |
| **Accuracy** | **92.47%** | Phân loại chính xác 92.47% tổng số chuỗi hành vi. |
| **Precision** | **80.34%** | Trong các cảnh báo đưa ra, 80.34% phản ánh đúng sự cố té ngã thực tế. |
| **Recall (Fall)** | **91.62%** | Nhận diện chính xác 91.62% các trường hợp té ngã thực tế. |
| **F1-Score** | **85.61%** | Đạt mức cân bằng tối ưu giữa khả năng phát hiện sự cố và hạn chế cảnh báo nhầm. |

---

## 5. Phân tích Chi tiết & Đánh giá (Analysis)

### 5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)

<p align="center">
  <img src="plots/Confusion Matrix Fall Detection.png" alt="Normalized Confusion Matrix" width="550">
</p>

Đánh giá ma trận nhầm lẫn chuẩn hóa trên tập Test độc lập (với threshold = 0.46) cho thấy:
* **ADL (Sinh hoạt bình thường):** Phân loại chính xác **93.0%**.
* **Fall (Té ngã):** Nhận diện chính xác **92.0%** (tương ứng với độ nhạy Recall cao trên tập dữ liệu kiểm thử).

> **Phân tích chỉ số lỗi:**
> * **Tỷ lệ báo động giả (False Alarm Rate):** Chỉ **7.0%** các hoạt động sinh hoạt bình thường bị ghi nhận nhầm thành té ngã (ADL bị dự đoán nhầm thành Fall).
> * **Tỷ lệ bỏ sót sự cố (False Negative Rate):** Mô hình bỏ sót **8.0%** các trường hợp té ngã thực tế (Fall bị dự đoán nhầm thành ADL). Đây là ngưỡng sai số hoàn toàn chấp nhận được và đáp ứng tốt yêu cầu an toàn đối với các hệ thống giám sát tự động dựa trên khung xương.

---

### 5.2. Biểu đồ lịch sử huấn luyện (Training History Analysis)

<p align="center">
  <img src="plots/Training History Fall Detection.png" alt="Training History Analysis" width="800">
</p>

#### a. Biểu đồ đường Loss
* **Độ hội tụ vượt trội:** Đường `Train Loss` giảm mạnh mẽ và mượt mà từ **0.0460** (Epoch 1) xuống còn **0.0029** (Epoch 50), cho thấy mô hình tối ưu hóa trọng số rất nhanh chóng nhờ vào cơ chế chú ý (Attention Mechanism) và đặc trưng động học Kinematics.
* **Độ ổn định cao:** Đường `Val Loss` giảm sâu ở các epoch đầu và dao động cực kỳ ổn định trong vùng cực tiểu (đạt mức thấp nhất **0.0216** ở Epoch 32 và duy trì trong dải **0.022 - 0.033**). Hiện tượng quá khớp (overfitting) hoàn toàn được kiểm soát nhờ các kỹ thuật Dropout và Regularization.
* **Điểm dừng tối ưu:** Trạng thái mô hình tốt nhất được tự động ghi nhận và khôi phục tại **Epoch 46** (thời điểm đạt đỉnh F1 toàn diện nhất trên tập Validation với `Val F1 = 90.1%`, `Val Recall = 88.6%`, `Val Acc = 95.2%`, `Train Loss = 0.0037` và `Train Acc = 98.6%`).

#### b. Biểu đồ độ chính xác (Accuracy)
* **Tập huấn luyện (Train Set):** Đường `Train Accuracy` bứt phá bền vững từ **80.1%** lên đến **98.3%** ở epoch 50 (đạt đỉnh **98.9%** ở epoch 47 & 49).
* **Tập kiểm thử nội bộ (Validation Set):** Đường `Val Accuracy` tăng tốc nhanh chóng ngay từ 10 epoch đầu tiên (vượt mốc 91%) và duy trì ổn định trong biên độ cao từ **92.5% - 95.2%** cho đến cuối quá trình huấn luyện, khẳng định tính tổng quát hóa cao của kiến trúc **BiGRU-Attention**.
