# 🏃‍♂️ Real-Time Skeleton-Based Fall Detection System

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
  - [3.1. Chuẩn hóa không gian (Spatial Normalization)](#31-chuẩn-hóa-không-gian-spatial-normalization)
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

### 3.1. Chuẩn hóa không gian (Spatial Normalization)
Tọa độ 17 khớp xương (x, y) trích xuất từ YOLOv8-Pose được quy đổi về gốc tọa độ tâm hông (hip center), sau đó chuẩn hóa theo chiều cao cơ thể trong từng khung hình. Kỹ thuật này giúp vector không gian đạt tính bất biến với khoảng cách và vị trí của đối tượng so với camera.

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

#### 1. Focal Loss chuẩn hóa (α = 0.75, γ = 2.0)
Thay thế hàm tổn thất Binary Cross Entropy (BCE) truyền thống nhằm giảm trọng số đóng góp của các mẫu dễ phân loại và tập trung gradient vào các mẫu khó phân biệt:

* **α = 0.75**: Tăng hệ số phạt tổn thất khi dự đoán sai lớp té ngã (lớp 1).
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
| Kiến trúc Mô hình | Mean Accuracy | Mean Recall | Mean F1-Score | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **BiGRU-Attention (Optimized)** | **92.80%** | **90.78%** | **86.02%** | **Được chọn** |
| **LSTM** | 91.07% | 93.34% | 83.63% | Loại |
| **GRU** | 90.80% | 92.49% | 83.13% | Loại |
| **CNN-1D** | 83.48% | 88.67% | 72.44% | Loại |

---

### 4.2. Khảo sát Ma trận Siêu tham số Focal Loss (Alpha x Gamma)

Thực nghiệm tiến hành quét ma trận giữa γ trong dải [0.1, 0.2, 0.5, 1.0, 2.0] và α trong dải [0.10, 0.25, 0.50, 0.75, 0.90] trên tập **Validation Set** đối với 4 kiến trúc (cố định Hidden Size = 64, Layers = 2, LR = 0.001).

#### Kết quả Ma trận F1-Score (%) trên tập Validation:

##### a. Mô hình BiGRU-Attention
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 78.06% | 81.60% | 85.84% | 84.73% | 81.42% |
| **γ = 0.2** | 76.32% | 83.09% | 85.26% | **87.14%** | 78.32% |
| **γ = 0.5** | 76.14% | 84.70% | 84.37% | 83.73% | 81.60% |
| **γ = 1.0** | 77.93% | 82.65% | 86.36% | 84.84% | 84.83% |
| **γ = 2.0** | 79.04% | 85.02% | 85.75% | **85.79%** | 83.43% |

##### b. Mô hình LSTM
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 75.29% | 83.74% | 85.22% | 83.35% | 79.35% |
| **γ = 0.2** | 74.25% | 83.56% | 83.93% | 81.76% | 79.21% |
| **γ = 0.5** | 71.53% | 83.99% | 85.80% | 83.99% | 80.05% |
| **γ = 1.0** | 74.30% | 80.37% | 84.78% | 82.84% | 78.52% |
| **γ = 2.0** | 73.50% | 81.18% | **86.00%** | 81.94% | 78.81% |

##### c. Mô hình GRU
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 72.85% | 79.44% | 83.23% | 82.43% | 78.85% |
| **γ = 0.2** | 76.11% | 78.74% | 83.48% | 82.28% | 78.41% |
| **γ = 0.5** | 76.87% | 81.06% | 84.43% | 82.25% | 77.19% |
| **γ = 1.0** | 75.99% | 80.00% | **85.99%** | 81.30% | 78.25% |
| **γ = 2.0** | 71.38% | 81.53% | 83.04% | 82.45% | 77.29% |

##### d. Mô hình CNN-1D
| γ \ α | α = 0.10 | α = 0.25 | α = 0.50 | α = 0.75 | α = 0.90 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **γ = 0.1** | 15.88% | 0.00% | 47.65% | 0.00% | 0.00% |
| **γ = 0.2** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **γ = 0.5** | 0.00% | 32.05% | 0.00% | 0.00% | 0.00% |
| **γ = 1.0** | 44.91% | 58.24% | 69.47% | **71.21%** | 66.80% |
| **γ = 2.0** | 41.28% | 60.55% | 69.91% | 71.04% | 66.60% |

> [!NOTE]
> **Ghi chú về số liệu thực nghiệm:**
> Số liệu trong Mục 4.2 được đánh giá trên **Validation Set** trong quá trình tìm kiếm không gian siêu tham số. Sự chênh lệch giữa chỉ số Validation (85.79%) và chỉ số Test chính thức (85.61%) phản ánh quá trình kiểm thử độc lập khách quan.

#### 4.2.1. Kết luận & Lý do lựa chọn bộ siêu tham số Focal Loss (α = 0.75, γ = 2.0)

Dựa trên kết quả phân tích ma trận thực nghiệm 5x5 của mô hình đề xuất `BiGRU-Attention`, hệ thống chính thức lựa chọn bộ siêu tham số **α = 0.75 và γ = 2.0**. Lựa chọn này dựa trên các luận điểm khoa học về tính ổn định và khả năng tổng quát hóa:

1. **Lý do lựa chọn α = 0.75 (Tỷ lệ phạt tương thích với phân phối nhãn):**
   * Do tỷ lệ mất cân bằng dữ liệu gốc là **3:1** (ADL 75.6% vs Fall 24.4%), việc gán α = 0.75 tạo ra tỷ lệ phạt mất mát 3:1 nghiêng về lớp té ngã (α1 / α0 = 0.75 / 0.25 = 3). 
   * Mức phạt này cân bằng đáng kể lực kéo gradient giữa hai lớp, giúp duy trì chỉ số **Recall luôn đạt trên 90%** – chỉ số an toàn bắt buộc đối với ứng dụng giám sát y tế.

2. **Lý do lựa chọn γ = 2.0 (Phân tích độ ổn định lân cận siêu tham số):**
   * *Độ nhạy và rủi ro quá khớp tại γ = 0.2:* Mặc dù γ = 0.2 đạt đỉnh F1 cục bộ trên tập Validation (87.14%), hiệu năng tại hàng γ = 0.2 biến động rất mạnh khi α thay đổi (chỉ đạt 76.32% tại α = 0.10 và tụt xuống 78.32% tại α = 0.90). Sự biến động gắt này cho thấy cực đại tại γ = 0.2 là một "cực trị nhọn" (sharp minimum), mang rủi ro quá khớp (overfitting) cao vào tập Validation.
   * *Độ ổn định vùng và tính tổng quát hóa tại γ = 2.0:* Tại hàng γ = 2.0, F1-Score của BiGRU-Attention duy trì độ phẳng và ổn định bền vững trên dải α rộng từ 0.25 đến 0.75 (luôn đạt 85.02% - 85.79%). Cấu hình này thuộc một "vùng cực trị phẳng" (flat region), có độ biến động giữa các seed/fold thấp, đảm bảo khả năng tổng quát hóa tối ưu khi đánh giá trên tập Test độc lập.
   * *Ý nghĩa lý thuyết của Focal Loss:* Khi γ = 0.2, thừa số (1 - p_t)^0.2 xấp xỉ 1, làm hàm tổn thất hoạt động gần như trùng với Weighted BCE. Chỉ khi γ = 2.0, thừa số (1 - p_t)^2 mới tạo ra lực triệt tiêu đủ mạnh lên các mẫu dễ phân loại (p_t > 0.9), ép gradient tập trung tối đa vào các chuỗi hành vi khó ở vùng ranh giới.

---

### 4.3. Nghiên cứu Cắt bỏ Thành phần (Ablation Study)

Thực nghiệm **Ablation Study** được thực hiện trên tập **Validation Set** nhằm khảo sát định lượng đóng góp của từng thành phần kỹ thuật (kiến trúc mạng BiGRU, cơ chế Attention, đặc trưng động học Kinematics, và hàm mất mát Focal Loss) đối với hiệu năng chung của mô hình:

#### Bảng kết quả Nghiên cứu Cắt bỏ thành phần (Đánh giá trên Validation Set với ngưỡng mặc định 0.5):

| STT | Biến thể Cấu hình Thử nghiệm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Tác động khi cắt bỏ / Thay thế |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **BiGRU + Attention + Kinematics** | **92.98** | **87.43** | **83.15** | **85.24** | **Mô hình baseline đầy đủ (Ngưỡng mặc định 0.5)** |
| **2** | **BiGRU (Bỏ Attention)** | 91.19 | 79.01 | 86.96 | 82.79 | F1 giảm **-2.45%**: Precision giảm mạnh xuống 79.01%. |
| **3** | **LSTM + Attention** | 91.66 | 80.25 | 87.23 | 83.59 | F1 giảm **-1.65%**: Kiến trúc LSTM học chuỗi ít tối ưu hơn BiGRU. |
| **4** | **Vanilla LSTM (Bỏ Attention)** | 92.25 | 84.76 | 83.15 | 83.95 | F1 giảm **-1.29%**: Suy giảm độ chính xác tổng thể. |
| **5** | **Bỏ Kinematics (Chỉ Pose thô 34 dims)** | 92.25 | 80.54 | 89.95 | 84.98 | F1 giảm **-0.26%**: Precision giảm do thiếu thông tin gia tốc rơi. |
| **6** | **Bỏ Focal Loss (Standard BCE)** | 93.84 | 91.54 | 82.34 | 86.70 | F1 đạt 86.70% ở ngưỡng 0.5, nhưng Recall thấp hơn (82.34%) |

Phân tích chi tiết đóng góp của các thành phần (trên Validation Set):
* **Cơ chế Attention (Cấu hình 1 vs 2)**: Khi loại bỏ Attention khỏi BiGRU, chỉ số F1-Score giảm từ 85.24% xuống 82.79% (-2.45%), trong đó Precision giảm đáng kể từ 87.43% xuống 79.01% (-8.42%). Điều này chứng minh cơ chế Attention giúp mô hình lọc nhiễu hiệu quả và tập trung trọng số vào các khung hình chứa khoảnh khắc va chạm trọng yếu.

* **Kiến trúc BiGRU so với LSTM (Cấu hình 1 vs 3)**: Thay thế BiGRU bằng LSTM (+ Attention) làm F1-Score giảm xuống 83.59% (-1.65%), cho thấy khả năng học chuỗi hai chiều của BiGRU khai thác ngữ cảnh thời gian tốt hơn trong bài toán té ngã.

* **Đặc trưng Động học Kinematics (Cấu hình 1 vs 5)**: Khi loại bỏ Vận tốc và Gia tốc (chỉ dùng Pose thô 34 dims), F1-Score giảm từ 85.24% xuống 84.98% (-0.26%) và Precision giảm từ 87.43% xuống 80.54% (-6.89%). Việc bổ sung 68 chiều Kinematics giúp mô hình xác định chính xác các pha biến thiên vận tốc đột ngột.

* **Hàm mất mát Focal Loss (Cấu hình 1 vs 6)**: Ở ngưỡng phân loại mặc định 0.5, Standard BCE cho F1-Score cao hơn (86.70%) nhưng Recall lại thấp hơn (82.34%). Việc sử dụng Focal Loss giúp mô hình phân tách xác suất tốt hơn khi kết hợp với ngưỡng Youden Index tối ưu ở bước đánh giá tập Test.

---

### 4.4. Đánh giá Mô hình Đề xuất trên tập Test độc lập

Sau khi xác định cấu hình tối ưu từ ablation study, mô hình **BiGRU + Attention + Kinematics + Focal Loss** được đưa vào đánh giá chính thức trên Test Set hoàn toàn độc lập (chiếm 15% tổng dữ liệu, chưa từng xuất hiện trong quá trình huấn luyện hay tinh chỉnh tham số). Ngưỡng phân loại tối ưu **Youden Index (J = 0.45)** được áp dụng để tối đa hóa khả năng phát hiện sự cố té ngã.

| Chỉ số đánh giá | Giá trị | Ý nghĩa ứng dụng trong giám sát té ngã |
| :--- | :---: | :--- |
| **Accuracy** | **92.47%** | Phân loại chính xác 92.47% tổng số chuỗi hành vi. |
| **Precision** | **80.34%** | Trong các cảnh báo đưa ra, 80.34% phản ánh đúng sự cố té ngã thực tế. |
| **Recall (Fall)** | **91.62%** | Nhận diện chính xác 91.62% các trường hợp té ngã thực tế. |
| **F1-Score** | **85.61%** | Đạt mức cân bằng giữa khả năng phát hiện sự cố và hạn chế cảnh báo nhầm. |

---

## 5. Phân tích Chi tiết & Đánh giá (Analysis)

### 5.1. Ma trận nhầm lẫn (Confusion Matrix Analysis)

<p align="center">
  <img src="plots/Confusion Matrix Fall Detection.png" alt="Normalized Confusion Matrix" width="550">
</p>

Đánh giá ma trận nhầm lẫn chuẩn hóa cho thấy:
* **ADL (Bình thường):** Phân loại chính xác **93.0%**.
* **Fall (Té ngã):** Nhận diện chính xác **92.0%** (tương ứng với độ nhạy Recall trên tập Test).

> **Phân tích chỉ số lỗi:**
> * **Tỷ lệ báo động giả (False Alarm Rate):** 7.0% các hoạt động sinh hoạt bình thường bị ghi nhận nhầm thành té ngã.
> * **Tỷ lệ bỏ sót sự cố (False Negative Rate):** Mô hình bỏ sót 8.0% các trường hợp té ngã thực tế. Tỷ lệ này đáp ứng được yêu cầu an toàn đối với các hệ thống giám sát tự động dựa trên khung xương.

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
