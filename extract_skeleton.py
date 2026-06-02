import os
import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO('yolov8n-pose.pt')

DATASET_DIR = "./Dataset"
OUTPUT_DIR = "./data_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 16


def normalize_skeleton(keypoints):
    hip_center = (keypoints[11] + keypoints[12]) / 2.0
    shifted_kp = keypoints - hip_center
    head_y = keypoints[0][1]
    foot_y = max(keypoints[15][1], keypoints[16][1])
    height = max(abs(foot_y - head_y), 1.0)
    return shifted_kp / height


def parse_annotation_advanced(annotation_path):
    fall_frames = set()
    start_fall, end_fall = -1, -1

    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()

            clean_lines = [line.strip() for line in lines if line.strip()]

            # Bước 1: Thử lấy mốc từ 2 dòng đầu
            if len(clean_lines) >= 2:
                if clean_lines[0].isdigit() and clean_lines[1].isdigit():
                    start_fall = int(clean_lines[0])
                    end_fall = int(clean_lines[1])
                    for f_idx in range(start_fall, end_fall + 1):
                        fall_frames.add(f_idx)

            # Bước 2: Quét kiểm tra mã trạng thái chi tiết từng dòng để tránh sót nhãn lạ (như video 49)
            for line in clean_lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        f_idx = int(parts[0].strip())
                        status_code = int(parts[1].strip())
                        # Quy ước các mã trạng thái liên quan đến hành vi ngã trong IMVIA Le2i
                        if status_code in [2, 4, 7, 8]:
                            fall_frames.add(f_idx)
                    except ValueError:
                        continue  # Bỏ qua 2 dòng đầu nếu nó không có dấu phẩy
        except Exception as e:
            pass

    return fall_frames


def process_imvia_dataset():
    all_sequences = []
    all_labels = []

    print("================ BẮT ĐẦU TRÍCH XUẤT SEQUENCE TỪ VIDEO IMVIA ================")

    # Duyệt qua các thư mục phòng (Coffee_room_01, Home_01,...)
    for room_dir in sorted(os.listdir(DATASET_DIR)):
        room_path = os.path.join(DATASET_DIR, room_dir)
        if not os.path.isdir(room_path):
            continue

        video_folder = os.path.join(room_path, "Videos")
        annotation_folder = os.path.join(room_path, "Annotation_files")

        if not os.path.exists(video_folder):
            continue

        print(f"\n--> Đang quét bối cảnh không gian: {room_dir}")

        # Duyệt qua từng file video .avi
        for video_name in sorted(os.listdir(video_folder)):
            if not video_name.lower().endswith('.avi'):
                continue

            video_path = os.path.join(video_folder, video_name)
            base_name = os.path.splitext(video_name)[0]

            # Gọi hàm bóc tách tập hợp frame ngã nâng cao
            annotation_path = os.path.join(annotation_folder, base_name + ".txt")
            fall_frames_set = parse_annotation_advanced(annotation_path)

            print(f"   Xử lý: {video_name} | Số lượng frame phát hiện hành vi ngã: {len(fall_frames_set)}")

            # Đọc video bằng OpenCV
            cap = cv2.VideoCapture(video_path)
            video_frames_skeletons = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                current_skeleton = None

                # Trích xuất pose từ YOLOv8
                results = pose_model(frame, verbose=False)
                for r in results:
                    if r.keypoints is not None and len(r.keypoints.xy) > 0:
                        kp = r.keypoints.xy[0].cpu().numpy()
                        if kp.shape == (17, 2):
                            current_skeleton = normalize_skeleton(kp)
                            break

                if current_skeleton is None:
                    current_skeleton = np.zeros((17, 2))

                video_frames_skeletons.append((frame_idx, current_skeleton))

            cap.release()

            # Sử dụng kỹ thuật Cửa sổ trượt (Sliding Window) để tạo chuỗi dữ liệu 16 frames
            for i in range(0, len(video_frames_skeletons) - WINDOW_SIZE + 1, 4):  # Step = 4
                window = video_frames_skeletons[i: i + WINDOW_SIZE]

                # Trích xuất riêng bộ xương ra mảng
                window_skeletons = [item[1] for item in window]

                # KIỂM TRA NHÃN CHUỖI: Nếu có bất kỳ frame nào nằm trong tập hợp fall_frames_set
                is_fall_sequence = any(item[0] in fall_frames_set for item in window)
                label = 1 if is_fall_sequence else 0

                all_sequences.append(window_skeletons)
                all_labels.append(label)

    if len(all_sequences) == 0:
        print("\n[Lỗi] Không trích xuất được chuỗi dữ liệu nào. Hãy kiểm tra lại cấu trúc!")
        return

    # Lưu mảng nén mượt mà ra file
    np.save(os.path.join(OUTPUT_DIR, 'sequences.npy'), np.array(all_sequences, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), np.array(all_labels))

    all_labels_np = np.array(all_labels)
    print(f"\n================ HOÀN THÀNH TIỀN XỬ LÝ IMVIA TỐI ƯU ENHANCED ================")
    print(f" Tổng số chuỗi thời gian (Mẫu) thu hoạch được: {len(all_sequences)}")
    print(f" -> Số mẫu Bình thường (Nhãn 0): {np.sum(all_labels_np == 0)}")
    print(f" -> Số mẫu Té ngã (Nhãn 1): {np.sum(all_labels_np == 1)}")
    print(f" Dữ liệu đã lưu an toàn tại thư mục: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    process_imvia_dataset()