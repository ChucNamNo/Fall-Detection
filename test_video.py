import os
import time
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================
# 1. ĐỌC CẤU HÌNH & KHỞI TẠO MODEL (ĐÃ SỬA ĐƯỜNG DẪN CHUẨN THEO ẢNH)
# =============================================================
try:
    config_path = os.path.join(BASE_DIR, 'models', 'Best BigRU Attention Config.npy')
    best_config = np.load(config_path, allow_pickle=True).item()
    H_SIZE         = best_config['hidden_size']
    N_LAYERS       = best_config['num_layers']
    ALERT_THRESHOLD = best_config.get('threshold', 0.4577)
    print(f"--> Cấu hình tối ưu BiGRU-Attention: Hidden={H_SIZE} | Layers={N_LAYERS} | Threshold={ALERT_THRESHOLD:.4f}")
except Exception:
    print("--> [Cảnh báo] Không tìm thấy file config, dùng tham số tối ưu mặc định.")
    H_SIZE, N_LAYERS, ALERT_THRESHOLD = 64, 2, 0.4577


class BiGRUAttentionModel(nn.Module):
    def __init__(self, input_size=102, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        attn_scores = self.attention(out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights, dim=1)
        return self.fc(context)


classifier_model = BiGRUAttentionModel(
    input_size=102, hidden_size=H_SIZE, num_layers=N_LAYERS
).to(device)

model_path = os.path.join(BASE_DIR, 'models', 'Fall Detection BigRU Model.pth')
if os.path.exists(model_path):
    classifier_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"--> Nạp trọng số BiGRU-Attention thành công: {model_path}")
else:
    raise FileNotFoundError(f"Không tìm thấy '{model_path}'. Hãy kiểm tra lại tên file!")
classifier_model.eval()

pose_model = YOLO('yolov8n-pose.pt')


# =============================================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG ĐỘNG HỌC 102 CHIỀU
# =============================================================
prev_raw_flat = None
prev_velocity = None


def normalize_and_extract_kinetics(keypoints):
    global prev_raw_flat, prev_velocity

    hip_center  = (keypoints[11] + keypoints[12]) / 2.0
    shifted_kp  = keypoints - hip_center
    head_y      = keypoints[0][1]
    foot_y      = max(keypoints[15][1], keypoints[16][1])
    height      = max(abs(foot_y - head_y), 1.0)
    current_raw = (shifted_kp / height).flatten()

    if prev_raw_flat is None:
        velocity     = np.zeros(34)
        acceleration = np.zeros(34)
    else:
        velocity     = current_raw - prev_raw_flat
        acceleration = velocity - prev_velocity if prev_velocity is not None else np.zeros(34)

    prev_raw_flat = current_raw.copy()
    prev_velocity = velocity.copy()

    return np.hstack((current_raw, velocity, acceleration))


def reset_kinetics():
    global prev_raw_flat, prev_velocity
    prev_raw_flat = None
    prev_velocity = None


# =============================================================
# 3. MODULE VẼ HUD CHUYÊN NGHIỆP
# =============================================================
def draw_overlay_panel(frame, x, y, w, h, alpha=0.55, color=(10, 10, 10)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_probability_bar(frame, x, y, w, h, prob, threshold):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)

    fill_w = int(w * prob)
    if prob < threshold * 0.6:
        bar_color = (0, 200, 80)       # Xanh lá
    elif prob < threshold:
        bar_color = (0, 200, 220)      # Vàng
    else:
        bar_color = (40, 40, 220)      # Đỏ

    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), bar_color, -1)

    thresh_x = x + int(w * threshold)
    cv2.line(frame, (thresh_x, y - 3), (thresh_x, y + h + 3), (255, 255, 255), 2)

    pct_text = f"{prob * 100:.1f}%"
    cv2.putText(frame, pct_text, (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame, fall_prob, is_fall, fall_count, fps, frame_idx, total_frames, threshold, person_detected):
    H, W = frame.shape[:2]

    panel_w, panel_h = 260, 130
    draw_overlay_panel(frame, 10, 10, panel_w, panel_h, alpha=0.6)

    cv2.putText(frame, "FALL DETECT [BiGRU-Attn]", (18, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (18, 36), (18 + panel_w - 16, 36), (80, 80, 80), 1)

    if not person_detected:
        status_text  = "NO PERSON"
        status_color = (130, 130, 130)
    elif is_fall:
        status_text  = " FALL DETECTED "
        status_color = (40, 40, 220)
    else:
        status_text  = " NORMAL "
        status_color = (0, 190, 80)

    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    badge_x, badge_y = 18, 42
    cv2.rectangle(frame, (badge_x - 3, badge_y),
                  (badge_x + tw + 6, badge_y + th + 6), status_color, -1)
    cv2.putText(frame, status_text, (badge_x + 1, badge_y + th + 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    cv2.putText(frame, "Prob:", (18, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    draw_probability_bar(frame, x=18, y=88, w=190, h=10, prob=fall_prob, threshold=threshold)

    cv2.putText(frame, f"Thr:{threshold:.2f}  Falls:{fall_count}  FPS:{fps:.0f}",
                (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    if is_fall and (frame_idx % 20 < 10):
        cv2.circle(frame, (W - 30, 30), 14, (0, 0, 210), -1)
        cv2.putText(frame, "!", (W - 36, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if is_fall:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 190), 4)

    bar_y   = H - 14
    bar_h   = 5
    bar_pad = 10
    draw_overlay_panel(frame, 0, H - 22, W, 22, alpha=0.5)
    cv2.rectangle(frame, (bar_pad, bar_y), (W - bar_pad, bar_y + bar_h), (55, 55, 55), -1)
    if total_frames > 0:
        prog_w = int((W - 2 * bar_pad) * min(frame_idx / total_frames, 1.0))
        cv2.rectangle(frame, (bar_pad, bar_y), (bar_pad + prog_w, bar_y + bar_h), (90, 170, 240), -1)
    pct = min(int(frame_idx / max(total_frames, 1) * 100), 100)
    cv2.putText(frame, f"{pct}%", (W - bar_pad - 28, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1, cv2.LINE_AA)

    return frame


# =============================================================
# 4. TIẾN TRÌNH XỬ LÝ VIDEO CHÍNH
# =============================================================
def process_video(input_video_path, output_video_path):
    reset_kinetics()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở video: {input_video_path}")
        return

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_src, (width, height))

    skeleton_buffer = deque(maxlen=MAX_LEN)

    # Hệ thống lọc nhiễu Temporal Smoothing bằng cơ chế bỏ phiếu 
    decision_buffer = deque(maxlen=8)
    VOTE_THRESHOLD  = 0.625

    fall_prob      = 0.0
    is_fall        = False
    fall_count     = 0
    prev_is_fall   = False
    frame_idx      = 0

    fps_display    = 0.0
    fps_counter    = 0
    fps_timer      = time.time()

    print(f"\n[Start] Xử lý với BiGRU-Attention + Smoothing: '{input_video_path}' ({total_frames} frames @ {fps_src:.1f}fps)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx  += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()

        person_detected  = False
        current_features = None

        results = pose_model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            xy = results[0].keypoints.xy
            if len(xy) > 0:
                kp = xy[0].cpu().numpy()
                if kp.shape == (17, 2) and np.sum(kp) > 0:
                    person_detected  = True
                    current_features = normalize_and_extract_kinetics(kp)

        if current_features is not None:
            skeleton_buffer.append(current_features)
        else:
            reset_kinetics()
            if skeleton_buffer:
                skeleton_buffer.append(np.zeros(102))

        raw_is_fall = False

        if len(skeleton_buffer) == MAX_LEN:
            input_tensor = torch.tensor(
                [np.array(skeleton_buffer, dtype=np.float32)],
                dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                fall_prob = torch.sigmoid(classifier_model(input_tensor)).item()

            raw_is_fall = fall_prob >= ALERT_THRESHOLD
            decision_buffer.append(raw_is_fall)
        else:
            decision_buffer.append(False)

        if len(decision_buffer) > 0:
            fall_ratio = sum(decision_buffer) / len(decision_buffer)
            is_fall = fall_ratio >= VOTE_THRESHOLD
        else:
            is_fall = False

        if is_fall and not prev_is_fall:
            fall_count += 1
        prev_is_fall = is_fall

        if person_detected:
            frame = results[0].plot(conf=False, labels=False)

        frame = draw_hud(
            frame,
            fall_prob      = fall_prob,
            is_fall        = is_fall,
            fall_count     = fall_count,
            fps            = fps_display,
            frame_idx      = frame_idx,
            total_frames   = total_frames,
            threshold      = ALERT_THRESHOLD,
            person_detected= person_detected
        )

        out_writer.write(frame)
        cv2.imshow("BiGRU-Attention + Smoothing — Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("--> Dừng sớm theo yêu cầu.")
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"[OK] Video kết quả đã xuất tại: '{output_video_path}'")
    print(f"[OK] Tổng số sự kiện Fall phát hiện: {fall_count}")


if __name__ == "__main__":
    INPUT_VIDEO  = os.path.join(BASE_DIR, "sample1.mp4")
    OUTPUT_VIDEO = os.path.join(BASE_DIR, "test_fall_v2_result_sample1.mp4")

    if not os.path.exists(INPUT_VIDEO):
        print(f"[Lưu ý] Đặt file video kiểm thử tại: '{INPUT_VIDEO}'")
    else:
        process_video(INPUT_VIDEO, OUTPUT_VIDEO)
