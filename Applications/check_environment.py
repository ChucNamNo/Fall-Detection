"""Quick Windows-friendly environment and model check."""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parent
required_modules = ["django", "cv2", "numpy", "torch", "ultralytics"]
missing = []
print("=" * 58)
print("KIỂM TRA MÔI TRƯỜNG FALLGUARD AI")
print("=" * 58)
for name in required_modules:
    try:
        module = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        missing.append(name)
        print(f"[THIẾU] {name}: {exc}")

assets = [
    ROOT / "ai_models" / "yolov8n-pose.pt",
    ROOT / "ai_models" / "Best_BiGRU_Attention_Model.pth",
    ROOT / "ai_models" / "Best_BiGRU_Attention_Config.npy",
]
for path in assets:
    print(f"[{'OK' if path.exists() else 'THIẾU'}] {path.relative_to(ROOT)}")
    if not path.exists():
        missing.append(str(path))

try:
    import torch
    print(f"[INFO] CUDA khả dụng: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[CẢNH BÁO] Web vẫn chạy bằng CPU nhưng sẽ chậm hơn.")
except Exception:
    pass

if missing:
    print("\nChạy install_dependencies.bat rồi kiểm tra lại.")
    sys.exit(1)
print("\nMôi trường cơ bản hợp lệ. Chạy run_web.bat để mở web.")
