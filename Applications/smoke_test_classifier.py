"""Load the supplied BiGRU model and execute a forward pass without Django/YOLO.

Standalone sanity check: verifies the checkpoint loads, config keys exist,
and the model produces a finite, reasonable output on both zero and random input.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ai_models"
CONFIG_PATH = MODEL_DIR / "Best BigRU Attention Config.npy"
WEIGHTS_PATH = MODEL_DIR / "Best BigRU Attention Model.pth"
INPUT_SIZE = 102
SEQ_LEN = 16


class BiGRUAttentionModel(nn.Module):
    def __init__(self, input_size=102, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        weights = torch.softmax(self.attention(out), dim=1)
        return self.fc(torch.sum(out * weights, dim=1))


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # --- locate assets, fail loudly with the exact missing path ---
    missing = [p for p in (CONFIG_PATH, WEIGHTS_PATH) if not p.exists()]
    if missing:
        fail("Missing required file(s):\n" + "\n".join(f"  - {p}" for p in missing))

    # --- load + validate config ---
    config = np.load(CONFIG_PATH, allow_pickle=True).item()
    for key in ("hidden_size", "num_layers"):
        if key not in config:
            fail(f"Config missing required key: '{key}'")
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    threshold = float(config.get("threshold", 0.4577))

    # --- build + load weights ---
    model = BiGRUAttentionModel(INPUT_SIZE, hidden_size, num_layers)
    try:
        weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        print("[WARN] torch.load has no weights_only support in this version; "
              "loading with weights_only=False.", file=sys.stderr)
        weights = torch.load(WEIGHTS_PATH, map_location="cpu")

    try:
        model.load_state_dict(weights)
    except RuntimeError as exc:
        fail(f"state_dict mismatch (check hidden_size/num_layers in config): {exc}")

    model.eval()

    # --- sanity forward passes ---
    torch.manual_seed(0)
    zero_input = torch.zeros(1, SEQ_LEN, INPUT_SIZE)
    random_input = torch.randn(1, SEQ_LEN, INPUT_SIZE)

    with torch.inference_mode():
        zero_out = model(zero_input)
        random_out = model(random_input)

    zero_prob = torch.sigmoid(zero_out).item()
    random_prob = torch.sigmoid(random_out).item()

    if not (np.isfinite(zero_prob) and np.isfinite(random_prob)):
        fail("Model produced non-finite output (NaN/Inf) — check weights or architecture mismatch.")

    print("Model loaded successfully")
    print(f"Input shape        : (1, {SEQ_LEN}, {INPUT_SIZE})")
    print(f"Output shape       : {tuple(zero_out.shape)}")
    print(f"Threshold          : {threshold}")
    print(f"Sigmoid(zero input): {zero_prob:.6f}")
    print(f"Sigmoid(rand input): {random_prob:.6f}")


if __name__ == "__main__":
    main()