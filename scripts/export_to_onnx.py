import sys
from pathlib import Path
import torch

# allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from lstm_model import NextTokenLSTM

def export(weights_path: str, onnx_path: str, seq_size: int = 64, vocab_size: int = 30522):
    """Load PyTorch weights and export NextTokenLSTM to ONNX."""
    device = torch.device("cpu")
    model = NextTokenLSTM(vocab_size=vocab_size)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros((1, seq_size), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=17,
    )

if __name__ == "__main__":
    weights = ROOT / "models/next_token_64_250826_201234.pth"
    onnx_out = ROOT / "models/next_token_64.onnx"
    export(str(weights), str(onnx_out))
