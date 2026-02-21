"""
Step 2/4: Export Big and Little models to ONNX for measurement

Why ONNX export is needed:
- To claim hardware-awareness credibly, we should measure latency using an actual
  inference runtime. FLOPs/MACs often fail to predict real latency because latency
  depends on kernel implementations, memory behavior, and runtime scheduling.
- ONNX Runtime provides a simple, reproducible way to measure "real" latency on a
  target machine (CPU in this demo). This is the minimal hardware-awareness step.

What this script does:
- Loads artifacts/little.pt and artifacts/big.pt
- Reconstructs the models with their saved widths
- Exports each to a fixed ONNX graph:
    artifacts/little.onnx
    artifacts/big.onnx

Why fixed graphs matter:
- Dynamic slicing / slimmable execution inside a single graph is tricky to measure
  fairly without specialized runtimes.
- Exporting two fixed models ensures we measure latency for the exact compute path
  that will be executed at inference time.

Next step:
- measure_latency.py will benchmark these ONNX models and write artifacts/latency.json
  for use in latency-regularized router training.
"""

import torch
from models import SmallCNN

def export(pt_path, onnx_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    model = SmallCNN(widths=tuple(ckpt["widths"]))
    model.load_state_dict(ckpt["state"])
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes=None
    )
    print("Exported:", onnx_path)

if __name__ == "__main__":
    export("artifacts/little.pt", "artifacts/little.onnx")
    export("artifacts/big.pt", "artifacts/big.onnx")
