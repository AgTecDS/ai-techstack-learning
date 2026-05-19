"""
export_onnx.py
--------------
Exports a trained PyTorch MNIST CNN to ONNX format.

ONNX export works by tracing the model with a dummy input tensor.
PyTorch records every operation on that tensor and serializes the resulting
computation graph into the ONNX protobuf format.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint ../phase1_python_ml/mnist_cnn.pth --output mnist_cnn.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np

# ── Model (same architecture as phase1) ───────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3)
        self.conv2   = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64 * 5 * 5, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ── Export ─────────────────────────────────────────────────────────────────────

def export(checkpoint_path: str, output_path: str, opset: int = 17):
    device = torch.device("cpu")   # export on CPU for maximum compatibility
    model  = MnistCNN().to(device)
    model.eval()  # CRITICAL: eval mode disables dropout during tracing

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: no checkpoint at {checkpoint_path}, exporting with random weights.")

    # Dummy input: batch_size=1, channels=1, height=28, width=28
    # The values don't matter for export — only the shape and dtype do.
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    # dynamic_axes lets ONNX Runtime accept any batch size at inference time.
    # Without this the model is compiled for batch_size=1 only.
    dynamic_axes = {
        "input":  {0: "batch_size"},   # dim 0 is dynamic
        "output": {0: "batch_size"},
    }

    print(f"Exporting to {output_path} with opset={opset} ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,            # bake trained weights into the file
        opset_version=opset,           # ONNX operator set version
        do_constant_folding=True,      # fold constant expressions at export time
        input_names=["input"],         # name the input node (visible in Netron)
        output_names=["output"],       # name the output node
        dynamic_axes=dynamic_axes,
    )
    print("Export complete.")

    # ── Verify the exported model ──────────────────────────────────────────────
    # onnx.checker.check_model validates the graph structure against the ONNX spec.
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed.")

    # Print model metadata
    print(f"  IR version:      {model_onnx.ir_version}")
    print(f"  Opset version:   {model_onnx.opset_import[0].version}")
    print(f"  Graph inputs:    {[i.name for i in model_onnx.graph.input]}")
    print(f"  Graph outputs:   {[o.name for o in model_onnx.graph.output]}")
    print(f"  File size:       {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path

# ── Verify with ONNX Runtime ───────────────────────────────────────────────────

def verify_onnxruntime(onnx_path: str, checkpoint_path: str):
    """
    Run the same input through PyTorch and ONNX Runtime, compare outputs.
    They should match to within floating-point rounding error (~1e-5).
    """
    print("\nVerifying ONNX Runtime output matches PyTorch ...")

    # Load PyTorch model
    device = torch.device("cpu")
    pt_model = MnistCNN().to(device)
    pt_model.eval()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        pt_model.load_state_dict(ckpt["model_state_dict"])

    # Create ONNX Runtime inference session
    # Providers define the execution backend: CPUExecutionProvider, CUDAExecutionProvider, etc.
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Get the input name (defined during export as "input")
    input_name = sess.get_inputs()[0].name

    # Run on a fixed random batch
    np.random.seed(42)
    batch = np.random.randn(4, 1, 28, 28).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(batch)).numpy()

    # ONNX Runtime forward
    ort_out = sess.run(None, {input_name: batch})[0]

    # Compare
    max_diff = np.abs(pt_out - ort_out).max()
    print(f"  Max absolute difference (PyTorch vs ONNX Runtime): {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs diverged too much: {max_diff}"
    print("  Outputs match. Export verified.")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../phase1_python_ml/mnist_cnn.pth")
    parser.add_argument("--output",     default="mnist_cnn.onnx")
    parser.add_argument("--opset",      type=int, default=17)
    args = parser.parse_args()

    onnx_path = export(args.checkpoint, args.output, args.opset)
    verify_onnxruntime(onnx_path, args.checkpoint)

    print(f"\nSuccess! ONNX model at: {os.path.abspath(onnx_path)}")
    print("Next: python quantize_model.py to apply int8 quantization")

if __name__ == "__main__":
    main()
