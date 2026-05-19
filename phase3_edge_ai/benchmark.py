"""
benchmark.py
------------
Measures and compares inference latency for:
  1. PyTorch (float32, CPU)
  2. ONNX Runtime (float32, CPU)
  3. ONNX Runtime (int8, CPU)

Key metrics:
  - p50 (median) latency — typical case
  - p95 latency — worst case for most users
  - p99 latency — tail latency under load
  - Throughput (images/sec)

Usage:
    python benchmark.py
    python benchmark.py --checkpoint ../phase1_python_ml/mnist_cnn.pth --batch-size 32 --runs 200
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model definition ───────────────────────────────────────────────────────────

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


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def percentile(latencies: List[float], p: float) -> float:
    """Return the p-th percentile of a list of latency values (in ms)."""
    sorted_l = sorted(latencies)
    idx = max(0, int(len(sorted_l) * p / 100) - 1)
    return sorted_l[idx]


def print_stats(name: str, latencies: List[float], batch_size: int):
    """Pretty-print latency statistics and throughput."""
    arr   = np.array(latencies)
    mean  = arr.mean()
    p50   = np.percentile(arr, 50)
    p95   = np.percentile(arr, 95)
    p99   = np.percentile(arr, 99)
    throughput = (batch_size / mean) * 1000   # images per second

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Mean latency : {mean:.2f} ms")
    print(f"  p50 latency  : {p50:.2f} ms")
    print(f"  p95 latency  : {p95:.2f} ms")
    print(f"  p99 latency  : {p99:.2f} ms")
    print(f"  Throughput   : {throughput:.1f} images/sec")


def warmup_pytorch(model, dummy, n: int = 10):
    """A few warmup runs let the JIT and OS page caches settle."""
    with torch.no_grad():
        for _ in range(n):
            model(dummy)


def warmup_ort(session, input_name, dummy_np: np.ndarray, n: int = 10):
    for _ in range(n):
        session.run(None, {input_name: dummy_np})


# ── Benchmark functions ────────────────────────────────────────────────────────

def benchmark_pytorch(model, batch_size: int, runs: int) -> List[float]:
    device = torch.device("cpu")
    dummy  = torch.randn(batch_size, 1, 28, 28, device=device)

    model.eval()
    warmup_pytorch(model, dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)   # convert to ms

    return latencies


def benchmark_ort(model_path: str, batch_size: int, runs: int) -> List[float]:
    if not os.path.exists(model_path):
        return []

    # SessionOptions lets you tune threading; IntraOpNumThreads=1 isolates
    # the benchmark to a single CPU core for fair comparison
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    session    = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_np   = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)

    warmup_ort(session, input_name, dummy_np)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_np})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return latencies


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="../phase1_python_ml/mnist_cnn.pth")
    parser.add_argument("--onnx-fp32",     default="mnist_cnn.onnx")
    parser.add_argument("--onnx-int8",     default="mnist_cnn_int8.onnx")
    parser.add_argument("--batch-size",    type=int, default=1)
    parser.add_argument("--runs",          type=int, default=200,
                        help="Number of timed inference calls")
    args = parser.parse_args()

    print(f"Benchmark settings: batch_size={args.batch_size}  runs={args.runs}")

    # ── PyTorch ───────────────────────────────────────────────────────────────
    model = MnistCNN()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    pt_latencies = benchmark_pytorch(model, args.batch_size, args.runs)
    print_stats("PyTorch (float32, CPU)", pt_latencies, args.batch_size)

    # ── ONNX Runtime float32 ──────────────────────────────────────────────────
    ort_fp32_latencies = benchmark_ort(args.onnx_fp32, args.batch_size, args.runs)
    if ort_fp32_latencies:
        print_stats("ONNX Runtime (float32, CPU)", ort_fp32_latencies, args.batch_size)
        speedup_fp32 = np.mean(pt_latencies) / np.mean(ort_fp32_latencies)
        print(f"  Speedup vs PyTorch: {speedup_fp32:.2f}x")
    else:
        print(f"\nSkipping ONNX fp32 — file not found: {args.onnx_fp32}")
        print("  Run export_onnx.py first.")

    # ── ONNX Runtime int8 ─────────────────────────────────────────────────────
    ort_int8_latencies = benchmark_ort(args.onnx_int8, args.batch_size, args.runs)
    if ort_int8_latencies:
        print_stats("ONNX Runtime (int8, CPU)", ort_int8_latencies, args.batch_size)
        speedup_int8 = np.mean(pt_latencies) / np.mean(ort_int8_latencies)
        print(f"  Speedup vs PyTorch: {speedup_int8:.2f}x")
    else:
        print(f"\nSkipping ONNX int8 — file not found: {args.onnx_int8}")
        print("  Run quantize_model.py first.")

    print("\nBenchmark complete.")
    print("Note: latency differences are most visible on larger models and batch sizes.")
    print("Try --batch-size 32 to see throughput gains from batching.")


if __name__ == "__main__":
    main()
