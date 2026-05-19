"""
quantize_model.py
-----------------
Applies dynamic int8 quantization to an ONNX model using ONNX Runtime's
built-in quantization tools.

Dynamic quantization: weights are quantized ahead of time (offline), but
activations are quantized on-the-fly per tensor during inference. No
calibration data set is needed — making this the easiest quantization method.

Static quantization (not shown here) quantizes activations using statistics
collected from a representative calibration dataset. It is more accurate but
requires extra setup.

Usage:
    python quantize_model.py
    python quantize_model.py --input mnist_cnn.onnx --output mnist_cnn_int8.onnx
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_dynamic,
)


def quantize(input_path: str, output_path: str):
    print(f"Input model:  {input_path}  ({os.path.getsize(input_path)/1024:.1f} KB)")

    # quantize_dynamic rewrites the ONNX graph, replacing float32 MatMul/Gemm
    # ops with int8 equivalents. Weights are stored as int8 in the file.
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        # weight_type=QuantType.QInt8  uses signed int8 (good for CPUs with AVX-512)
        # weight_type=QuantType.QUInt8 uses unsigned int8 (better for some ARM cores)
        weight_type=QuantType.QInt8,
        # QOperator embeds quantize/dequantize nodes inline with ops (faster)
        # QDQ (Quantize-Dequantize) puts them as explicit nodes (more portable)
        quant_format=QuantFormat.QOperator,
    )

    print(f"Output model: {output_path}  ({os.path.getsize(output_path)/1024:.1f} KB)")
    reduction = (1 - os.path.getsize(output_path) / os.path.getsize(input_path)) * 100
    print(f"Size reduction: {reduction:.1f}%")


def compare_outputs(fp32_path: str, int8_path: str, n_samples: int = 100):
    """
    Check how much the quantized model's predictions differ from the fp32 model.
    For most CNNs on MNIST, accuracy degradation is < 0.5%.
    """
    print(f"\nComparing outputs on {n_samples} random inputs ...")

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    in_name_fp32 = sess_fp32.get_inputs()[0].name
    in_name_int8 = sess_int8.get_inputs()[0].name

    np.random.seed(0)
    data = np.random.randn(n_samples, 1, 28, 28).astype(np.float32)

    out_fp32 = sess_fp32.run(None, {in_name_fp32: data})[0]   # (n, 10)
    out_int8 = sess_int8.run(None, {in_name_int8: data})[0]   # (n, 10)

    # Agreement: do both models predict the same class?
    preds_fp32 = out_fp32.argmax(axis=1)
    preds_int8 = out_int8.argmax(axis=1)
    agreement  = (preds_fp32 == preds_int8).mean() * 100

    # Max logit difference
    max_diff = np.abs(out_fp32 - out_int8).max()

    print(f"  Prediction agreement: {agreement:.1f}%")
    print(f"  Max logit difference: {max_diff:.4f}")

    if agreement < 95:
        print("  Warning: significant accuracy drop. Consider static quantization "
              "with a calibration dataset.")
    else:
        print("  Quantization looks good!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="mnist_cnn.onnx",      help="fp32 ONNX model")
    parser.add_argument("--output", default="mnist_cnn_int8.onnx",  help="int8 output path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run export_onnx.py first.")
        return

    quantize(args.input, args.output)
    compare_outputs(args.input, args.output)

    print(f"\nDone. Quantized model saved to: {os.path.abspath(args.output)}")
    print("Next: python benchmark.py to measure the speedup")


if __name__ == "__main__":
    main()
