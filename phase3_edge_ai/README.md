# Phase 3: Edge AI — ONNX Export and Optimization

## What You Will Build

An optimized version of your MNIST CNN that runs faster and on hardware that does not have
PyTorch installed. ONNX (Open Neural Network Exchange) is an open format that decouples the
model from its training framework. You will also apply int8 quantization to shrink the model
and speed up CPU inference.

## Key Concepts

**ONNX** represents a neural network as a computation graph with standardized operators. Any
runtime that implements the ONNX spec (onnxruntime, TensorRT, OpenVINO, CoreML) can execute
the graph. This is how you deploy to mobile, edge devices, or cloud without rewriting the model.

**Dynamic axes** tell the ONNX exporter which dimensions are variable at runtime (typically
batch size and sequence length). Without this the model is compiled for a fixed shape.

**Quantization** reduces weight precision from float32 to int8. For most CNN tasks this costs
less than 1% accuracy while reducing model size by ~4x and improving throughput on CPUs that
have efficient int8 SIMD instructions.

**Benchmarking** is the only way to know whether an optimization actually helps. Always measure
wall-clock latency on realistic batch sizes on the target hardware.

## How to Run

```bash
pip install torch torchvision onnx onnxruntime

# Export the PyTorch checkpoint to ONNX
python export_onnx.py --checkpoint ../phase1_python_ml/mnist_cnn.pth

# Apply int8 quantization
python quantize_model.py --input mnist_cnn.onnx --output mnist_cnn_int8.onnx

# Benchmark PyTorch vs ONNX vs quantized ONNX
python benchmark.py --checkpoint ../phase1_python_ml/mnist_cnn.pth
```

## What to Study Next

- ONNX operator set versions and custom op registration
- Static (calibration-based) vs dynamic quantization
- TensorRT for GPU-optimized inference
- Move to **Phase 4** to store embeddings in a vector database
