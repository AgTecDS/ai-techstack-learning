# Phase 1: Python & PyTorch Fundamentals

## What You Will Build

A convolutional neural network (CNN) trained on the MNIST handwritten digit dataset. By the end
of this phase you will have a saved model checkpoint you can carry into Phase 2 for serving.

## Key Concepts

**Tensors** are the core data structure in PyTorch — multidimensional arrays that live on CPU or
GPU and track gradients automatically via autograd. Every neural network operation is a graph of
tensor ops that PyTorch can differentiate.

**CNNs** exploit spatial locality: convolutional layers learn local filters, pooling layers
downsample, and fully-connected layers classify. For images this beats plain MLPs significantly.

**The training loop** is always: zero gradients → forward pass → compute loss → backward pass →
update weights. Understanding this cycle is more important than any framework detail.

**Validation** after each epoch tells you whether the model is overfitting. Watch the gap between
training loss and validation loss.

## How to Run

```bash
pip install torch torchvision
python tensor_exploration.py   # understand tensors first
python train_mnist.py          # trains for 5 epochs, saves mnist_cnn.pth
```

Training on CPU takes ~3-5 minutes. Add `--epochs 10` for a more accurate model.

## Files

- `tensor_exploration.py` — interactive tour of tensor ops, autograd, broadcasting
- `train_mnist.py` — full training pipeline with validation and checkpoint saving

## What to Study Next

- PyTorch documentation: `torch.nn`, `torch.optim`, `torch.utils.data`
- Batch normalization and dropout for regularization
- Learning rate schedulers (`torch.optim.lr_scheduler`)
- Move to **Phase 2** to serve your saved model via a REST API
