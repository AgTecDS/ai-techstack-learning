"""
tensor_exploration.py
---------------------
A hands-on tour of PyTorch tensors, autograd, and broadcasting.
Run this script and read the printed output alongside the comments.
"""

import torch
import torch.nn as nn

# ── 1. Creating tensors ────────────────────────────────────────────────────────

# From a Python list — dtype is inferred
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("Shape:", a.shape)        # torch.Size([2, 3])
print("Dtype:", a.dtype)        # torch.float32
print("Device:", a.device)      # cpu (unless you have a GPU)

# Factory functions
zeros = torch.zeros(3, 4)
ones  = torch.ones(3, 4)
rand  = torch.rand(3, 4)       # uniform [0, 1)
randn = torch.randn(3, 4)      # standard normal

print("\nzeros:\n", zeros)
print("rand:\n",  rand)

# ── 2. Basic operations ────────────────────────────────────────────────────────

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("\nElement-wise add:", x + y)        # [5, 7, 9]
print("Element-wise mul:", x * y)          # [4, 10, 18]
print("Dot product:", torch.dot(x, y))     # 32.0
print("Sum:", x.sum())
print("Mean:", x.mean())
print("Max:", x.max())

# ── 3. Reshaping ───────────────────────────────────────────────────────────────

t = torch.arange(12)           # [0, 1, 2, ..., 11]
print("\nOriginal:", t.shape)   # [12]

t2d = t.reshape(3, 4)
print("Reshaped (3,4):\n", t2d)

# view vs reshape: view requires contiguous memory, reshape handles both
t3d = t.reshape(2, 2, 3)
print("Reshaped (2,2,3):\n", t3d)

# Squeeze removes size-1 dims; unsqueeze adds them
t_col = x.unsqueeze(1)         # shape [3] → [3, 1]
print("\nunsqueeze(1):", t_col.shape)   # [3, 1]
print("squeeze back:", t_col.squeeze().shape)  # [3]

# ── 4. Broadcasting ────────────────────────────────────────────────────────────
# Broadcasting lets you do arithmetic on tensors with compatible but unequal shapes.
# Rules (applied right-to-left on dimensions):
#   - If a dim is 1, it is stretched to match the other.
#   - If a dim is missing, it is treated as 1.

row = torch.tensor([[1.0, 2.0, 3.0]])    # shape [1, 3]
col = torch.tensor([[10.0],              # shape [3, 1]
                    [20.0],
                    [30.0]])

# Result shape: [3, 3] — every row of `col` added to every column of `row`
print("\nBroadcast add [1,3] + [3,1]:\n", row + col)

# ── 5. Autograd — automatic differentiation ────────────────────────────────────
# requires_grad=True tells PyTorch to track operations on this tensor
# so it can compute gradients during backward().

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# Simple computation: y = w*x + b, loss = (y - target)^2
x_scalar = torch.tensor(3.0)
target    = torch.tensor(7.0)

y_pred = w * x_scalar + b       # forward pass
loss   = (y_pred - target) ** 2

print("\nw:", w.item(), "b:", b.item())
print("y_pred:", y_pred.item())
print("loss:", loss.item())

# Compute gradients — fills .grad on all leaf tensors with requires_grad=True
loss.backward()

print("dL/dw:", w.grad.item())  # 2 * (y_pred - target) * x_scalar
print("dL/db:", b.grad.item())  # 2 * (y_pred - target)

# ── 6. Gradient descent step ───────────────────────────────────────────────────
# In a real training loop an optimizer does this, but here we do it manually
# to understand what's happening under the hood.

learning_rate = 0.01

with torch.no_grad():   # do not track this update in the computation graph
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

# Always zero out gradients before the next backward() call!
w.grad.zero_()
b.grad.zero_()

print("\nAfter one gradient step — w:", w.item(), "b:", b.item())

# ── 7. GPU support ─────────────────────────────────────────────────────────────
# Moving tensors to GPU is as simple as .to("cuda") or .cuda()
# All operations then run on the GPU automatically.

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

t_device = torch.randn(4, 4).to(device)
print("Tensor on", t_device.device, "shape:", t_device.shape)

# ── 8. Matrix multiplication (the core of every neural network) ────────────────

A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = A @ B                    # equivalent to torch.matmul(A, B)
print("\nMatrix multiply (3,4) @ (4,5) →", C.shape)

# Batched matmul — first dim is batch
batch_A = torch.randn(8, 3, 4)
batch_B = torch.randn(8, 4, 5)
batch_C = batch_A @ batch_B  # (8, 3, 5)
print("Batched matmul (8,3,4) @ (8,4,5) →", batch_C.shape)

# ── 9. A tiny linear layer by hand ────────────────────────────────────────────
# nn.Linear(in, out) is just:  y = x @ W.T + b
# Let's replicate it manually so the magic is gone.

in_features  = 4
out_features = 3
batch_size   = 2

W = torch.randn(out_features, in_features)
b_vec = torch.zeros(out_features)
x_in  = torch.randn(batch_size, in_features)

y_hand  = x_in @ W.T + b_vec          # (2, 3)
linear  = nn.Linear(in_features, out_features, bias=True)
# Override weights to match our hand-computed ones
with torch.no_grad():
    linear.weight.copy_(W)
    linear.bias.copy_(b_vec)

y_nn = linear(x_in)
print("\nHand vs nn.Linear match:", torch.allclose(y_hand, y_nn))

print("\nDone! All tensor concepts covered. Move on to train_mnist.py.")
