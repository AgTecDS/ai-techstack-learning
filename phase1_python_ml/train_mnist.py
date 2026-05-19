"""
train_mnist.py
--------------
Full PyTorch CNN training pipeline on the MNIST dataset.
Covers: data loading, model definition, training loop, validation, checkpointing.

Usage:
    python train_mnist.py
    python train_mnist.py --epochs 10 --lr 0.001 --batch-size 128
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs",     type=int,   default=5,    help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Mini-batch size")
    parser.add_argument("--checkpoint", type=str,   default="mnist_cnn.pth", help="Path to save model")
    return parser.parse_args()

# ── Model definition ───────────────────────────────────────────────────────────

class MnistCNN(nn.Module):
    """
    A small CNN for MNIST:
      Conv1 (1→32 filters, 3x3) → ReLU → MaxPool (2x2)
      Conv2 (32→64 filters, 3x3) → ReLU → MaxPool (2x2)
      Flatten → Linear(1600→128) → ReLU → Dropout(0.5) → Linear(128→10)

    MNIST images are 1×28×28. After two 2x2 pools the spatial size is 7×7,
    giving 64×7×7 = 3136 features... but we use valid (no-pad) convolutions
    so the spatial dims shrink by 2 each time: 28→26→13→11→5 → 64*5*5=1600.
    """

    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # Regularization: dropout randomly zeros activations during training
        # to prevent co-adaptation of neurons (overfitting)
        self.dropout = nn.Dropout(p=0.5)

        # Classifier head
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)   # 10 digit classes

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))           # (batch, 32, 26, 26)
        x = F.max_pool2d(x, kernel_size=2)  # (batch, 32, 13, 13)

        x = F.relu(self.conv2(x))           # (batch, 64, 11, 11)
        x = F.max_pool2d(x, kernel_size=2)  # (batch, 64, 5, 5)

        x = x.flatten(start_dim=1)          # (batch, 1600)

        x = F.relu(self.fc1(x))             # (batch, 128)
        x = self.dropout(x)                 # applied only in model.train() mode
        x = self.fc2(x)                     # (batch, 10) — raw logits
        return x                            # loss fn handles softmax internally

# ── Data loading ───────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int):
    """
    torchvision.datasets downloads and caches MNIST automatically.
    The transform normalizes pixel values from [0,255] → standard normal
    using the MNIST mean (0.1307) and std (0.3081) — precomputed constants.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # num_workers > 0 uses multiprocessing for data loading — speeds up GPU training
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader

# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()   # enables dropout, batch norm training behaviour
    total_loss  = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # 1. Zero gradients — if you skip this, gradients accumulate across batches
        optimizer.zero_grad()

        # 2. Forward pass
        logits = model(images)

        # 3. Compute loss — CrossEntropyLoss = log_softmax + NLLLoss
        loss = criterion(logits, labels)

        # 4. Backward pass — computes d(loss)/d(every parameter)
        loss.backward()

        # 5. Update weights using computed gradients
        optimizer.step()

        # Bookkeeping
        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 200 == 0:
            running_acc = total_correct / total_samples * 100
            print(f"  Epoch {epoch} | step {batch_idx+1}/{len(loader)} "
                  f"| loss {loss.item():.4f} | acc {running_acc:.1f}%")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

# ── Validation loop ────────────────────────────────────────────────────────────

@torch.no_grad()   # decorator disables gradient tracking for the whole function
def evaluate(model, loader, criterion, device):
    model.eval()   # disables dropout, uses running stats in batch norm
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model weights + optimizer state + metadata."""
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":    val_acc,
    }, path)
    print(f"  Checkpoint saved → {path}")

def load_checkpoint(model, optimizer, path, device):
    """Resume training from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.2f}%)")
    return checkpoint["epoch"]

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Epochs={args.epochs}  LR={args.lr}  BatchSize={args.batch_size}\n")

    # Data
    train_loader, val_loader = get_dataloaders(args.batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # Model, loss, optimizer
    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler: reduces LR by 0.1 when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    best_val_acc = 0.0
    start_epoch  = 1

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"  Train → loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  Val   → loss: {val_loss:.4f}  acc: {val_acc:.2f}%")

        # Step the scheduler with validation loss
        scheduler.step(val_loss)

        # Save the best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, args.checkpoint)

        print()

    print(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.abspath(args.checkpoint)}")
    print("Next step: export to ONNX in phase3_edge_ai/export_onnx.py")

if __name__ == "__main__":
    main()
