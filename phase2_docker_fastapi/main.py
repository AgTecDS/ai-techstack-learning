"""
main.py
-------
FastAPI application that serves MNIST digit predictions.

Endpoints:
  GET  /health   — liveness check, confirms model is loaded
  POST /predict  — accepts an image file, returns predicted digit + probabilities

The model is loaded once at startup using a lifespan context manager.
This avoids reloading on every request (expensive) while keeping state
out of global variables (which don't work cleanly with async).
"""

import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# Add the parent directory to the path so we can import the model class
# In production you would package this properly, but for learning clarity
# we keep the CNN definition here inline and also importable from phase1.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_python_ml"))

from schemas import ErrorResponse, HealthResponse, PredictionResponse

# ── Model definition (mirrored from phase1) ───────────────────────────────────
# Keeping it here makes this phase self-contained for Docker deployment.

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

# ── Shared application state ──────────────────────────────────────────────────

class ModelState:
    """Holds the loaded model and preprocessing transform."""
    model:   Optional[MnistCNN]         = None
    device:  Optional[torch.device]     = None
    transform: Optional[transforms.Compose] = None

state = ModelState()

# ── Image preprocessing ───────────────────────────────────────────────────────

MNIST_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # ensure single channel
    transforms.Resize((28, 28)),                   # MNIST is always 28x28
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # MNIST statistics
])

# ── Lifespan: load/unload model around the server lifetime ────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before the `yield` runs at startup; after the yield runs at shutdown.
    FastAPI replaced the old @app.on_event("startup") pattern with this approach.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MnistCNN().to(device)
    model.eval()   # disable dropout for inference

    # Load checkpoint if it exists — the path is configurable via env var
    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "../phase1_python_ml/mnist_cnn.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path} "
              f"(val_acc={checkpoint.get('val_acc', '?'):.2f}%)")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}. "
              "Model will use random weights. Run phase1_python_ml/train_mnist.py first.")

    state.model     = model
    state.device    = device
    state.transform = MNIST_TRANSFORM

    yield   # server is running — handle requests

    # Shutdown: release GPU memory
    del state.model
    torch.cuda.empty_cache()
    print("Model unloaded.")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MNIST Digit Classifier",
    description="Classifies handwritten digit images (0-9) using a CNN trained on MNIST.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests so a web frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["ops"],
)
async def health():
    """Returns service status and whether the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=str(state.device) if state.device else "unknown",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify a handwritten digit image",
    tags=["inference"],
)
async def predict(file: UploadFile = File(..., description="PNG/JPEG image of a handwritten digit")):
    """
    Upload a grayscale or RGB image of a handwritten digit.
    Returns the predicted class (0-9), its confidence, and full probability distribution.
    """
    if state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Validate content type
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Send image/png or image/jpeg.",
        )

    # Read raw bytes and decode to PIL image
    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("L")   # grayscale
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    # Preprocess: apply the same transform used during training
    tensor = state.transform(pil_image)  # (1, 28, 28)
    tensor = tensor.unsqueeze(0)         # (1, 1, 28, 28) — add batch dim
    tensor = tensor.to(state.device)

    # Inference — torch.no_grad() skips building the computation graph (faster)
    with torch.no_grad():
        logits = state.model(tensor)                   # (1, 10)
        probs  = F.softmax(logits, dim=1).squeeze(0)   # (10,)

    predicted = int(probs.argmax().item())
    confidence = float(probs[predicted].item())
    probabilities = [round(float(p), 6) for p in probs.tolist()]

    return PredictionResponse(
        predicted_digit=predicted,
        confidence=confidence,
        probabilities=probabilities,
    )
