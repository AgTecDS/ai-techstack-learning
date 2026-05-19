# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# We install all dependencies here. In the final image we only copy the
# installed packages — not the build tools (pip cache, gcc, etc.).
# This multi-stage pattern keeps the production image lean.

FROM python:3.11-slim AS builder

WORKDIR /build

# Copy and install dependencies first.
# Docker caches each layer. Putting requirements before application code means
# a code-only change won't re-run the slow pip install step.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Start from the same slim base — no build tools needed at runtime.

FROM python:3.11-slim AS runtime

# Create a non-root user — running as root inside a container is a security risk
RUN useradd --create-home appuser
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py schemas.py ./

# The model checkpoint is mounted at runtime via a volume or copied in.
# Set MODEL_CHECKPOINT env var to the path inside the container.
ENV MODEL_CHECKPOINT=/app/models/mnist_cnn.pth
ENV PYTHONUNBUFFERED=1

# Switch to the non-root user before starting the server
USER appuser

# Expose the port uvicorn will listen on
EXPOSE 8000

# Health check so Docker/Kubernetes knows when the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the ASGI server
# --host 0.0.0.0 makes it reachable from outside the container
# --workers 1 is fine for a single-GPU inference service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
