# Phase 2: FastAPI + Docker

## What You Will Build

A production-style REST API that serves your MNIST CNN. Clients POST an image and receive a
digit prediction with confidence scores. The whole thing runs inside a Docker container so it
is reproducible on any machine.

## Key Concepts

**FastAPI** is a modern Python web framework built on Starlette and Pydantic. It generates
OpenAPI docs automatically, validates request/response shapes via type hints, and handles
async I/O natively. Hit `http://localhost:8000/docs` after starting the server to explore the
interactive Swagger UI.

**Pydantic** enforces data contracts at the boundary of your API. The models in `schemas.py`
define exactly what JSON shapes are valid — bad requests are rejected before your code ever
runs.

**Docker** packages your application, its runtime, and all dependencies into an image. This
eliminates "works on my machine" problems. The multi-stage Dockerfile here first installs
dependencies in a build stage, then copies only what is needed into a slim runtime image,
keeping the final image small.

## How to Run

```bash
# Local development (no Docker)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# With Docker
docker build -t mnist-api .
docker run -p 8000:8000 mnist-api

# Test it
curl -X POST http://localhost:8000/predict \
     -F "file=@some_digit.png"
```

## Files

- `main.py` — FastAPI app with health check and prediction endpoint
- `schemas.py` — Pydantic request/response models
- `Dockerfile` — multi-stage Docker build
- `requirements.txt` — pinned dependencies

## What to Study Next

- FastAPI dependency injection (`Depends`)
- Background tasks and async endpoints
- Docker layer caching and image optimization
- Move to **Phase 3** to export the model to ONNX for faster, lighter inference
