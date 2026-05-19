# AI Tech Stack Learning

A hands-on, phase-by-phase curriculum covering the full ML engineering stack —
from training a neural network to deploying it as a containerized, searchable application.

## Structure

| Phase | Topic | What you build |
|-------|-------|----------------|
| `phase1_python_ml/` | PyTorch fundamentals | CNN trained on MNIST |
| `phase2_docker_fastapi/` | FastAPI + Docker | REST API serving the model |
| `phase3_edge_ai/` | ONNX + quantization | Optimized edge-ready model |
| `phase4_databases/` | Qdrant, PostgreSQL, MinIO | Storage layer for AI data |
| `phase5_capstone/` | Full-stack app | Image search with Streamlit |

## Quick Start

```bash
# Clone and set up environment
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Start infrastructure (Docker required)
docker compose up -d postgres qdrant minio

# Begin Phase 1
cd phase1_python_ml
python tensor_exploration.py
python train_mnist.py
```

## Learning Path

Work through the phases in order. Each README explains what to build, the key concepts,
and what to study next. The capstone in Phase 5 connects all the pieces.

## Prerequisites

- Python 3.10+
- Docker Desktop
- 4 GB RAM minimum (8 GB recommended for training)
