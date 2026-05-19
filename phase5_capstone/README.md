# Phase 5: Capstone — Image Search App

## What You Will Build

A full image similarity search application. A user uploads a handwritten digit image, the app:
1. Runs ONNX inference to classify it and extract a 128-dim embedding
2. Queries Qdrant for the 5 most visually similar images from a pre-indexed set
3. Displays the results in a Streamlit web UI

This ties together every phase: PyTorch training, FastAPI patterns, ONNX inference, Qdrant
vector search, MinIO storage, and a Streamlit frontend.

## Architecture

```
User browser
    └── Streamlit app (app.py)
            └── pipeline.py
                    ├── ONNX Runtime (inference + embedding extraction)
                    ├── Qdrant (vector similarity search)
                    └── MinIO / S3 (image retrieval for display)
```

The `docker-compose.yml` starts Qdrant, MinIO, and the Streamlit app together.

## Key Concepts

**Embedding extraction** means taking the intermediate layer output of a CNN (before the
final classifier) as a fixed-length vector representing the image's semantic content. Similar
images have embeddings that are close in vector space.

**Similarity search at scale** relies on approximate nearest-neighbor (ANN) indexes like
HNSW (Qdrant's default). Exact search is O(N); HNSW is O(log N) with controllable accuracy.

## How to Run

```bash
# One-command startup (Docker required)
docker compose up --build

# Open http://localhost:8501 in your browser

# Seed the database with MNIST test images first
python pipeline.py --seed 500
```

## What to Study Next

- Replace the CNN embedding with a pretrained vision model (CLIP, ResNet-50)
- Add authentication to the Streamlit app
- Deploy to Kubernetes with a Helm chart
- Implement async batch processing for indexing large datasets
