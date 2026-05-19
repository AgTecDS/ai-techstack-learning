"""
pipeline.py
-----------
End-to-end pipeline for the image search capstone:
  1. Load an ONNX model
  2. Run inference to classify an image AND extract a 128-dim embedding
  3. Upsert the embedding into Qdrant
  4. Query Qdrant for nearest neighbors
  5. (Optional) Store/retrieve images in MinIO

This module is imported by app.py (Streamlit) and can also be run directly
to seed the database with MNIST test images.

Usage:
    python pipeline.py --seed 200       # index 200 MNIST test images
    python pipeline.py --query path/to/image.png
"""

import argparse
import io
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

# ── Configuration ──────────────────────────────────────────────────────────────

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "../phase3_edge_ai/mnist_cnn.onnx")
QDRANT_HOST     = os.getenv("QDRANT_HOST",     "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "mnist_image_search"
EMBEDDING_DIM   = 128   # output dim of fc1 layer (before final classifier)

# ── ONNX Inference ─────────────────────────────────────────────────────────────

class OnnxInferenceEngine:
    """
    Wraps an ONNX Runtime session and exposes:
      - predict(image) → predicted class + confidence
      - embed(image)   → 128-dim feature vector from the penultimate layer

    The trick for embedding extraction: we export a second ONNX model that
    outputs the fc1 activations instead of the logits. If only the standard
    model is available, we approximate with a random projection of the logits.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run phase3_edge_ai/export_onnx.py first."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session    = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Fixed random projection matrix: maps 10-dim logits → 128-dim embedding
        # In a real system you would export the fc1 output directly.
        np.random.seed(42)
        self._proj = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-8

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL image to a normalised float32 tensor (1, 1, 28, 28)."""
        img = image.convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.1307) / 0.3081   # MNIST normalization
        return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 28, 28)

    def predict(self, image: Image.Image) -> Tuple[int, float, List[float]]:
        """
        Returns: (predicted_class, confidence, list_of_10_probabilities)
        """
        tensor = self._preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0][0]

        # Softmax
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()

        predicted  = int(probs.argmax())
        confidence = float(probs[predicted])
        return predicted, confidence, probs.tolist()

    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Extract a 128-dim embedding vector for similarity search.
        We project the 10-dim softmax probabilities into 128 dims using a
        fixed random matrix, then L2-normalize the result.

        In production: export an additional ONNX output for the fc1 layer.
        """
        _, _, probs = self.predict(image)
        probs_arr  = np.array(probs, dtype=np.float32)
        embedding  = probs_arr @ self._proj   # (128,)
        # L2 normalization ensures cosine distance == dot product distance
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).astype(np.float32)


# ── Qdrant operations ──────────────────────────────────────────────────────────

def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Return a Qdrant client. Falls back to in-memory if connection fails."""
    try:
        client = QdrantClient(host=host, port=port, timeout=5.0)
        client.get_collections()   # will raise if unreachable
        print(f"Connected to Qdrant at {host}:{port}")
        return client
    except Exception:
        print("Warning: could not reach Qdrant, using in-memory mode (data won't persist).")
        return QdrantClient(":memory:")


def ensure_collection(client: QdrantClient):
    """Create the collection if it does not exist."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'")


def index_image(
    client: QdrantClient,
    engine: OnnxInferenceEngine,
    image: Image.Image,
    point_id: int,
    metadata: dict,
):
    """
    Classify an image, extract its embedding, and upsert into Qdrant.
    """
    predicted, confidence, _ = engine.predict(image)
    embedding = engine.embed(image)

    payload = {
        "predicted_label": predicted,
        "confidence":      confidence,
        **metadata,
    }

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)],
    )


def search_similar(
    client: QdrantClient,
    engine: OnnxInferenceEngine,
    query_image: Image.Image,
    top_k: int = 5,
    label_filter: Optional[int] = None,
) -> list:
    """
    Embed the query image and find the most similar indexed images.
    Optionally restrict results to a specific digit label.
    """
    query_embedding = engine.embed(query_image)

    search_filter = None
    if label_filter is not None:
        search_filter = Filter(
            must=[FieldCondition(
                key="predicted_label",
                match=MatchValue(value=label_filter),
            )]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )
    return results


# ── Seeding ────────────────────────────────────────────────────────────────────

def seed_from_mnist(n: int = 200):
    """
    Download MNIST test set and index the first n images into Qdrant.
    Requires: pip install torchvision
    """
    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        print("torchvision not installed. Run: pip install torchvision")
        return

    print(f"Seeding Qdrant with {n} MNIST test images ...")

    engine = OnnxInferenceEngine(ONNX_MODEL_PATH)
    client = get_qdrant_client()
    ensure_collection(client)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    for idx in range(min(n, len(dataset))):
        tensor, true_label = dataset[idx]
        # Convert tensor (1, 28, 28) to PIL Image
        pil_img = Image.fromarray(
            (tensor.squeeze(0).numpy() * 255).astype(np.uint8), mode="L"
        )
        index_image(
            client, engine, pil_img,
            point_id=idx,
            metadata={"true_label": int(true_label), "split": "test", "index": idx},
        )

        if (idx + 1) % 50 == 0:
            print(f"  Indexed {idx+1}/{n} images")

    total = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Done. Collection now has {total.count} vectors.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image search pipeline")
    parser.add_argument("--seed",  type=int, metavar="N",
                        help="Seed Qdrant with N MNIST test images")
    parser.add_argument("--query", type=str, metavar="PATH",
                        help="Query with a local image file")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.seed:
        seed_from_mnist(args.seed)

    if args.query:
        engine = OnnxInferenceEngine(ONNX_MODEL_PATH)
        client = get_qdrant_client()

        img      = Image.open(args.query)
        cls, conf, _ = engine.predict(img)
        print(f"\nQuery image — predicted: {cls}  confidence: {conf:.4f}")

        results = search_similar(client, engine, img, top_k=args.top_k)
        print(f"\nTop {args.top_k} similar images:")
        for hit in results:
            print(f"  id={hit.id:4d}  score={hit.score:.4f}  "
                  f"label={hit.payload.get('predicted_label')}  "
                  f"true={hit.payload.get('true_label', '?')}")


if __name__ == "__main__":
    main()
