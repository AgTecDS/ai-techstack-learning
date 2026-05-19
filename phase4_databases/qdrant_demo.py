"""
qdrant_demo.py
--------------
Demonstrates Qdrant vector database operations:
  - Create a collection with cosine distance
  - Upsert vectors with payloads
  - Nearest-neighbor search
  - Filtered search (vector similarity + metadata condition)
  - Delete a point

Requires: pip install qdrant-client
Requires: docker run -p 6333:6333 qdrant/qdrant
"""

import random
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ── Connect ────────────────────────────────────────────────────────────────────

# In-memory mode (no Docker needed) — great for testing/CI
# Switch to QdrantClient(host="localhost", port=6333) for a real Qdrant instance
client = QdrantClient(":memory:")

COLLECTION = "mnist_embeddings"
VECTOR_DIM  = 128   # must match the embedding dimension your model produces

# ── 1. Create collection ───────────────────────────────────────────────────────

def create_collection():
    """
    A collection is roughly equivalent to a table in a relational DB.
    VectorParams defines:
      - size: embedding dimensionality
      - distance: the similarity metric used for nearest-neighbor search
        - Distance.COSINE  → angle between vectors (common for NLP embeddings)
        - Distance.DOT     → dot product (used with normalized vectors)
        - Distance.EUCLID  → L2 distance (common for image embeddings)
    """
    # Delete if it already exists (idempotent demo)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION}' (dim={VECTOR_DIM}, metric=cosine)")

# ── 2. Upsert vectors ──────────────────────────────────────────────────────────

def upsert_vectors(n: int = 100):
    """
    A point consists of:
      - id: unique identifier (int or UUID string)
      - vector: the embedding (list of floats, length = VECTOR_DIM)
      - payload: arbitrary JSON metadata stored alongside the vector
    """
    labels = list(range(10))   # digits 0-9
    points = []

    for i in range(n):
        # Simulate an embedding from the penultimate layer of a CNN
        vector = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]

        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={
                "label":     random.choice(labels),
                "source":    random.choice(["mnist_train", "mnist_test", "custom"]),
                "confidence": round(random.uniform(0.7, 0.999), 4),
                "image_path": f"images/digit_{i:04d}.png",
            }
        ))

    # upsert inserts or updates — safe to call multiple times with same IDs
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {n} vectors into '{COLLECTION}'")

# ── 3. Search — nearest neighbors ────────────────────────────────────────────

def search_similar(query_vector, top_k: int = 5):
    """
    Find the top_k most similar vectors to query_vector.
    Returns hits with id, score (cosine similarity), and payload.
    """
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,   # include the metadata in results
    )
    print(f"\nTop {top_k} nearest neighbors:")
    for hit in results:
        print(f"  id={hit.id:3d}  score={hit.score:.4f}  "
              f"label={hit.payload['label']}  source={hit.payload['source']}")
    return results

# ── 4. Filtered search ────────────────────────────────────────────────────────

def search_with_filter(query_vector, label: int, top_k: int = 3):
    """
    Combine vector similarity with a metadata filter.
    This returns the top_k most similar vectors that also have label == label.

    Qdrant evaluates the filter BEFORE the ANN search, so you only get
    results that satisfy both conditions.
    """
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="label",
                    match=MatchValue(value=label),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    print(f"\nTop {top_k} vectors with label={label}:")
    for hit in results:
        print(f"  id={hit.id:3d}  score={hit.score:.4f}  "
              f"label={hit.payload['label']}  confidence={hit.payload['confidence']}")
    return results

# ── 5. Count and scroll ───────────────────────────────────────────────────────

def count_and_scroll():
    """count() returns total points. scroll() pages through all points."""
    total = client.count(collection_name=COLLECTION, exact=True)
    print(f"\nTotal points in collection: {total.count}")

    # Scroll returns points without a query vector — useful for data inspection
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        limit=3,            # page size
        offset=0,
        with_payload=True,
        with_vectors=False, # skip returning vectors to save bandwidth
    )
    print("First 3 points (no vectors):")
    for p in points:
        print(f"  id={p.id}  payload={p.payload}")

# ── 6. Delete a point ─────────────────────────────────────────────────────────

def delete_point(point_id: int):
    client.delete(
        collection_name=COLLECTION,
        points_selector=[point_id],
    )
    print(f"\nDeleted point id={point_id}")
    new_total = client.count(collection_name=COLLECTION, exact=True)
    print(f"New total: {new_total.count}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    create_collection()
    upsert_vectors(n=100)

    # Random query vector simulating a new image embedding
    query = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]

    search_similar(query, top_k=5)
    search_with_filter(query, label=3, top_k=3)
    count_and_scroll()
    delete_point(point_id=0)

    print("\nQdrant demo complete.")
    print("Key takeaway: vector search finds semantically similar items fast,")
    print("even with millions of vectors, using approximate nearest-neighbor indexes.")

if __name__ == "__main__":
    main()
