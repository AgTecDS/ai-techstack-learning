# Phase 4: Databases — Vector, Relational, Object Storage

## What You Will Build

Three standalone demo scripts that each cover a different storage layer you will need for an
AI system: a vector database for similarity search, a relational database for metadata, and
object storage for large binary files like model checkpoints.

## Key Concepts

**Vector databases** (Qdrant here) store high-dimensional embedding vectors and answer nearest-
neighbor queries: "give me the 10 vectors most similar to this query vector." Traditional
databases cannot do this efficiently. Qdrant also supports payload filters, so you can combine
vector similarity with exact-match conditions.

**Relational databases** (PostgreSQL via asyncpg) are still the right choice for structured
metadata: who uploaded what, when, what label, what accuracy. The asyncio-native asyncpg driver
lets you issue queries without blocking the event loop — essential in a FastAPI service.

**Object storage** (S3-compatible, shown with MinIO) stores large blobs — model files, images,
datasets — cheaply and durably. The boto3 library covers both AWS S3 and local MinIO identically.
Presigned URLs let you grant time-limited access to private objects without sharing credentials.

## How to Run

```bash
# Start local services
docker run -p 6333:6333 qdrant/qdrant
docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
docker run -p 9000:9000 -e MINIO_ROOT_USER=minio -e MINIO_ROOT_PASSWORD=minio123 minio/minio server /data

pip install qdrant-client asyncpg boto3

python qdrant_demo.py
python postgres_demo.py
python s3_demo.py
```

## What to Study Next

- Qdrant collection configuration: distance metrics (cosine, dot, Euclidean)
- PostgreSQL connection pooling with asyncpg.create_pool
- S3 multipart upload for files larger than 5 GB
- Move to **Phase 5** to wire all of these together in the capstone app
