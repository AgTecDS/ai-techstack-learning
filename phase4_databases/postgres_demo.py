"""
postgres_demo.py
----------------
asyncpg-based CRUD operations for an embeddings metadata table in PostgreSQL.

asyncpg is the fastest PostgreSQL driver for Python. It speaks the binary
PostgreSQL wire protocol directly and integrates natively with asyncio, so
it never blocks the event loop in an async web server.

Requires:
    pip install asyncpg
    docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

import asyncpg

# ── Configuration ──────────────────────────────────────────────────────────────

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS embedding_metadata (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_path  TEXT        NOT NULL,
    label       INTEGER     NOT NULL CHECK (label >= 0 AND label <= 9),
    model_name  TEXT        NOT NULL,
    confidence  FLOAT       NOT NULL,
    qdrant_id   INTEGER,                 -- foreign key into Qdrant collection
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    extra       JSONB                    -- flexible metadata field
);

-- Index for fast lookup by label
CREATE INDEX IF NOT EXISTS idx_embedding_label ON embedding_metadata (label);
-- Index for full-text / JSON queries
CREATE INDEX IF NOT EXISTS idx_embedding_extra ON embedding_metadata USING GIN (extra);
"""

# ── CRUD functions ─────────────────────────────────────────────────────────────

async def create_tables(conn: asyncpg.Connection):
    await conn.execute(CREATE_TABLE_SQL)
    print("Tables created (or already exist).")


async def insert_embedding(
    conn: asyncpg.Connection,
    image_path: str,
    label: int,
    model_name: str,
    confidence: float,
    qdrant_id: Optional[int] = None,
    extra: Optional[dict]    = None,
) -> UUID:
    """
    INSERT a new metadata row and return the generated UUID.
    asyncpg automatically maps Python types to PostgreSQL types:
      str   → TEXT
      int   → INTEGER
      float → FLOAT8
      dict  → JSONB (when you pass json.dumps)
    """
    row_id = await conn.fetchval(
        """
        INSERT INTO embedding_metadata
            (image_path, label, model_name, confidence, qdrant_id, extra)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        RETURNING id
        """,
        image_path,
        label,
        model_name,
        confidence,
        qdrant_id,
        json.dumps(extra) if extra else None,
    )
    return row_id


async def get_by_id(conn: asyncpg.Connection, row_id: UUID) -> Optional[asyncpg.Record]:
    """SELECT a single row by primary key. Returns None if not found."""
    return await conn.fetchrow(
        "SELECT * FROM embedding_metadata WHERE id = $1",
        row_id,
    )


async def list_by_label(
    conn: asyncpg.Connection, label: int, limit: int = 10
) -> List[asyncpg.Record]:
    """Return up to `limit` rows for a given digit label, newest first."""
    return await conn.fetch(
        """
        SELECT id, image_path, label, model_name, confidence, created_at
        FROM   embedding_metadata
        WHERE  label = $1
        ORDER  BY created_at DESC
        LIMIT  $2
        """,
        label, limit,
    )


async def update_qdrant_id(
    conn: asyncpg.Connection, row_id: UUID, qdrant_id: int
) -> bool:
    """UPDATE the qdrant_id column for a given row. Returns True if a row was updated."""
    result = await conn.execute(
        "UPDATE embedding_metadata SET qdrant_id = $1 WHERE id = $2",
        qdrant_id, row_id,
    )
    # execute() returns a string like "UPDATE 1"
    rows_affected = int(result.split()[-1])
    return rows_affected > 0


async def delete_embedding(conn: asyncpg.Connection, row_id: UUID) -> bool:
    result = await conn.execute(
        "DELETE FROM embedding_metadata WHERE id = $1",
        row_id,
    )
    return int(result.split()[-1]) > 0


async def aggregate_stats(conn: asyncpg.Connection):
    """
    GROUP BY query showing average confidence per label.
    asyncpg returns asyncpg.Record objects — access columns by name like a dict.
    """
    rows = await conn.fetch(
        """
        SELECT   label,
                 COUNT(*)               AS total,
                 AVG(confidence)        AS avg_confidence,
                 MAX(confidence)        AS max_confidence
        FROM     embedding_metadata
        GROUP BY label
        ORDER BY label
        """
    )
    return rows


# ── Connection pool demo ───────────────────────────────────────────────────────

async def demo_with_pool():
    """
    In production, use a connection pool instead of a single connection.
    The pool maintains a set of open connections and lends them to coroutines,
    dramatically reducing connection overhead.
    """
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)

    async with pool.acquire() as conn:
        await create_tables(conn)

        # Insert sample rows
        ids = []
        for i in range(5):
            row_id = await insert_embedding(
                conn,
                image_path=f"images/sample_{i:04d}.png",
                label=i % 10,
                model_name="mnist_cnn_v1",
                confidence=0.95 + i * 0.005,
                qdrant_id=i * 10,
                extra={"augmented": i % 2 == 0, "split": "test"},
            )
            ids.append(row_id)
            print(f"Inserted row id={row_id}  label={i % 10}")

        # Read one back
        row = await get_by_id(conn, ids[0])
        print(f"\nFetched by id: {dict(row)}")

        # List by label
        rows = await list_by_label(conn, label=0, limit=5)
        print(f"\nRows with label=0: {len(rows)}")
        for r in rows:
            print(f"  {r['image_path']}  confidence={r['confidence']:.4f}")

        # Update
        updated = await update_qdrant_id(conn, ids[0], qdrant_id=9999)
        print(f"\nUpdated qdrant_id: {updated}")

        # Aggregate
        stats = await aggregate_stats(conn)
        print("\nAggregate stats:")
        for s in stats:
            print(f"  label={s['label']}  total={s['total']}  "
                  f"avg_conf={s['avg_confidence']:.4f}")

        # Delete
        deleted = await delete_embedding(conn, ids[-1])
        print(f"\nDeleted last row: {deleted}")

    await pool.close()
    print("\nPostgreSQL demo complete.")


async def main():
    try:
        await demo_with_pool()
    except OSError as e:
        print(f"\nCould not connect to PostgreSQL: {e}")
        print("Start PostgreSQL with:")
        print("  docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16")


if __name__ == "__main__":
    asyncio.run(main())
