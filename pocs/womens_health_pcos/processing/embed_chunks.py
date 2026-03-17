"""
Generates embeddings for pending chunks using OpenAI text-embedding-3-small.
Stores the resulting 1536-dim vectors in ChunkMetadata.embedding via pgvector.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
load_dotenv()

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import ChunkMetadata

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def _get_openai_client():
    from openai import OpenAI
    return OpenAI()


def embed_pending_chunks(batch_size: int = BATCH_SIZE) -> int:
    """Embed all chunks with embedding_status='pending'. Returns count embedded."""
    init_db()
    db = SessionLocal()
    client = _get_openai_client()
    total_embedded = 0

    try:
        while True:
            pending = (
                db.query(ChunkMetadata)
                .filter(ChunkMetadata.embedding_status == "pending")
                .limit(batch_size)
                .all()
            )
            if not pending:
                break

            texts = [chunk.chunk_text for chunk in pending]
            print(f"[Embed] Embedding batch of {len(texts)} chunks...")

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            for chunk, emb_data in zip(pending, response.data):
                chunk.embedding = emb_data.embedding
                chunk.embedding_status = "complete"

            db.commit()
            total_embedded += len(pending)
            print(f"[Embed] Embedded {total_embedded} chunks so far.")

    except Exception as e:
        print(f"[Embed] Error: {e}")
        db.rollback()
    finally:
        db.close()

    print(f"[Embed] Done. Total embedded: {total_embedded}")
    return total_embedded


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns 1536-dim vector."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


if __name__ == "__main__":
    embed_pending_chunks()
