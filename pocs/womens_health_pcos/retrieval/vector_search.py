"""
Vector similarity search over ChunkMetadata embeddings using pgvector.
Falls back to keyword ILIKE search when embeddings are unavailable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sqlalchemy import or_

from pocs.womens_health_pcos.database.session import SessionLocal
from pocs.womens_health_pcos.database.models import ChunkMetadata, SourcePage, Source


def _has_embeddings(db) -> bool:
    """Check if any chunks have been embedded."""
    return (
        db.query(ChunkMetadata)
        .filter(ChunkMetadata.embedding_status == "complete")
        .limit(1)
        .count()
        > 0
    )


def vector_search(query: str, limit: int = 15) -> list[dict]:
    """
    Search chunks by vector cosine similarity.
    Falls back to keyword ILIKE if no embeddings exist.

    Returns list of dicts: {title, url, source_domain, content_type, content, confidence_score, source_label}
    """
    db = SessionLocal()
    try:
        use_vector = False
        query_embedding = None

        if _has_embeddings(db):
            try:
                from pocs.womens_health_pcos.processing.embed_chunks import embed_query
                query_embedding = embed_query(query)
                use_vector = True
                print("[Vector Search] Using cosine similarity.")
            except Exception as e:
                print(f"[Vector Search] Embedding failed, falling back to ILIKE: {e}")

        if use_vector and query_embedding is not None:
            results = (
                db.query(ChunkMetadata, SourcePage, Source)
                .join(SourcePage, ChunkMetadata.source_id == SourcePage.page_id)
                .join(Source, SourcePage.source_id == Source.source_id)
                .filter(
                    SourcePage.is_active == 1,
                    ChunkMetadata.embedding_status == "complete",
                )
                .order_by(ChunkMetadata.embedding.cosine_distance(query_embedding))
                .limit(limit)
                .all()
            )
        else:
            print("[Vector Search] Falling back to keyword ILIKE search.")
            stop_words = {"what", "is", "for", "the", "a", "an", "and", "or", "to", "of", "in"}
            keywords = [
                kw.strip("?.,!")
                for kw in query.lower().split()
                if len(kw) > 3 and kw not in stop_words
            ]
            if not keywords:
                return []

            conditions = []
            for kw in keywords:
                conditions.append(ChunkMetadata.chunk_text.ilike(f"%{kw}%"))

            results = (
                db.query(ChunkMetadata, SourcePage, Source)
                .join(SourcePage, ChunkMetadata.source_id == SourcePage.page_id)
                .join(Source, SourcePage.source_id == Source.source_id)
                .filter(SourcePage.is_active == 1, or_(*conditions))
                .order_by(Source.trust_level.desc())
                .limit(limit)
                .all()
            )

        output = []
        seen_urls = set()
        for chunk, page, source in results:
            if page.page_url in seen_urls:
                continue
            seen_urls.add(page.page_url)

            source_type = source.source_type.upper()
            content_type = "ANECDOTAL" if source_type == "COMMUNITY" else source_type

            output.append({
                "title": source.source_name or "Medical Resource",
                "url": page.page_url,
                "source_domain": source.base_url.replace("https://", "").replace("http://", "").split("/")[0],
                "content_type": content_type,
                "content": chunk.chunk_text,
                "confidence_score": 90 if source.trust_level == "HIGH" else 70,
                "source_label": source.source_name,
            })

        return output

    except Exception as e:
        print(f"[Vector Search] Error: {e}")
        return []
    finally:
        db.close()
