"""
Seed the PostgreSQL knowledge base with trusted medical content.

Usage:
    python seed_kb.py                    # Seed with default PCOS URLs
    python seed_kb.py --topic "diabetes" # Seed with custom topic URLs
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ingestion.article_scraper import scrape_batch_deep as scrape_articles
from ingestion.youtube_scraper import scrape_batch as scrape_videos


# Default seed URLs for PCOS knowledge base
DEFAULT_ARTICLE_URLS = [
    "https://www.mayoclinic.org/diseases-conditions/pcos/symptoms-causes/syc-20353439",
    "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome",
    "https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/",
    "https://www.cdc.gov/diabetes/about/pcos-and-diabetes.html",
    "https://my.clevelandclinic.org/health/diseases/8316-polycystic-ovary-syndrome-pcos",
    "https://www.endocrine.org/patient-engagement/endocrine-library/pcos",
    "https://pubmed.ncbi.nlm.nih.gov/35255175/",
    "https://www.mayoclinic.org/diseases-conditions/pcos/diagnosis-treatment/drc-20353443",
    "https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/treatment/",
    "https://www.nichd.nih.gov/health/topics/pcos",
]

DEFAULT_VIDEO_URLS = [
    "https://www.youtube.com/watch?v=Gn7GmF8Wa1o",  # Cleveland Clinic PCOS
    "https://www.youtube.com/watch?v=R1mHiIqU0Gw",  # Mayo Clinic PCOS
]

CHUNK_SIZE = 800  # characters per chunk

_HIGH_CONFIDENCE_DOMAINS = {
    "who.int", "cdc.gov", "nih.gov", "nhs.uk",
    "mayoclinic.org", "clevelandclinic.org", "endocrine.org", "eshre.eu"
}


def _get_trust_level(domain: str) -> str:
    if any(domain == d or domain.endswith("." + d) for d in _HIGH_CONFIDENCE_DOMAINS):
        return "HIGH"
    return "MEDIUM"


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of approximately `size` characters, breaking on newlines."""
    chunks = []
    current = []
    current_len = 0
    for line in text.split("\n"):
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > size and current:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]


def _get_or_create_source(db, domain: str, source_name: str, source_type: str, trust_level: str) -> int:
    """Get or create a Source record for a domain. Returns source_id."""
    from pocs.womens_health_pcos.database.models import Source
    base_url = f"https://{domain}"
    existing = db.query(Source).filter(Source.base_url == base_url).first()
    if existing:
        return existing.source_id
    source = Source(
        source_name=source_name,
        base_url=base_url,
        source_type=source_type,
        trust_level=trust_level,
    )
    db.add(source)
    db.flush()
    return source.source_id


def store_records_to_pg(records: list[dict], topic: str) -> int:
    """Store scraped records into the PostgreSQL Source → SourcePage → ChunkMetadata schema."""
    from pocs.womens_health_pcos.database.session import SessionLocal
    from pocs.womens_health_pcos.database.models import SourcePage, ChunkMetadata

    db = SessionLocal()
    stored_pages = 0
    stored_chunks = 0
    try:
        for record in records:
            url = record.get("url", "")
            domain = record.get("source_domain", "")
            content = record.get("content", "")
            title = record.get("title", url)
            content_type = record.get("content_type", "ARTICLE")

            if not url or not content:
                continue

            trust_level = _get_trust_level(domain)
            source_type = "video" if content_type == "VIDEO" else "article"
            source_name = (title or domain)[:200]

            # 1. Get or create Source (domain level)
            source_id = _get_or_create_source(db, domain, source_name, source_type, trust_level)

            # 2. Get or create SourcePage (URL level)
            existing_page = db.query(SourcePage).filter(SourcePage.page_url == url).first()
            if existing_page:
                page_id = existing_page.page_id
                print(f"  [~] Page exists: {url[:70]}")
            else:
                page = SourcePage(
                    source_id=source_id,
                    page_url=url,
                    discovery_method="deep_scrape",
                    crawl_depth=0,
                    page_status="active",
                    is_active=1,
                )
                db.add(page)
                db.flush()
                page_id = page.page_id
                stored_pages += 1
                print(f"  [+] New page: {url[:70]}")

            # 3. Store chunks (skip if this page already has chunks)
            existing_chunks = db.query(ChunkMetadata).filter(
                ChunkMetadata.source_id == page_id
            ).count()
            if existing_chunks == 0:
                for i, chunk_text in enumerate(_chunk_text(content)):
                    db.add(ChunkMetadata(
                        source_id=page_id,
                        chunk_index=i,
                        chunk_text=chunk_text,
                        char_count=len(chunk_text),
                        token_count=len(chunk_text) // 4,  # rough estimate
                        embedding_status="pending",
                    ))
                    stored_chunks += 1

        db.commit()
        print(f"\n[PG Store] Stored {stored_pages} new pages, {stored_chunks} chunks.")
        return stored_pages
    except Exception as e:
        db.rollback()
        print(f"[PG Store] Error: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        db.close()


def seed(article_urls: list[str] = None, video_urls: list[str] = None, topic: str = "PCOS"):
    """Deep-scrape and store content into the PostgreSQL knowledge base."""
    article_urls = article_urls or DEFAULT_ARTICLE_URLS
    video_urls = video_urls or DEFAULT_VIDEO_URLS

    print(f"\n{'='*60}")
    print(f"[Seed KB] Seeding PostgreSQL knowledge base for topic: {topic}")
    print(f"  Article seed URLs: {len(article_urls)} (depth=5, max_pages=50)")
    print(f"  Video URLs: {len(video_urls)}")
    print(f"{'='*60}\n")

    # Phase 1: Deep article scrape
    print("[Phase 1: Deep Article Scrape]")
    article_records = scrape_articles(article_urls, depth=5, max_pages=50)
    for r in article_records:
        r["query_topics"] = topic

    # Phase 2: Videos (shallow — no link following needed for YouTube)
    print("\n[Phase 2: Videos]")
    video_records = scrape_videos(video_urls)
    for r in video_records:
        r["query_topics"] = topic

    # Phase 3: Store everything to PostgreSQL
    print("\n[Phase 3: Storing to PostgreSQL]")
    stored = store_records_to_pg(article_records + video_records, topic)

    print(f"\n{'='*60}")
    print(f"[Seed KB] Complete!")
    print(f"  Deep-scraped: {len(article_records)} articles, {len(video_records)} videos")
    print(f"  New pages stored to PostgreSQL: {stored}")
    print(f"{'='*60}\n")

    return stored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the medical knowledge base")
    parser.add_argument("--topic", default="PCOS", help="Topic label for stored content")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

    seed(topic=args.topic)
