"""Step 1: Clean raw scraped PCOS articles for fine-tuning.

Loads 174 JSONs, strips HTML/nav/ads/scraper noise, deduplicates by URL,
and discards articles with < 200 chars of usable medical content.
"""

import json
import re
from pathlib import Path

from config import RAW_DATA_DIR, CLEANED_DIR, MIN_CLEANED_CHARS

# ── Patterns to strip ──────────────────────────────────────────────────────────

# Scraper debug / fetchCache noise
_FETCH_CACHE_RE = re.compile(
    r"(Post or Recipes debug info:.*?(?=\n[A-Z]))"
    r"|fetchCache\[.*?\]\..*"
    r"|Cache (hit|miss) for key.*"
    r"|CCCache\.dataFetchCount:.*"
    r"|Cache uses for key.*"
    r"|client:\s*\{.*?\}"
    r"|Now:\s*\d+"
    r"|Cache Key:.*"
    r"|--\s*Key:.*seconds remaining.*"
    r"|conditions:.*?(?=\n[A-Z])"
    r"|All fetchCache expiration times:.*",
    re.DOTALL,
)

# Navigation / boilerplate
_NAV_PATTERNS = [
    re.compile(r"Skip to main content.*?(?=\n\n)", re.DOTALL),
    re.compile(r"(?:Expand|Collapse) Navigation.*", re.DOTALL),
    re.compile(r"Search\nAdvertisement\nAdvertisement\n"),
    re.compile(r"Advertisement\n?", re.MULTILINE),
]

# Cookie / consent / signup banners
_COOKIE_RE = re.compile(
    r"(Accept (all )?cookies?|Cookie (policy|settings|preferences)"
    r"|We use cookies|Sign up for our.*?emails?.*?(?=\n\n)"
    r"|Better health starts here.*?(?=\n\n)"
    r"|Example email\nSign up\nSign up\nExample email)",
    re.IGNORECASE | re.DOTALL,
)

# "Related Articles" and everything after
_RELATED_RE = re.compile(
    r"\n(Related Articles|Trending Topics|Health Categories"
    r"|Other Popular Categories|Top Posts|reReddit"
    r"|Reddit Rules|Privacy Policy|User Agreement"
    r"|Accessibility|Reddit, Inc\.).*",
    re.DOTALL,
)

# Editorial / policy boilerplate
_EDITORIAL_RE = re.compile(
    r"(Learn more about our\s*editorial process\.?"
    r"|Cleveland Clinic is a non-profit.*?Policy"
    r"|Advertising on our site helps.*?Policy)",
    re.DOTALL,
)

# Image placeholders
_IMAGE_RE = re.compile(
    r"Image\ncontent:\nThis\nimage\nis available.*?\)\n?",
    re.DOTALL,
)

# Reddit boilerplate
_REDDIT_BOILERPLATE = re.compile(
    r"(Go to \w+\nr/\w+\n•\n\w+)"
    r"|Reddit - The heart of the internet",
    re.DOTALL,
)

# Rendered timestamp lines
_RENDERED_RE = re.compile(r"Rendered:.*$", re.MULTILINE)

# Collapse excessive whitespace
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _find_content_boundary(text: str, headings: list[str]) -> tuple[int, int]:
    """Use extracted headings to find the start/end of actual medical content.

    Returns (start_idx, end_idx) within the text.  Non-medical trailing
    headings (Related Articles, Trending, etc.) are used as the end boundary.
    """
    non_content_headings = {
        "Related Articles", "Trending Topics", "Health Categories To Explore",
        "Other Popular Categories", "Top Posts",
    }

    start = 0
    end = len(text)

    # Find first medical heading as start
    for h in headings:
        if h in non_content_headings:
            continue
        idx = text.find(h)
        if idx != -1:
            start = idx
            break

    # Find first non-content heading as end
    for h in headings:
        if h in non_content_headings:
            idx = text.find(h)
            if idx != -1 and idx < end:
                end = idx

    return start, end


def clean_text(raw_text: str, headings: list[str] | None = None) -> str:
    """Clean a single article's raw_text field."""
    text = raw_text

    # Strip scraper debug output
    text = _FETCH_CACHE_RE.sub("", text)

    # Strip navigation
    for pat in _NAV_PATTERNS:
        text = pat.sub("", text)

    # Strip cookies / banners
    text = _COOKIE_RE.sub("", text)

    # Strip editorial boilerplate
    text = _EDITORIAL_RE.sub("", text)

    # Strip image placeholders
    text = _IMAGE_RE.sub("", text)

    # Strip Reddit boilerplate
    text = _REDDIT_BOILERPLATE.sub("", text)

    # Strip rendered timestamps
    text = _RENDERED_RE.sub("", text)

    # Use headings to narrow to medical content
    if headings:
        start, end = _find_content_boundary(text, headings)
        if end - start > MIN_CLEANED_CHARS:
            text = text[start:end]

    # Strip "Related Articles" and everything after (if headings didn't catch it)
    text = _RELATED_RE.sub("", text)

    # Collapse whitespace
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()

    return text


def load_all_raw_articles() -> list[dict]:
    """Load all JSON files from raw_sources/pcos/ subdirectories."""
    articles = []
    for subdir in sorted(RAW_DATA_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                data["_source_dir"] = subdir.name
                data["_file_path"] = str(json_file)
                articles.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"  [SKIP] Failed to load {json_file.name}: {e}")
    return articles


def deduplicate_by_url(articles: list[dict]) -> list[dict]:
    """Deduplicate articles that share the same URL (e.g. article/ vs articles/)."""
    seen_urls: dict[str, dict] = {}
    for art in articles:
        url = art.get("url", "").rstrip("/")
        if not url:
            continue
        # Keep the one with more raw_text
        if url not in seen_urls or len(art.get("raw_text", "")) > len(seen_urls[url].get("raw_text", "")):
            seen_urls[url] = art
    return list(seen_urls.values())


def run() -> list[dict]:
    """Main cleaning pipeline. Returns list of cleaned article dicts."""
    print("[Step 1] Loading raw articles...")
    articles = load_all_raw_articles()
    print(f"  Loaded {len(articles)} raw articles")

    print("[Step 1] Deduplicating by URL...")
    articles = deduplicate_by_url(articles)
    print(f"  {len(articles)} unique articles after dedup")

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = []
    skipped = 0

    for art in articles:
        headings = art.get("extracted_sections", {}).get("headings", [])
        cleaned_text = clean_text(art.get("raw_text", ""), headings)

        if len(cleaned_text) < MIN_CLEANED_CHARS:
            skipped += 1
            continue

        out = {
            "source_id": art.get("source_id"),
            "url": art.get("url"),
            "title": art.get("title"),
            "domain": art.get("domain"),
            "source_type": art.get("source_type", art.get("_source_dir")),
            "topic": art.get("topic"),
            "confidence_level": art.get("confidence_level", "MEDIUM"),
            "cleaned_text": cleaned_text,
            "headings": headings,
            "char_count": len(cleaned_text),
        }
        cleaned.append(out)

        # Write individual cleaned file
        out_path = CLEANED_DIR / f"{art['source_id']}_{art.get('domain', 'unknown')}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Step 1] Done: {len(cleaned)} cleaned, {skipped} skipped (< {MIN_CLEANED_CHARS} chars)")

    # Write summary manifest
    manifest = {
        "total_raw": len(articles) + skipped,
        "total_cleaned": len(cleaned),
        "skipped": skipped,
        "by_source_type": {},
    }
    for art in cleaned:
        st = art["source_type"]
        manifest["by_source_type"][st] = manifest["by_source_type"].get(st, 0) + 1

    (CLEANED_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  Manifest: {manifest['by_source_type']}")

    return cleaned


if __name__ == "__main__":
    run()
