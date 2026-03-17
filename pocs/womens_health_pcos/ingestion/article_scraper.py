"""
Article Scraper for building the local medical knowledge base.

Uses trafilatura for clean text extraction from trusted medical websites.
Falls back to requests + BeautifulSoup for simpler pages.

Supports deep crawling via `scrape_article_deep()` and `scrape_batch_deep()`:
- `depth` controls how many link-hops to follow from each seed URL
- Only trusted (whitelisted) domains are ever crawled
- `max_pages` caps total pages across the entire crawl to prevent runaway requests
"""

import os
import sys

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.domain_filter import is_trusted, get_domain
from retrieval.confidence_scoring import score_source


def scrape_article(url: str) -> dict | None:
    """
    Scrape a single article URL and return a structured record.

    Returns None if scraping fails or domain is not trusted.
    """
    if not is_trusted(url):
        print(f"[Scraper] Rejected untrusted domain: {url}")
        return None

    domain = get_domain(url)
    content = _extract_content(url)

    if not content or len(content) < 50:
        print(f"[Scraper] No usable content from: {url}")
        return None

    title = _extract_title(url, content)
    scoring = score_source(url, "ARTICLE")

    return {
        "title": title,
        "url": url,
        "source_domain": domain,
        "content_type": "ARTICLE",
        "content": content,
        "confidence_level": scoring["confidence_level"],
        "confidence_score": scoring["confidence_score"],
        "source_label": domain,
    }


def _extract_content(url: str) -> str:
    """Extract clean text content from a URL."""
    # Try trafilatura first (best quality)
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text) > 50:
                return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[Scraper] Trafilatura failed for {url}: {e}")

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:10000]  # Cap at 10k chars
    except Exception as e:
        print(f"[Scraper] BeautifulSoup fallback failed for {url}: {e}")

    return ""


def _extract_title(url: str, content: str) -> str:
    """Extract a title from the content or URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()[:200]
    except Exception:
        pass

    # Fallback: first line of content
    first_line = content.split("\n")[0].strip()
    return first_line[:200] if first_line else url.split("/")[-1].replace("-", " ").title()


def scrape_batch(urls: list[str]) -> list[dict]:
    """Scrape multiple URLs and return list of valid records."""
    results = []
    for i, url in enumerate(urls):
        print(f"[Scraper] ({i+1}/{len(urls)}) {url[:70]}...")
        record = scrape_article(url)
        if record:
            results.append(record)
    print(f"[Scraper] Successfully scraped {len(results)}/{len(urls)} articles.")
    return results


def _extract_links(url: str) -> list[str]:
    """
    Extract all outbound links from a page that point to trusted domains.

    Strips URL fragments (#section) and deduplicates results.
    Returns an empty list on any fetch/parse failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        seen = set()
        links = []
        for a_tag in soup.find_all("a", href=True):
            absolute = urljoin(url, a_tag["href"]).split("#")[0]  # strip fragments
            if (
                absolute.startswith(("http://", "https://"))
                and absolute != url
                and absolute not in seen
                and is_trusted(absolute)
            ):
                seen.add(absolute)
                links.append(absolute)

        return links
    except Exception as e:
        print(f"[Scraper] Link extraction failed for {url}: {e}")
        return []


def scrape_article_deep(
    url: str,
    depth: int = 5,
    max_pages: int = 50,
    visited: set | None = None,
) -> list[dict]:
    """
    Recursively scrape a seed URL and the trusted links found within it.

    Args:
        url:       Seed URL to start from.
        depth:     How many link-hops to follow (0 = seed only, 5 = default).
        max_pages: Hard cap on total pages scraped across the whole crawl.
                   Prevents runaway requests when a trusted domain is large.
        visited:   Internal set used to track already-visited URLs across
                   recursive calls. Pass None (default) on the initial call.

    Returns:
        List of scraped article records (same schema as `scrape_article()`).
    """
    if visited is None:
        visited = set()

    # Hard cap — checked before every page, not just at depth==0
    if len(visited) >= max_pages:
        return []

    if url in visited:
        return []

    visited.add(url)

    print(f"[DeepScraper] depth={depth} pages_so_far={len(visited)} | {url[:80]}")

    results = []
    record = scrape_article(url)
    if record:
        results.append(record)

    if depth == 0:
        return results

    for link in _extract_links(url):
        if len(visited) >= max_pages:
            print(f"[DeepScraper] max_pages={max_pages} reached, stopping.")
            break
        results.extend(
            scrape_article_deep(link, depth=depth - 1, max_pages=max_pages, visited=visited)
        )

    return results


def scrape_batch_deep(
    urls: list[str],
    depth: int = 5,
    max_pages: int = 50,
) -> list[dict]:
    """
    Deep-crawl multiple seed URLs.

    The `visited` set is shared across all seeds so the same page is never
    scraped twice even when different seeds link to it.

    Args:
        urls:      List of seed URLs.
        depth:     Link-hop depth limit (default 5).
        max_pages: Total page cap across all seeds combined (default 50).

    Returns:
        Deduplicated list of scraped article records.
    """
    visited: set = set()
    results = []
    for i, url in enumerate(urls):
        if len(visited) >= max_pages:
            print(f"[DeepScraper] max_pages={max_pages} reached after {i} seeds.")
            break
        print(f"[DeepScraper] Seed ({i+1}/{len(urls)}): {url[:70]}")
        results.extend(
            scrape_article_deep(url, depth=depth, max_pages=max_pages, visited=visited)
        )

    # Deduplicate by URL in case two seeds share linked pages
    seen_urls: set = set()
    unique = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    print(
        f"[DeepScraper] Done. {len(unique)} unique pages from {len(urls)} seeds "
        f"(depth={depth}, max_pages={max_pages})."
    )
    return unique


if __name__ == "__main__":
    test_urls = [
        "https://www.mayoclinic.org/diseases-conditions/pcos/symptoms-causes/syc-20353439",
        "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome",
        "https://www.cdc.gov/diabetes/about/pcos-and-diabetes.html",
    ]

    # Shallow batch (original behaviour)
    print("=== Shallow batch ===")
    records = scrape_batch(test_urls)
    for r in records:
        print(f"  {r['title'][:60]} | {r['source_domain']} | {r['confidence_level']} | {len(r['content'])} chars")

    # Deep crawl from a single seed (depth=2, capped at 10 pages for the demo)
    print("\n=== Deep crawl (depth=2, max_pages=10) ===")
    deep_records = scrape_batch_deep(test_urls[:1], depth=2, max_pages=10)
    for r in deep_records:
        print(f"  {r['title'][:60]} | {r['source_domain']} | {len(r['content'])} chars")
