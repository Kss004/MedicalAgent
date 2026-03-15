"""
Standalone Scraper using Notte.cc for high-quality extraction.
Falls back to BeautifulSoup for reliability.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Add root project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import Source, SourcePage
from pocs.womens_health_pcos.database.registry_db import save_page_content
from pocs.womens_health_pcos.processing.text_chunker import process_source_chunks

BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_sources")

def get_notte_content(url: str):
    """
    Experimental Notte.cc extraction.
    Note: Requires NOTTE_API_KEY in environment.
    """
    api_key = os.getenv("NOTTE_API_KEY")
    if not api_key:
        return None
    
    # Placeholder for actual Notte.cc SDK/API call
    # Assuming Notte provides a simple 'scrape' endpoint that returns clean markdown/text
    # and structured metadata.
    try:
        # This is a hypothetical implementation based on common AI scraping patterns
        # Replace with actual SDK once confirmed
        headers = {"Authorization": f"Bearer {api_key}"}
        # In a real scenario, we'd use the Notte SDK: 
        # from notte import Notte; client = Notte(api_key); data = client.scrape(url)
        response = requests.post("https://api.notte.cc/v1/scrape", json={"url": url}, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Notte] API failed for {url}: {e}")
    return None

def bs4_fallback(url: str):
    """Clean text extraction using BeautifulSoup."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 MedicalAgent/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        return {
            "title": soup.title.string.strip() if soup.title else "Untitled Article",
            "raw_text": text,
            "extracted_sections": {"headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])]}
        }
    except Exception as e:
        print(f"[Fallback] BeautifulSoup failed for {url}: {e}")
        return None

def scrape_and_store(db, page: SourcePage):
    """Scrape a single URL using Notte or Fallback, then store as JSON and update DB."""
    print(f"Scraping: {page.page_url}")
    
    # Get associated Source for metadata
    source = db.query(Source).filter(Source.source_id == page.source_id).first()
    if not source:
        print(f"  -> Error: No source root found for page {page.page_id}")
        return False

    # 1. Try Notte
    data = get_notte_content(page.page_url)
    method = "notte"
    
    # 2. Try Fallback
    if not data:
        print(f"  -> Falling back to BeautifulSoup...")
        data = bs4_fallback(page.page_url)
        method = "bs4_fallback"
        
    if not data or not data.get("raw_text"):
        print(f"  -> Error: No content extracted for {page.page_url}")
        return False
        
    # 3. Save to JSON (maintained for backward compatibility/raw storage)
    domain_folder = os.path.join(BASE_STORAGE_DIR, "pcos", source.source_type)
    os.makedirs(domain_folder, exist_ok=True)
    
    domain_label = source.base_url.replace("https://", "").replace("http://", "").split("/")[0]
    filename = f"{page.page_id}_{domain_label}.json"
    file_path = os.path.join("pcos", source.source_type, filename)
    abs_path = os.path.join(BASE_STORAGE_DIR, file_path)
    
    full_data = {
        "page_id": page.page_id,
        "url": page.page_url,
        "title": data.get("title", "Untitled"),
        "domain": domain_label,
        "scraped_at": datetime.utcnow().isoformat(),
        "source_type": source.source_type,
        "raw_text": data.get("raw_text"),
        "extracted_sections": data.get("extracted_sections", {}),
        "metadata": {
            "scraper_method": method,
            "discovery_method": page.discovery_method
        }
    }
    
    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
        
    # 4. Update DB via the new Save Page Content logic (Handles Hashing & Versioning)
    save_page_content(
        db, 
        page.page_id, 
        data.get("raw_text"), 
        json_data=full_data["extracted_sections"]
    )
    
    # 5. Trigger Chunking - We pass the page object as 'source' for existing chunker compatibility
    # The chunker logic expects an object with page_id/source_id and text content access.
    process_source_chunks(db, page)
    
    return True

def run_notte_ingestion(limit=500):
    """Ingest up to `limit` source pages from the registry that haven't been scraped yet."""
    init_db()
    db = SessionLocal()
    
    # Get active source pages without a content hash (meaning they haven't been successfully scraped/versioned)
    pages = db.query(SourcePage).filter(
        SourcePage.is_active == 1,
        SourcePage.last_hash == None
    ).limit(limit).all()
    
    print(f"Found {len(pages)} source pages to scrape.")
    
    success_count = 0
    for page in pages:
        if scrape_and_store(db, page):
            success_count += 1
            
    print(f"Ingestion complete. Successfully scraped {success_count} sources.")
    db.close()

if __name__ == "__main__":
    # If called directly, scrape up to 500 sources
    run_notte_ingestion(500)
