import sys
import os
import json

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pocs.womens_health_pcos.database.session import SessionLocal
from pocs.womens_health_pcos.database.models import SourceRegistry

def run_stats():
    db = SessionLocal()
    try:
        sources = db.query(SourceRegistry).all()
        
        domain_counts = {}
        type_counts = {}
        longest_sources = []
        
        for s in sources:
            domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
            type_counts[s.source_type] = type_counts.get(s.source_type, 0) + 1
            
            if s.file_path:
                base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pocs', 'womens_health_pcos', 'data', 'raw_sources')
                abs_path = os.path.join(base_dir, s.file_path)
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            text_len = len(data.get('raw_text', ''))
                            longest_sources.append({
                                "title": s.title,
                                "length": text_len,
                                "url": s.url,
                                "method": data.get("metadata", {}).get("scraper_method", "unknown")
                            })
                    except:
                        pass
        
        print("📊 --- Knowledge Base Distribution by Domain ---")
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for d, c in sorted_domains:
            print(f"- {d}: {c} sources")
            
        print("\n📂 --- Sources by Content Type ---")
        for t, c in type_counts.items():
            print(f"- {t.title()}: {c}")
            
        print("\n📜 --- Top 5 Deepest Articles (Character Count) ---")
        longest_sources.sort(key=lambda x: x['length'], reverse=True)
        for i, l in enumerate(longest_sources[:5]):
            print(f"{i+1}. {l['title'][:60]}...")
            print(f"   Length: {l['length']:,} chars | Method: {l['method']}")
            print(f"   URL: {l['url']}")
            
    finally:
        db.close()

if __name__ == "__main__":
    run_stats()
