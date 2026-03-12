import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.registry_db import create_source, get_source_by_url
from pocs.womens_health_pcos.schema.models import SourceRegistryRecord

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# Domains mapped by confidence
HIGH_DOMAINS = [
    "monash.edu", "eshre.eu", "endocrine.org", "acog.org", "nice.org.uk", "rcog.org.uk",
    "who.int", "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov", "nhs.uk", "icmr.nic.in",
    "mayoclinic.org", "clevelandclinic.org", "hopkinsmedicine.org", "stanfordmedicine.org",
    "massgeneral.org", "mountsinai.org", "uclahealth.org", "cedars-sinai.org"
]

MEDIUM_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "academic.oup.com", "jamanetwork.com", "med.stanford.edu"
]

LOW_DOMAINS = [
    "reddit.com", "quora.com", "youtube.com"
]

TOPICS = [
    "PCOS overview",
    "insulin resistance",
    "irregular menstrual cycles",
    "hyperandrogenism",
    "hirsutism",
    "acne",
    "infertility",
    "anovulation",
    "ovarian morphology",
    "AMH",
    "LH/FSH ratio",
    "thyroid and fertility",
    "hypothyroidism and menstrual health",
    "metabolic syndrome",
    "obesity and lean PCOS",
    "exercise and PCOS",
    "diet and PCOS",
    "adolescent PCOS",
    "fertility outcomes",
    "emotional / mental health aspects",
    "long-term cardiovascular risk",
    "diabetes risk",
    "pregnancy-related implications"
]

def classify_source_type(url: str, domain: str) -> str:
    url_lower = url.lower()
    domain_lower = domain.lower()
    if "youtube.com" in domain_lower or "youtu.be" in domain_lower:
        return "video"
    elif "guideline" in url_lower or domain_lower in ["nice.org.uk", "monash.edu", "acog.org", "eshre.eu"]:
        return "guideline"
    elif "pubmed" in domain_lower or "pmc" in url_lower:
        return "research_paper"
    elif domain_lower in ["who.int", "cdc.gov", "nih.gov", "nhs.uk", "icmr.nic.in"]:
        return "public_health"
    elif "reddit.com" in domain_lower or "quora.com" in domain_lower:
        return "community"
    else:
        return "article"

def get_domain_from_url(url: str) -> str:
    try:
        if "://" in url:
            return url.split("/")[2].replace("www.", "")
        return url.split("/")[0].replace("www.", "")
    except Exception:
        return "unknown"

def get_confidence_level(domain: str) -> str:
    domain_lower = domain.lower()
    for hd in HIGH_DOMAINS:
        if hd in domain_lower or domain_lower in hd:
            return "HIGH"
    for md in MEDIUM_DOMAINS:
        if md in domain_lower or domain_lower in md:
            return "MEDIUM"
    return "LOW"

def run_discovery():
    print("Starting Source Discovery...")
    init_db()
    db = SessionLocal()

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not TavilyClient or not api_key:
        print("Tavily not configured (missing key or package). Simulating discovery for POC.")
        # Fallback to a few static links to demonstrate functionality if Tavily is missing
        static_urls = [
            ("https://www.mayoclinic.org/diseases-conditions/pcos/symptoms-causes/syc-20353439", "PCOS overview", "mayoclinic.org", "HIGH"),
            ("https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome", "PCOS overview", "who.int", "HIGH"),
            ("https://www.monash.edu/medicine/mchri/pcos/guideline", "PCOS overview", "monash.edu", "HIGH")
        ]
        found_count = 0
        for item in static_urls:
            if not get_source_by_url(db, item[0]):
                record = SourceRegistryRecord(
                    url=item[0],
                    title="Simulated Discovery URL",
                    domain=item[2],
                    topic=item[1],
                    source_type=classify_source_type(item[0], item[2]),
                    confidence_level=item[3],
                    discovery_method="static_fallback",
                    is_active=True
                )
                create_source(db, record)
                found_count += 1
        print(f"Fallback complete. Added {found_count} simulated urls.")
        return

    tvly = TavilyClient(api_key=api_key) 
    found_count = 0
    duplicate_count = 0

    # Execute dynamic search through TOPICS
    for topic in TOPICS:
        print(f"Searching for topic: {topic}")
        # Combine high/med domains for search
        valid_domains = HIGH_DOMAINS + MEDIUM_DOMAINS
        
        try:
            res = tvly.search(query=topic + " PCOS women's health", max_results=5, include_domains=valid_domains)
            for r in res.get("results", []):
                domain = get_domain_from_url(r["url"])
                
                # Deduplicate
                if get_source_by_url(db, r["url"]):
                    duplicate_count += 1
                else:
                    record = SourceRegistryRecord(
                        url=r["url"],
                        title=r["title"],
                        domain=domain,
                        topic=topic,
                        source_type=classify_source_type(r["url"], domain),
                        confidence_level=get_confidence_level(domain),
                        discovery_method="tavily",
                        is_active=True
                    )
                    create_source(db, record)
                    found_count += 1
            
        except Exception as e:
            print(f"Search failed for {topic}: {e}")
            
        # Optional LOW confidence search limits
        try:
            res_low = tvly.search(query=topic + " PCOS personal experience", max_results=2, include_domains=LOW_DOMAINS)
            for r in res_low.get("results", []):
                domain = get_domain_from_url(r["url"])
                if get_source_by_url(db, r["url"]):
                    duplicate_count += 1
                else:
                    record = SourceRegistryRecord(
                        url=r["url"],
                        title=r["title"],
                        domain=domain,
                        topic=topic,
                        source_type=classify_source_type(r["url"], domain),
                        confidence_level="LOW",
                        discovery_method="tavily",
                        is_active=True
                    )
                    create_source(db, record)
                    found_count += 1
        except Exception as e:
            pass

        time.sleep(1) # rate limit prevention

    print(f"Discovery complete. Found {found_count} new sources. Skipped {duplicate_count} duplicates.")
    db.close()

if __name__ == "__main__":
    run_discovery()
