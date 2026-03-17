"""
Ingests Community Reports mapping anecdotal experiences from Reddit (r/PCOS, r/PCOSloseit).
Stores extracted sentiments and outcomes into `community_reports`.
These sources MUST be assigned a LOW confidence score (e.g., 30)
to ensure they are treated as anecdotal, not medical fact.

Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in .env.
Falls back to mock data if credentials are missing or API fails.
"""

import sys
import os
import json

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
load_dotenv()

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import CommunityReport

# Subreddits to scrape
TARGET_SUBREDDITS = ["PCOS", "PCOSloseit"]
SEARCH_QUERIES = [
    "fatigue inositol supplement",
    "metformin side effects",
    "hair loss spironolactone",
    "weight loss what worked",
    "irregular periods tips",
    "acne treatment experience",
]
MAX_POSTS_PER_QUERY = 5

# Fallback mock data when Reddit API is unavailable
MOCK_REDDIT_DATA = [
    {
        "title": "What actually helped your PCOS fatigue?",
        "source_domain": "reddit.com",
        "source_link": "https://reddit.com/r/PCOS/comments/mock1",
        "text_content": "I was sleeping 10 hours a day and still exhausted. I started taking Ovasitol (Inositol) and focusing on 30g protein at breakfast. Within 3 weeks my energy came back.",
        "symptom": "Fatigue",
        "treatment": "Ovasitol / Myo-inositol supplement",
        "medication": "None",
        "lifestyle_change": "High protein breakfast",
        "reported_outcome": "Improved energy levels",
        "sentiment": "Positive"
    },
    {
        "title": "Metformin side effects are brutal",
        "source_domain": "reddit.com",
        "source_link": "https://reddit.com/r/PCOSloseit/comments/mock2",
        "text_content": "Has anyone else had terrible stomach issues on 1000mg Metformin? It's been 2 weeks and I can barely leave the house. Is extended release better?",
        "symptom": "Insulin Resistance",
        "treatment": "Metformin",
        "medication": "Metformin 1000mg",
        "lifestyle_change": "None",
        "reported_outcome": "Severe GI side effects",
        "sentiment": "Negative"
    },
    {
        "title": "Hair loss finally stopping after 2 years",
        "source_domain": "reddit.com",
        "source_link": "https://reddit.com/r/womenshealth/comments/mock3",
        "text_content": "Just wanted to share a win. My hair was thinning so bad at the crown. Spironolactone 100mg plus rosemary oil massages finally stopped the shedding. It took 6 months to see a difference.",
        "symptom": "Hair loss / Alopecia",
        "treatment": "Spironolactone + Rosemary Oil",
        "medication": "Spironolactone 100mg",
        "lifestyle_change": "Rosemary oil massages",
        "reported_outcome": "Shedding stopped",
        "sentiment": "Positive"
    }
]


def _get_reddit_client():
    """Initialize PRAW Reddit client from env vars. Returns None if unavailable."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "pcos-health-bot/1.0")

    if not client_id or not client_secret:
        return None

    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Test connectivity
        reddit.read_only = True
        return reddit
    except Exception as e:
        print(f"[Community] PRAW init failed: {e}")
        return None


def _classify_sentiment(text: str) -> str:
    """Simple keyword-based sentiment classification."""
    text_lower = text.lower()
    positive_words = {"helped", "improved", "better", "worked", "love", "finally", "win", "success", "recommend"}
    negative_words = {"worse", "terrible", "horrible", "side effects", "didn't work", "awful", "brutal", "painful"}
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


def _scrape_reddit(reddit) -> list[dict]:
    """Scrape posts from target subreddits using PRAW."""
    posts = []
    seen_ids = set()

    for sub_name in TARGET_SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            for query in SEARCH_QUERIES:
                for submission in subreddit.search(query, sort="relevance", limit=MAX_POSTS_PER_QUERY):
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)

                    body = submission.selftext or ""
                    if len(body) < 50:
                        continue

                    posts.append({
                        "title": submission.title,
                        "source_domain": "reddit.com",
                        "source_link": f"https://reddit.com{submission.permalink}",
                        "text_content": body[:2000],
                        "symptom": query.split()[0].title(),
                        "treatment": None,
                        "medication": None,
                        "lifestyle_change": None,
                        "reported_outcome": None,
                        "sentiment": _classify_sentiment(body),
                    })
        except Exception as e:
            print(f"[Community] Error scraping r/{sub_name}: {e}")

    print(f"[Community] Scraped {len(posts)} posts from Reddit.")
    return posts


def run_ingestion():
    init_db()
    print("Starting Community Experience Ingestion...")
    db = SessionLocal()
    total_stored = 0

    try:
        reddit = _get_reddit_client()
        if reddit:
            print("[Community] Reddit API connected — scraping live data.")
            post_data = _scrape_reddit(reddit)
        else:
            print("[Community] Reddit API unavailable — using mock data.")
            post_data = MOCK_REDDIT_DATA

        for post in post_data:
            exists = db.query(CommunityReport).filter(
                CommunityReport.source_link == post["source_link"]
            ).first()
            if not exists:
                record = CommunityReport(
                    title=post["title"],
                    source_domain=post["source_domain"],
                    text_content=post["text_content"],
                    confidence_score=30,  # STRICT LOW CONFIDENCE TAG
                    source_link=post["source_link"],
                    symptom=post.get("symptom"),
                    treatment=post.get("treatment"),
                    medication=post.get("medication"),
                    lifestyle_change=post.get("lifestyle_change"),
                    reported_outcome=post.get("reported_outcome"),
                    sentiment=post.get("sentiment"),
                )
                db.add(record)
                total_stored += 1
                print(f"  [+] Stored community report: {post['title'][:50]}...")
            else:
                print(f"  [-] Skipped (Duplicate): {post['title'][:40]}...")

        db.commit()
        print(f"Ingestion complete. Stored {total_stored} community reports.")
    finally:
        db.close()


if __name__ == "__main__":
    run_ingestion()
