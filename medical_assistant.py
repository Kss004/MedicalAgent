import os
import json
import ast
import asyncio
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import YouTubeSearchTool
from openai import AsyncOpenAI

# Load environment variables (API keys)
load_dotenv()

# Ensure API keys are set
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable is not set.")
if not os.getenv("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY environment variable is not set.")

# --- 1. Define Trusted Sources ---
# This is the "Whitelist" that ensures we only search verified domains
ALLOWED_DOMAINS = [
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nhs.uk",
    "mayoclinic.org",
    "clevelandclinic.org",
    "endocrine.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "monash.edu",
    "eshre.eu"
]

HIGH_CONFIDENCE_DOMAINS = [
    "who.int", "cdc.gov", "nih.gov", "nhs.uk", 
    "mayoclinic.org", "clevelandclinic.org", "endocrine.org", "eshre.eu"
]

MEDIUM_CONFIDENCE_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "monash.edu"
]

def get_domain_from_url(url: str) -> str:
    """Extracts the base domain from a URL for strict allowlist matching."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def get_confidence_score(url: str) -> tuple:
    """Returns (label, base_score) for a domain."""
    domain = get_domain_from_url(url)
    if any(domain == d or domain.endswith("." + d) for d in HIGH_CONFIDENCE_DOMAINS):
        return "HIGH", 90
    if any(domain == d or domain.endswith("." + d) for d in MEDIUM_CONFIDENCE_DOMAINS):
        return "MEDIUM", 70
    return "LOW", 0

# --- 2. Initialize Search Tool ---
# We configure Tavily to specifically search ONLY the included domains.
# We fetch max 5 results to give the LLM enough context without overwhelming it.
tavily_search = TavilySearch(
    max_results=20
)

# --- 2b. Initialize Social Search Tools ---
youtube_search = YouTubeSearchTool()

SOCIAL_DOMAINS = ["youtube.com", "instagram.com"]
shorts_tavily = TavilySearch(
    max_results=5
)

# --- Local Medical Dictionaries for Prescription Pipeline ---
medical_terms_db = {
    "tsh": "Thyroid Stimulating Hormone, a hormone used to evaluate thyroid function.",
    "irregular menstrual cycle": "A menstrual cycle that does not occur at regular intervals.",
    "ultrasound": "A medical imaging technique used to examine internal organs.",
    "u/s": "Ultrasound, a medical imaging technique used to examine internal organs.",
    "pcos": "Polycystic Ovary Syndrome, a hormonal disorder.",
    "lh": "Luteinizing Hormone, a hormone associated with reproduction.",
    "fsh": "Follicle-Stimulating Hormone, a hormone associated with reproduction."
}

medicine_db = {
    "mif": "appears to be a medication name detected in the prescription. The exact medication and its purpose should be verified with a pharmacist or healthcare provider.",
    "metformin": "appears to be a medication name detected in the prescription. The exact medication and its purpose should be verified with a pharmacist or healthcare provider.",
    "letrozole": "appears to be a medication name detected in the prescription. The exact medication and its purpose should be verified with a pharmacist or healthcare provider.",
    "clomid": "appears to be a medication name detected in the prescription. The exact medication and its purpose should be verified with a pharmacist or healthcare provider.",
    "clomiphene": "appears to be a medication name detected in the prescription. The exact medication and its purpose should be verified with a pharmacist or healthcare provider."
}

# --- Model Fallback Layer ---
_active_model = "none"

class FallbackLLM:
    """Wrapper that tries PCOS fine-tuned model first, then OpenAI, then local Mistral."""

    def __init__(self):
        global _active_model
        self._pcos = None
        self._openai = None
        self._mistral = None

        # Initialize fine-tuned PCOS model (highest priority for PCOS queries)
        try:
            from langchain_ollama import ChatOllama
            self._pcos = ChatOllama(model="pcos-gemma-2b", temperature=0)
            print("[Model] Fine-tuned PCOS Gemma-2B available")
        except Exception as e:
            print(f"[Model] PCOS model init failed (not installed?): {e}")

        # Initialize OpenAI (gpt-4o-mini for speed/cost similar to flash)
        if os.getenv("OPENAI_API_KEY"):
            try:
                self._openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                print("[Model] OpenAI GPT-4o-mini available")
            except Exception as e:
                print(f"[Model] OpenAI init failed: {e}")

        # Initialize local Mistral
        try:
            from langchain_ollama import ChatOllama
            self._mistral = ChatOllama(model="mistral:7b-instruct-q4_K_M", temperature=0)
            print("[Model] Local Mistral 7B Q4 available")
        except Exception as e:
            print(f"[Model] Mistral init failed: {e}")

        if self._pcos:
            _active_model = "pcos-gemma-2b"
        elif self._openai:
            _active_model = "gpt-4o-mini"
        elif self._mistral:
            _active_model = "mistral-local"
        else:
            _active_model = "none"

    def invoke(self, prompt, **kwargs):
        global _active_model
        # Try fine-tuned PCOS model first
        if self._pcos:
            try:
                result = self._pcos.invoke(prompt, **kwargs)
                _active_model = "pcos-gemma-2b"
                return result
            except Exception as e:
                print(f"[Model] PCOS model call failed ({type(e).__name__}), falling back to OpenAI...")

        # Try OpenAI
        if self._openai:
            try:
                result = self._openai.invoke(prompt, **kwargs)
                _active_model = "gpt-4o-mini"
                return result
            except Exception as e:
                print(f"[Model] OpenAI call failed ({type(e).__name__}), falling back to Mistral...")

        # Fallback to local Mistral
        if self._mistral:
            try:
                result = self._mistral.invoke(prompt, **kwargs)
                _active_model = "mistral-local"
                return result
            except Exception as e:
                print(f"[Model] Mistral call also failed: {e}")
                raise

        raise RuntimeError("No LLM available. Check OpenAI API key or Ollama installation.")

    async def ainvoke(self, prompt, **kwargs):
        global _active_model
        # Try fine-tuned PCOS model first
        if self._pcos:
            try:
                result = await self._pcos.ainvoke(prompt, **kwargs)
                _active_model = "pcos-gemma-2b"
                return result
            except Exception as e:
                print(f"[Model] PCOS model async call failed ({type(e).__name__}), falling back to OpenAI...")

        # Try OpenAI
        if self._openai:
            try:
                result = await self._openai.ainvoke(prompt, **kwargs)
                _active_model = "gpt-4o-mini"
                return result
            except Exception as e:
                print(f"[Model] OpenAI async call failed ({type(e).__name__}), falling back to Mistral...")

        # Fallback to local Mistral
        if self._mistral:
            try:
                result = await self._mistral.ainvoke(prompt, **kwargs)
                _active_model = "mistral-local"
                return result
            except Exception as e:
                print(f"[Model] Mistral async call also failed: {e}")
                raise

        raise RuntimeError("No LLM available. Check OpenAI API key or Ollama installation.")

llm = FallbackLLM()

async def get_youtube_keywords(query: str) -> str:
    """Extracts core medical keywords from a verbose user query for better YouTube search results."""
    prompt = f"""
    Extract the 2 to 3 most important core medical keywords or conditions from this user query.
    Return ONLY the keywords separated by spaces, nothing else.
    User Query: "{query}"
    """
    try:
        keywords = (await llm.ainvoke(prompt)).content.strip().replace('"', '').replace("'", "")
        return keywords if keywords else query
    except Exception:
        return query

async def contextualize_query(query: str, history: list) -> str:
    """Uses the chat history to rewrite the user's latest query into a standalone search term if needed."""
    if not history:
        return query

    chat_history_str = ""
    # Only use the last 4 messages for context to keep search focused and cheap
    for msg in history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        # Truncate very long assistant replies
        text = msg.get("content", "")
        if len(text) > 500:
            text = text[:500] + "..."
        chat_history_str += f"{role}: {text}\n"

    prompt = f"""
    Given the following chat history and a follow-up user query,
    rephrase the follow-up query to be a standalone search query that
    captures all relevant medical context from the history.
    If the follow-up query is already standalone, just return it exactly as is.
    Do NOT answer the query, only return the rephrased standalone query.

    Chat History:
    {chat_history_str}

    Follow-up query: {query}

    Standalone query:"""

    try:
        standalone_query = (await llm.ainvoke(prompt)).content.strip()
        # Clean up quotes if hallucinated
        if standalone_query.startswith('"') and standalone_query.endswith('"'):
            standalone_query = standalone_query[1:-1]
        return standalone_query
    except Exception:
        return query

async def verify_source_content(query: str, url: str, content: str) -> dict:
    """
    Uses the LLM to verify if a single piece of retrieved content is actually relevant
    and reliable for the specific conditions mentioned in the user's query.
    Returns a dictionary with 'is_reliable' (bool) and 'reliability_score' (int).
    """
    verification_prompt = f"""
    You are an expert medical content verifier. Your job is to determine if the provided text
    contains reliable, factual information that specifically addresses the conditions or
    symptoms mentioned in the user's query.

    User Query: "{query}"
    Source URL: {url}
    Content to verify:
    {content}

    Evaluate the content based on factual density, clinical backing, and relevance.
    Note that video content (YouTube, Instagram) may have brief or incomplete descriptions;
    evaluate them fairly based on topical relevance rather than penalizing for brevity.

    Respond ONLY with a valid JSON object in this exact format:
    {{"is_reliable": true/false, "reliability_score": <number 1-100>}}

    Set is_reliable to true IF the content is topically relevant to the query and medically sound.
    You must be lenient towards video/social media transcripts as they lack deep clinical backing.
    """
    try:
        response_text = (await llm.ainvoke(verification_prompt)).content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]

        result = json.loads(response_text)
        return {
            "is_reliable": result.get("is_reliable", False),
            "reliability_score": result.get("reliability_score", 0)
        }
    except Exception as e:
        # Default to false if verification fails or JSON parsing fails
        return {"is_reliable": False, "reliability_score": 0}

def _normalize_tavily_results(raw):
    """Normalize Tavily results into a list of dicts with 'url' and 'content' keys."""
    if isinstance(raw, dict):
        items = raw.get('results', [])
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"url": item, "content": ""})
    return normalized

MAX_SOURCES = 5
MIN_SOURCES = 2

def _get_source_label(domain: str) -> str:
    """Return a human-friendly label for a domain."""
    labels = {
        "who.int": "World Health Organization",
        "cdc.gov": "Centers for Disease Control",
        "nih.gov": "National Institutes of Health",
        "nhs.uk": "NHS (UK)",
        "mayoclinic.org": "Mayo Clinic",
        "clevelandclinic.org": "Cleveland Clinic",
        "endocrine.org": "Endocrine Society",
        "eshre.eu": "ESHRE",
        "icmr.gov.in": "ICMR India",
        "pubmed.ncbi.nlm.nih.gov": "PubMed Central",
        "ncbi.nlm.nih.gov": "NCBI / PubMed",
        "monash.edu": "Monash University",
        "youtube.com": "YouTube",
        "youtu.be": "YouTube",
    }
    for key, label in labels.items():
        if domain == key or domain.endswith("." + key):
            return label
    return domain.title()


def _select_diverse_sources(validated: list, max_total: int = MAX_SOURCES) -> list:
    """
    Select up to max_total sources, ensuring a 'mix and match' of 
    high-confidence medical sources and community experiences.

    Target Mix: 3-4 Medical sources, 1-2 Community/Anecdotal sources.
    """
    all_sources = list(validated)
    if not all_sources:
        return []

    # Separate into categories
    medical_sources = [s for s in all_sources if s.get("content_type") != "ANECDOTAL"]
    community_sources = [s for s in all_sources if s.get("content_type") == "ANECDOTAL"]

    # Sort each by confidence
    medical_sources.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
    community_sources.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)

    selected = []
    seen_domains = set()

    # 1. Fill medical sources (First pass: unique domains)
    for s in medical_sources:
        if len(selected) >= max_total - 1: # Leave at least one slot for community
            break
        domain = s.get("source_domain", "")
        if domain not in seen_domains:
            seen_domains.add(domain)
            selected.append(s)

    # 2. Add up to 2 community sources if available
    for s in community_sources:
        if len(selected) >= max_total:
            break
        if len([x for x in selected if x.get("content_type") == "ANECDOTAL"]) >= 2:
            break
        selected.append(s)

    # 3. Fill remaining slots with best remaining medical sources
    for s in medical_sources:
        if len(selected) >= max_total:
            break
        if s not in selected:
            selected.append(s)

    # Ensure at least 1 video if possible
    has_video = any(s.get("content_type") == "VIDEO" for s in selected)
    if not has_video:
        videos = [s for s in medical_sources if s.get("content_type") == "VIDEO" and s not in selected]
        if videos:
            idx_to_replace = -1
            # Replace a medical source if we have many, otherwise just append if slot open
            if len(selected) >= max_total:
                # Find the lowest score medical source to replace
                for i in range(len(selected)-1, -1, -1):
                    if selected[i].get("content_type") != "ANECDOTAL":
                        selected[i] = videos[0]
                        break
            else:
                selected.append(videos[0])

    print(f"  Mixed Sources: {sum(1 for s in selected if s.get('content_type') != 'ANECDOTAL')} medical, "
          f"{sum(1 for s in selected if s.get('content_type') == 'ANECDOTAL')} community")
    return selected


def retrieve_community_reports(query: str, limit: int = 3) -> list:
    """Retrieves anecdotal community experiences from PostgreSQL."""
    import sys
    import os
    pocs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocs", "womens_health_pcos")
    if pocs_path not in sys.path:
        sys.path.insert(0, pocs_path)
    
    from pocs.womens_health_pcos.database.session import SessionLocal
    from pocs.womens_health_pcos.database.models import Source, SourcePage, ChunkMetadata
    from sqlalchemy import or_

    db = SessionLocal()
    try:
        # Filter out common chat words to find the actual condition
        stop_words = {"experiences", "reddit", "community", "advice", "story", "stories", "what", "how", "are", "with", "for", "the", "some", "someone", "anyone", "else"}
        keywords = [kw.strip("?.,!") for kw in query.lower().split() if len(kw) > 3 and kw not in stop_words]
        
        # Base query joining ChunkMetadata -> SourcePage -> Source
        base_query = db.query(ChunkMetadata, SourcePage, Source).join(
            SourcePage, ChunkMetadata.source_id == SourcePage.page_id
        ).join(
            Source, SourcePage.source_id == Source.source_id
        ).filter(
            Source.source_type == 'community',
            SourcePage.is_active == 1
        )

        if not keywords:
            # If no specific condition, just return recent community posts
            results = base_query.limit(limit).all()
        else:
            conditions = []
            for kw in keywords:
                conditions.append(ChunkMetadata.chunk_text.ilike(f"%{kw}%"))
                conditions.append(Source.source_name.ilike(f"%{kw}%")) # Using source_name instead of title
                
            results = base_query.filter(or_(*conditions)).limit(limit * 2).all() # Fetch extra in case of duplicates
        
        community_sources = []
        seen_urls = set()
        for chunk, page, source in results:
            if page.page_url in seen_urls:
                continue
            seen_urls.add(page.page_url)
            
            content = chunk.chunk_text.strip() if chunk.chunk_text else "Community Experience"
            community_sources.append({
                "title": source.source_name or "Community Experience (Reddit)",
                "url": page.page_url,
                "source_domain": source.base_url.replace("https://", "").replace("http://", "").split("/")[0],
                "content_type": "ANECDOTAL",
                "content": content[:800] + "...", # truncate
                "confidence_score": 30, # Low confidence for reddit
                "confidence_level": "LOW",
                "source_label": "Community Experience (Reddit/Quora)"
            })
            if len(community_sources) >= limit:
                break
        return community_sources
    except Exception as e:
        print(f"Error retrieving community reports: {e}")
        return []
    finally:
        db.close()

def search_trusted_sources(query: str) -> dict:
    """
    Three-layer verified medical retrieval pipeline.

    Priority:
    1. Local knowledge base (fastest, pre-scraped trusted content)
    2. Tavily web search fallback (if local KB insufficient)
    3. Auto-cache Tavily results into local KB for future queries

    All layers enforce deterministic domain filtering and confidence scoring.
    """
    import sys
    import os
    pocs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocs", "womens_health_pcos")
    if pocs_path not in sys.path:
        sys.path.insert(0, pocs_path)

    try:
        from retrieval.domain_filter import filter_by_domain
        from retrieval.content_classifier import classify_results
        from retrieval.confidence_scoring import score_results
        from retrieval.constraints_validator import validate_batch
        from storage.local_kb import search as kb_search, store_batch

        print(f"\n[Pipeline] Query: {query}")
        print(f"[Pipeline] Model: {_active_model}")

        # === Layer 1: Local Knowledge Base (Chunk-based search) ===
        print("[Layer 1: Knowledge Base Search]")
        from pocs.womens_health_pcos.database.session import SessionLocal
        from pocs.womens_health_pcos.database.models import Source, SourcePage, ChunkMetadata
        from sqlalchemy import or_

        db = SessionLocal()
        local_results = []
        try:
            # Filter out stop words for search
            stop_words = {"what", "is", "for", "the", "a", "an", "and", "or", "to", "of", "in"}
            search_keywords = [kw.strip("?.,!") for kw in query.lower().split() if len(kw) > 3 and kw not in stop_words]
            
            if search_keywords:
                query_filter = []
                for kw in search_keywords:
                    query_filter.append(ChunkMetadata.chunk_text.ilike(f"%{kw}%"))
                
                # Search chunks in the new 3-layer structure
                # We join ChunkMetadata -> SourcePage -> Source
                results = db.query(ChunkMetadata, SourcePage, Source).join(
                    SourcePage, ChunkMetadata.source_id == SourcePage.page_id
                ).join(
                    Source, SourcePage.source_id == Source.source_id
                ).filter(
                    SourcePage.is_active == 1,
                    or_(*query_filter)
                ).order_by(Source.trust_level.desc()).limit(15).all()
                
                seen_urls = set()
                for chunk, page, source in results:
                    if page.page_url in seen_urls:
                        continue
                    seen_urls.add(page.page_url)
                    
                    source_type = source.source_type.upper()
                    # Normalize community/reddit sources to ANECDOTAL for the mixer
                    content_type = "ANECDOTAL" if source_type == 'COMMUNITY' else source_type
                    
                    local_results.append({
                        "title": source.source_name or "Medical Resource",
                        "url": page.page_url,
                        "source_domain": source.base_url.replace("https://", "").replace("http://", "").split("/")[0],
                        "content_type": content_type,
                        "content": chunk.chunk_text,
                        "confidence_score": 90 if source.trust_level == "HIGH" else 70,
                        "source_label": source.source_name
                    })
        except Exception as e:
            print(f"Error searching local DB: {e}")
        finally:
            db.close()

        print(f"  Found {len(local_results)} local source chunks.")

        # === Layer 2: Tavily Fallback ===
        print("[Layer 2: Tavily Fallback]")
        try:
            from retrieval.tavily_search import MedicalSearchEngine
            engine = MedicalSearchEngine(max_results=20)
            raw_results = engine.search(query)
            print(f"  Tavily returned {len(raw_results)} raw results.")
        except Exception as e:
            print(f"  Tavily failed: {e}")
            raw_results = []

        if not raw_results and not local_results:
            return {"context": "No verified medical sources found for this query.", "source_urls": []}

        # Filter Tavily results through the deterministic pipeline
        tavily_validated = []
        if raw_results:
            filtered, _ = filter_by_domain(raw_results)
            print(f"  {len(filtered)} passed domain filter.")

            if filtered:
                classified = classify_results(filtered)
                scored = score_results(classified)
                tavily_validated, _ = validate_batch(scored)
                print(f"  {len(tavily_validated)} passed constraints.")

                # Auto-cache into local KB for future queries
                if tavily_validated:
                    cache_records = []
                    for r in tavily_validated:
                        cache_records.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "source_domain": r.get("source_domain", ""),
                            "content_type": r.get("content_type", "ARTICLE"),
                            "content": r.get("content", ""),
                            "confidence_level": r.get("confidence_level", "MEDIUM"),
                            "confidence_score": r.get("confidence_score", 70),
                            "query_topics": query,
                            "source_label": _get_source_label(r.get("source_domain", "")),
                        })
                    cached = store_batch(cache_records)
                    print(f"  Cached {cached} new sources to local KB.")

        # Combine local + Tavily results (dedup by URL)
        all_results = list(local_results)
        seen_urls = {r.get("url", "") for r in all_results}
        for r in tavily_validated:
            if r.get("url", "") not in seen_urls:
                all_results.append(r)
                seen_urls.add(r.get("url", ""))

        # Always try to fetch some community sources to provide a "mixed" perspective
        print(f"  [Community] Fetching experiential content for mixed perspective...")
        community_results = retrieve_community_reports(query, limit=3)
        for cr in community_results:
            if cr.get("url", "") not in seen_urls:
                all_results.append(cr)
                seen_urls.add(cr.get("url", ""))

        if not all_results:
            return {"context": "No verified medical sources found for this query.", "source_urls": []}

        # Diversity selection (Mix and Match)
        diverse = _select_diverse_sources(all_results)
        layer = "local_kb+tavily+community_mixed"
        return _build_response(diverse, query, source_layer=layer)

    except Exception as e:
        print(f"Error during pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {"context": "Search failed. Please try again later.", "source_urls": []}


def _build_response(sources: list, query: str, source_layer: str = "unknown") -> dict:
    """Build the final context and source_urls from a list of validated sources."""
    context_parts = []
    source_urls = []

    print(f"[Building Response] {len(sources)} sources from {source_layer}")
    for doc in sources:
        url = doc.get("url", "")
        content = doc.get("content", "")
        title = doc.get("title", "") or url.split("/")[-1].replace("-", " ").title()
        content_type = doc.get("content_type", "ARTICLE")
        domain = doc.get("source_domain", get_domain_from_url(url))
        source_label = doc.get("source_label", "") or _get_source_label(domain)
        confidence = doc.get("confidence_score", 50)
        conf_level = doc.get("confidence_level", "MEDIUM")

        type_tag = "VIDEO" if content_type == "VIDEO" else "ARTICLE"
        context_parts.append(
            f"[Source {len(source_urls)+1}] ({type_tag}) {title}\n"
            f"Source: {source_label} | Confidence: {confidence}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )
        source_urls.append({
            "url": url,
            "score": confidence,
            "type": content_type.lower(),
            "title": title,
            "source_label": source_label,
            "confidence_level": conf_level,
        })

    if not context_parts:
        return {"context": "No verified medical sources found for this query.", "source_urls": []}

    context = "\n\n".join(context_parts)
    print(f"  → Returning {len(source_urls)} sources | Layer: {source_layer} | Model: {_active_model}")
    return {"context": context, "source_urls": source_urls}

# --- 3. The Strict System Prompt ---
system_prompt = """You are a trusted healthcare information assistant called Dory.
Your role is to provide clear, educational health information grounded only in verified medical sources.

Response Guidelines:
- Write in clear, accessible language that a general audience can understand.
- Structure your response with short paragraphs and headers when covering multiple topics.
- Summarize research findings in simple terms, avoiding unnecessary jargon.
- Always cite your sources inline using the [Source N] labels (e.g., "According to [Source 1], ...").
- When citing a video source, mention it is a video (e.g., "As explained in the video [Source 3], ...").

Source Rules:
- Use ONLY the sources provided in the context below.
- Each source has a type (ARTICLE, VIDEO, or ANECDOTAL), a confidence score, and a publication source.
- If community/anecdotal sources are provided, you MAY use them to answer questions about experiences, advice, or what people are saying.
- If the context does not contain sufficient information (such as asking a medical question with no medical sources provided, AND no community experiences provided), explicitly state: "I don't have enough verified information on this specific topic. Please consult a healthcare provider."

You must NEVER:
- Diagnose medical conditions
- Recommend specific medications or treatments
- Claim medical certainty
- Hallucinate facts or fabricate sources
- Use information not present in the provided context

If the user asks for community experiences/advice AND anecdotal sources are provided in the context, answer empathetically using ONLY the Community Experience sources provided. Clearly state that these are anecdotal experiences and not medical advice. Do NOT apologize or refuse to answer if only anecdotal sources are present.

All information is educational only. Always recommend consulting a healthcare professional for personal medical decisions.

Provided Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# LLM initialized above for use in verification

# --- 5. Main Interaction Function ---

def run_prescription_pipeline(extracted_json_str: str) -> dict:
    import json
    try:
        data = json.loads(extracted_json_str)
    except Exception as e:
        return {"context": "Failed to parse OCR data.", "source_urls": []}

    output_lines = []
    topics_for_search = []
    
    ocr_conf = data.get("ocr_confidence", "N/A")
    ext_conf = data.get("extraction_confidence", "N/A")

    def _extract_details(item):
        if isinstance(item, dict):
            # Try to grab common keys the model might hallucinate
            text = item.get("raw_text") or item.get("name") or item.get("condition") or item.get("instruction")
            if not text:
                text = " ".join(str(v) for k, v in item.items() if k not in ["confidence", "parsed_value"])
            conf = item.get("confidence", 1.0)
            return str(text), conf
        return str(item), 1.0

    def _format_entry(title, items, explanation_fallback, db_lookup=None):
        if not items:
            return
        output_lines.append(f"### {title}\n")
        for item in items:
            item_str, conf = _extract_details(item)
            item_lower = item_str.lower().strip()
            
            explanation = explanation_fallback
            if db_lookup:
                for key, val in db_lookup.items():
                    if key in item_lower:
                        if title == "Lab Values":
                            explanation = f"**{key.upper()}** (Thyroid Stimulating Hormone) is a hormone used to evaluate thyroid function." if 'tsh' in key else f"This test is often used to evaluate {key.upper()}: {val}"
                        else:
                            explanation = val
                        topics_for_search.append(key)
                        break
            
            # Formatting line with confidence
            conf_display = f"{float(conf):.2f}" if isinstance(conf, (float, int)) else "N/A"
            line = f"- **{item_str}**\n  *Explanation: {explanation}*\n  *Confidence: {conf_display}*"
            if isinstance(conf, (float, int)) and conf < 0.6:
                line += " ⚠ *Some text may be unclear due to handwriting.*"
            output_lines.append(line + "\n")

    _format_entry("Medicines", data.get("medicines", []), "A medication name was detected in the prescription.", medicine_db)
    _format_entry("Conditions", data.get("conditions", []), "This may refer to a medical condition or symptom.", medical_terms_db)
    _format_entry("Lab Values", data.get("lab_values", []), "This value is commonly associated with laboratory test results.", medical_terms_db)
    _format_entry("Doctor Instructions", data.get("doctor_instructions", []), "An instruction noted by the doctor.", medical_terms_db)

    # Append Extraction Confidence Summary
    output_lines.append("### Extraction Confidence Summary")
    output_lines.append(f"- OCR Confidence: {ocr_conf}")
    output_lines.append(f"- Entity Extraction Confidence: {ext_conf}\n")

    disclaimer = "\n> **Disclaimer:** This information is educational and does not replace professional medical advice. Always consult your healthcare provider for actual medical interpretation and treatment."
    
    output_lines.append(disclaimer)
    context = "\n".join(output_lines)
    
    # Fetch optional references using Tavily via search_trusted_sources
    source_urls = []
    if topics_for_search:
        search_query = " ".join(topics_for_search[:2])
        results = search_trusted_sources(search_query)
        raw_sources = results.get("source_urls", [])

        # Deduplicate domains and ensure max 5 sources
        unique_sources = {}
        for source in raw_sources:
            domain = get_domain_from_url(source.get("url", ""))
            if domain and domain not in unique_sources:
                unique_sources[domain] = source
        
        source_urls = list(unique_sources.values())[:5]

        # Ensure video if available
        has_video = any(s.get("type", "").lower() == "video" for s in source_urls)
        if not has_video:
            # We check raw sources for a video
            videos = [s for s in raw_sources if s.get("type", "").lower() == "video"]
            if videos:
                if len(source_urls) == 5:
                    source_urls[-1] = videos[0]
                else:
                    source_urls.append(videos[0])

    return {
        "context": context,
        "source_urls": source_urls
    }

async def ask_health_assistant(query: str, history: list = None) -> dict:
    """Main function to interact with the assistant. Returns dict with response and sources."""
    try:
        history = history or []
        
        if query.startswith("I uploaded my prescription"):
            # Extract JSON from the query
            import re
            match = re.search(r"Here is what was extracted:\n\n(.*?)\n\nPlease explain", query, re.DOTALL)
            if match:
                extracted_json_str = match.group(1).strip()
                # Run the targeted prescription pipeline
                result = run_prescription_pipeline(extracted_json_str)
                return {
                    "response": result["context"],
                    "sources": result["source_urls"]
                }
            else:
                return {
                    "response": "Error: Could not parse prescription data. Please ensure it was successfully extracted.",
                    "sources": []
                }

        # Step 1: Contextualize the query against history so the search engine works properly
        standalone_query = await contextualize_query(query, history)
        if standalone_query != query:
            print(f"\n[Contextualized Query for Search: '{standalone_query}']")

        # Step 2: Search trusted sources using the standalone query
        search_results = search_trusted_sources(standalone_query)
        context = search_results["context"]
        source_urls = search_results["source_urls"]

        # Formulate history for LangChain
        formatted_history = []
        if history:
            for msg in history[-6:]: # Keep the window sane
                if msg.get("role") == "user":
                    formatted_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    formatted_history.append(AIMessage(content=msg.get("content", "")))

        # Step 3: Format the prompt with context, query, and chat_history
        formatted_prompt = prompt.format_messages(
            context=context,
            query=query,
            chat_history=formatted_history
        )

        # Step 4: Get LLM response
        response = await llm.ainvoke(formatted_prompt)
        response_text = response.content

        return {
            "response": response_text,
            "sources": source_urls
        }

    except Exception as e:
        error_msg = f"System Error: Please ensure your API keys are set correctly. Error details: {e}"
        print(f"\n{error_msg}")
        return {
            "response": error_msg,
            "sources": []
        }

# --- 6. Prescription OCR Analysis ---
async def analyze_prescription(base64_images: list[str]) -> dict:
    """Analyze one or more prescription images using OpenAI Vision and extract structured information."""
    try:
        client = AsyncOpenAI()

        image_count = len(base64_images)
        if image_count == 1:
            prompt_text = "Please read and extract all information from this prescription image."
        else:
            prompt_text = (
                f"I am uploading {image_count} images. They may be multiple pages of the same prescription "
                "or separate prescriptions. Please read and extract all information from every image. "
                "If they appear to be pages of the same prescription, combine the information. "
                "If they are separate prescriptions, clearly label each one."
            )

        content = [{"type": "text", "text": prompt_text}]
        for img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

        max_tokens = min(1000 * image_count, 4000)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical prescription reader. Analyze the prescription image(s) and extract the information into a strict JSON format.\n"
                        "Return ONLY a JSON object with this exact structure:\n"
                        "{\n"
                        '  "ocr_confidence": 0.0 to 1.0,\n'
                        '  "extraction_confidence": 0.0 to 1.0,\n'
                        '  "medicines": [{ "name": "...", "confidence": 0.0 to 1.0 }],\n'
                        '  "conditions": [{ "name": "...", "confidence": 0.0 to 1.0 }],\n'
                        '  "lab_values": [{ "raw_text": "e.g., TSH > 21", "parsed_value": 21.0, "confidence": 0.0 to 1.0 }],\n'
                        '  "doctor_instructions": [{ "instruction": "...", "confidence": 0.0 to 1.0 }]\n'
                        "}\n\n"
                        "CRITICAL: When extracting lab values, you MUST preserve the original text with operators (e.g., '>', '<', '=') in 'raw_text'. The 'parsed_value' should only contain the numeric format if safe. Do not modify numeric values in raw_text.\n"
                        "If multiple images are provided, they may be pages of the same prescription or separate prescriptions. "
                        "Handle both cases appropriately.\n"
                        "Do NOT provide medical advice or interpretations beyond what is written."
                    )

                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=max_tokens,
            temperature=0
        )
        
        extracted_text = response.choices[0].message.content
        return {"success": True, "extracted_text": extracted_text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- 7. PCOS Pattern Evaluation ---
def evaluate_pcos(user_data: dict) -> dict:
    """
    Evaluates a user questionnaire profile (age, weight, cycles, symptoms), 
    determines patterns, provides risk awareness scores, 
    and fetches structured educational content mapped to those patterns.
    """
    from pocs.womens_health_pcos.assessment.content_mapper import evaluate_pcos_patterns
    return evaluate_pcos_patterns(user_data)


# --- 8. Interactive Testing Loop ---
if __name__ == "__main__":
    async def main():
        print("\n" + "*" * 60)
        print("Healthcare Information Assistant (Prototype)")
        print("Type 'exit' or 'quit' to stop.")
        print("Remember to set OPENAI_API_KEY and TAVILY_API_KEY in your environment or a .env file.")
        print("*" * 60 + "\n")

        while True:
            user_input = input("Ask a health-related question: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            await ask_health_assistant(user_input)

    asyncio.run(main())
