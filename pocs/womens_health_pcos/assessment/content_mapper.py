import os
import re
import sys

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.assessment.questionnaire import UserQuestionnaire
from pocs.womens_health_pcos.assessment.scoring_model import calculate_risk_score
from pocs.womens_health_pcos.assessment.pattern_engine import (
    detect_patterns,
    generate_educational_queries,
    get_monitoring_recommendations,
    get_lifestyle_suggestions,
    get_recommended_tests,
)
from medical_assistant import search_trusted_sources


# --- Phase 2: Content Cleaning ---

_CITATION_RE = re.compile(
    r'\[(?:DOI|PubMed|Google Scholar|PMC|PMID)[^\]]*\]',
    re.IGNORECASE,
)
_NAV_ARTIFACT_RE = re.compile(
    r'^\s*\*\s+(?:Health Library|Diseases|Conditions|Patient Care|Research|'
    r'Education|For Medical Professionals|Log in|Menu|Search).*$',
    re.MULTILINE | re.IGNORECASE,
)
_EXCESS_WHITESPACE_RE = re.compile(r'\n{3,}')

MAX_CONTENT_LENGTH = 2000


def _clean_content(text: str) -> str:
    """Strip citation fragments, navigation artifacts, and excessive whitespace."""
    if not text:
        return text
    text = _CITATION_RE.sub('', text)
    text = _NAV_ARTIFACT_RE.sub('', text)
    text = _EXCESS_WHITESPACE_RE.sub('\n\n', text)
    text = text.strip()
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH].rsplit(' ', 1)[0] + '...'
    return text


# --- Main Evaluation ---

def evaluate_pcos_patterns(user_data_dict: dict) -> dict:
    """
    Main evaluation workflow:
    1. Validates input via Pydantic model
    2. Runs scoring algorithm (LOW/MODERATE/HIGH)
    3. Runs pattern engine
    4. Derives monitoring, lifestyle, and test recommendations from patterns
    5. Fetches verified educational content mapped to those patterns
    6. Returns safe, non-diagnostic insights
    """

    # 1. Validate
    try:
        user_data = UserQuestionnaire(**user_data_dict)
    except Exception as e:
        return {"error": "Invalid input format.", "details": str(e)}

    # 2. Score
    risk_assessment = calculate_risk_score(user_data)

    # 3. Detect Patterns
    patterns = detect_patterns(user_data)

    # 4. Pattern-based recommendations
    monitoring = get_monitoring_recommendations(patterns)
    lifestyle = get_lifestyle_suggestions(patterns)
    tests = get_recommended_tests(patterns)

    # 5. Map to Content (Search verified DB)
    queries = generate_educational_queries(patterns)
    educational_content = []

    for query in queries[:2]:
        print(f"[Content Mapper] Fetching verified knowledge for: {query}")
        search_results = search_trusted_sources(query)
        if search_results and "context" in search_results:
            educational_content.append({
                "theme": query,
                "verified_sources": search_results["source_urls"],
                "content": _clean_content(search_results["context"]),
            })

    # 6. Build display text
    breakdown = risk_assessment.get("score_breakdown", {})
    response_text = "## Pattern Assessment Summary\n"
    response_text += f"> {risk_assessment['recommendation']}\n\n"

    # Score breakdown
    response_text += "### Score Breakdown\n"
    response_text += f"| Component | Score |\n|---|---|\n"
    response_text += f"| Symptom Score | {breakdown.get('symptom_score', 0)} |\n"
    response_text += f"| Lifestyle Score | {breakdown.get('lifestyle_score', 0)} |\n"
    response_text += f"| Family History Score | {breakdown.get('family_history_score', 0)} |\n"
    response_text += f"| Age Factor | {breakdown.get('age_factor', 0)} |\n"
    response_text += f"| **Total** | **{risk_assessment['total_score']}** ({risk_assessment['category']}) |\n\n"

    # Detected patterns
    if patterns:
        formatted_patterns = [p.replace("_", " ").title() for p in patterns]
        response_text += f"### Detected Patterns\n"
        response_text += f"{', '.join(formatted_patterns)}\n\n"
        response_text += "These are symptom clusters that are common in endocrine tracking. "
        response_text += "Your responses suggest patterns that are sometimes associated with hormonal imbalance.\n\n"

    response_text += f"**Contributing Factors:** {', '.join(risk_assessment['contributing_factors'])}\n\n"

    # Monitoring
    if monitoring:
        response_text += "### Recommended Monitoring\n"
        for item in monitoring:
            response_text += f"- {item}\n"
        response_text += "\n"

    # Lifestyle
    if lifestyle:
        response_text += "### Suggested Lifestyle Changes\n"
        for item in lifestyle:
            response_text += f"- {item}\n"
        response_text += "\n"

    # Tests
    if tests:
        response_text += "### Recommended Tests\n"
        response_text += "*Ask your healthcare provider about these tests:*\n"
        for item in tests:
            response_text += f"- {item}\n"
        response_text += "\n"

    response_text += f"---\n*Disclaimer: {risk_assessment['safety_disclaimer']}*\n---\n"

    return {
        "assessment": risk_assessment,
        "patterns_detected": patterns,
        "recommended_monitoring": monitoring,
        "suggested_lifestyle_changes": lifestyle,
        "recommended_tests": tests,
        "educational_payload": educational_content,
        "display_text": response_text,
    }
