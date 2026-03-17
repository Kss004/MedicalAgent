from pocs.womens_health_pcos.assessment.questionnaire import UserQuestionnaire

# --- Pattern-to-Recommendation Mappings ---

MONITORING_MAP = {
    "hormonal_imbalance": [
        "Track menstrual cycle length and regularity for 3+ months",
        "Note skin/hair changes monthly",
        "Record energy levels daily",
    ],
    "insulin_resistance_pattern": [
        "Monitor fasting blood glucose periodically",
        "Track weight trends weekly",
        "Log dietary intake patterns",
    ],
    "reproductive_irregularity": [
        "Track cycle dates and flow for 3-6 months",
        "Note any mid-cycle symptoms",
    ],
    "metabolic_risk": [
        "Monitor blood pressure regularly",
        "Track waist circumference monthly",
        "Log physical activity",
    ],
}

LIFESTYLE_MAP = {
    "hormonal_imbalance": [
        "Consider anti-inflammatory dietary patterns (fruits, vegetables, omega-3s)",
        "Aim for 7-9 hours of quality sleep",
        "Practice stress management (yoga, meditation, breathing exercises)",
    ],
    "insulin_resistance_pattern": [
        "Focus on low-glycemic-index foods",
        "30 minutes moderate exercise 5 days/week",
        "Reduce refined carbohydrates and added sugars",
        "Consider Mediterranean or DASH dietary patterns",
    ],
    "reproductive_irregularity": [
        "Maintain consistent sleep schedule",
        "Moderate regular exercise",
        "Ensure adequate nutrition (iron, folate, vitamin D)",
    ],
    "metabolic_risk": [
        "Gradual increase in physical activity",
        "Focus on whole foods over processed foods",
        "Stay hydrated",
        "Consider working with a registered dietitian",
    ],
}

TESTS_MAP = {
    "hormonal_imbalance": [
        "Free & Total Testosterone",
        "DHEA-S",
        "17-OH Progesterone",
        "LH/FSH ratio",
    ],
    "insulin_resistance_pattern": [
        "Fasting Insulin",
        "Fasting Glucose",
        "HbA1c",
        "HOMA-IR calculation",
        "Lipid panel",
    ],
    "reproductive_irregularity": [
        "LH/FSH ratio",
        "AMH (Anti-Mullerian Hormone)",
        "Pelvic ultrasound",
        "Progesterone (day 21)",
    ],
    "metabolic_risk": [
        "Fasting glucose",
        "HbA1c",
        "Lipid panel",
        "Blood pressure check",
        "Thyroid panel (TSH, T3, T4)",
    ],
}


def _merge_recommendations(mapping: dict, patterns: list[str]) -> list[str]:
    """Merge and deduplicate recommendations across detected patterns."""
    seen = set()
    result = []
    for pattern in patterns:
        for item in mapping.get(pattern, []):
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def get_monitoring_recommendations(patterns: list[str]) -> list[str]:
    return _merge_recommendations(MONITORING_MAP, patterns)


def get_lifestyle_suggestions(patterns: list[str]) -> list[str]:
    return _merge_recommendations(LIFESTYLE_MAP, patterns)


def get_recommended_tests(patterns: list[str]) -> list[str]:
    return _merge_recommendations(TESTS_MAP, patterns)


def detect_patterns(user_data: UserQuestionnaire) -> list[str]:
    """
    Rule-based pattern detection system.
    Identifies underlying physiological patterns based on reported symptoms.
    Does NOT diagnose. Identifies "patterns".
    """
    patterns = []
    symptoms = [s.lower() for s in user_data.symptoms]
    
    # 1. Hormonal Imbalance Pattern (Androgen excess + cycle issues)
    has_hyperandrogenism = any(s in symptoms for s in ["acne", "hirsutism", "hair_loss"])
    has_cycle_issue = user_data.cycle_regularity in ["irregular", "absent"]
    if has_hyperandrogenism and has_cycle_issue:
        patterns.append("hormonal_imbalance")

    # 2. Insulin Resistance Pattern (Weight, fatigue, family history)
    has_metabolic_symptoms = "weight_gain" in symptoms or user_data.energy_levels in ["low", "exhausted"]
    has_diabetes_risk = user_data.family_history_diabetes or (user_data.bmi and user_data.bmi >= 25.0)
    if has_metabolic_symptoms and has_diabetes_risk:
        patterns.append("insulin_resistance_pattern")

    # 3. Reproductive Irregularity Pattern (Cycles without distinct hyperandrogenism necessarily)
    if has_cycle_issue and not has_hyperandrogenism:
        patterns.append("reproductive_irregularity")
        
    # 4. General Metabolic Risk
    if user_data.bmi and user_data.bmi > 30.0:
        patterns.append("metabolic_risk")
        
    return patterns

def generate_educational_queries(patterns: list[str]) -> list[str]:
    """Maps detected patterns to standard search queries for the Retrieval Engine."""
    queries = []
    for p in patterns:
        if p == "hormonal_imbalance":
            queries.append("PCOS hormonal imbalance symptoms and lifestyle management")
        elif p == "insulin_resistance_pattern":
            queries.append("PCOS insulin resistance weight gain and dietary treatment")
        elif p == "reproductive_irregularity":
            queries.append("PCOS irregular cycles anovulation causes")
        elif p == "metabolic_risk":
            queries.append("PCOS metabolic syndrome risk factors and exercise")
            
    # Default fallback if no patterns detected
    if not queries:
        queries.append("PCOS general health guidelines and cycle tracking")
        
    return queries
