"""Step 3: Generate safety refusal examples for fine-tuning.

Creates 50-100 examples teaching the model to refuse diagnosis requests,
treatment prescriptions, and redirect to healthcare providers.
"""

import json
import random
from pathlib import Path

from config import QA_PAIRS_DIR, SYSTEM_PROMPT

# ── Templates ──────────────────────────────────────────────────────────────────

# Diagnosis-seeking questions
_DIAGNOSIS_QUESTIONS = [
    "Do I have PCOS?",
    "I have irregular periods and acne — is this PCOS?",
    "My ultrasound showed cysts on my ovaries. Does that mean I have PCOS?",
    "I'm gaining weight and losing hair. Can you diagnose me?",
    "Based on my symptoms, what condition do I have?",
    "I have high testosterone levels. Is it PCOS or something else?",
    "My periods are 45 days apart. Do I have polycystic ovaries?",
    "I have hirsutism and insulin resistance. Tell me what's wrong with me.",
    "Can you look at my lab results and tell me if I have PCOS?",
    "My BMI is 32 and I have acne. Is this hormonal? Do I have PCOS?",
    "I missed my period for 3 months. Is it PCOS?",
    "My sister has PCOS. I have similar symptoms — do I have it too?",
    "I have dark patches on my skin and irregular cycles. Diagnose me.",
    "Could my symptoms be PCOS or endometriosis?",
    "I took an online PCOS quiz and scored high. Do I have it?",
]

# Treatment/medication-seeking questions
_TREATMENT_QUESTIONS = [
    "What medication should I take for PCOS?",
    "Should I start taking metformin for my insulin resistance?",
    "What dosage of inositol should I take?",
    "Can you prescribe me birth control for PCOS?",
    "Should I take spironolactone for my PCOS acne?",
    "What's the best drug for PCOS hair loss?",
    "I want to stop taking metformin. Should I?",
    "Should I take Clomid to get pregnant with PCOS?",
    "Recommend a supplement stack for PCOS.",
    "What hormones should I take to fix my PCOS?",
    "Tell me the exact treatment plan for PCOS.",
    "Should I get ovarian drilling surgery?",
    "Is letrozole or Clomid better for PCOS fertility?",
    "Should I go on a keto diet to cure my PCOS?",
    "Can you write me a meal plan with exact portions?",
]

# Prognosis/certainty-seeking questions
_PROGNOSIS_QUESTIONS = [
    "Will my PCOS ever go away?",
    "Am I going to be infertile because of PCOS?",
    "Will I definitely get diabetes if I have PCOS?",
    "Is PCOS going to get worse as I age?",
    "Can you guarantee this diet will help my PCOS?",
    "Will I be able to have children?",
    "How long until my symptoms improve?",
    "Is my PCOS mild or severe?",
    "Will I need medication for the rest of my life?",
    "What are my chances of developing cancer from PCOS?",
]

# ── Refusal response templates ─────────────────────────────────────────────────

_DIAGNOSIS_REFUSALS = [
    "I understand you're concerned about your symptoms, but I'm not able to diagnose medical conditions. PCOS diagnosis requires a thorough evaluation by a healthcare provider, typically including blood tests, physical examination, and possibly an ultrasound. I'd strongly recommend scheduling an appointment with your doctor or an endocrinologist to discuss your symptoms. They can provide a proper assessment and determine the best path forward for you.",
    "I appreciate you sharing your symptoms with me, but diagnosing conditions is beyond what I'm able to do as a health information assistant. PCOS is typically diagnosed using the Rotterdam criteria, which requires a clinical evaluation by a qualified healthcare professional. Please consult your doctor — they can order the appropriate tests and give you a definitive answer.",
    "I can share general educational information about PCOS, but I'm not able to tell you whether you have it. A proper PCOS diagnosis involves clinical evaluation of symptoms, hormone testing, and sometimes imaging. I'd encourage you to bring your concerns to a healthcare provider who can properly assess your individual situation.",
]

_TREATMENT_REFUSALS = [
    "I'm not able to recommend specific medications or treatment plans, as these decisions should be made with a healthcare provider who knows your full medical history. Treatment for PCOS is highly individual and depends on your specific symptoms, health goals, and other medical conditions. Please discuss your options with your doctor or endocrinologist.",
    "Recommending specific medications or dosages is outside my scope as a health information assistant. What works for one person with PCOS may not be appropriate for another. I'd encourage you to discuss treatment options with your healthcare provider, who can tailor a plan to your specific needs and medical history.",
    "I understand you're looking for treatment guidance, but prescribing or recommending specific medications isn't something I'm able to do. PCOS management should be personalized by a healthcare professional. I can share general educational information about what approaches exist, but please consult your doctor for personal medical advice.",
]

_PROGNOSIS_REFUSALS = [
    "I understand your concern, but I'm not able to make predictions about individual health outcomes. PCOS affects everyone differently, and outcomes depend on many personal factors. A healthcare provider who knows your specific situation can give you a much more meaningful answer. I'd encourage you to discuss your concerns with your doctor.",
    "Making predictions about your specific health trajectory isn't something I'm able to do. What I can tell you is that PCOS is a manageable condition, and outcomes vary widely between individuals. For personalized guidance about what to expect, please consult your healthcare provider.",
    "I appreciate your question, but providing individual prognoses is beyond what I'm able to do. Every person's experience with PCOS is different. Your healthcare provider can review your specific situation and give you a much more accurate picture of what to expect. Please don't hesitate to raise these concerns with them.",
]


def _build_refusal_pair(question: str, category: str) -> dict:
    """Build a single refusal training example."""
    if category == "diagnosis":
        answer = random.choice(_DIAGNOSIS_REFUSALS)
    elif category == "treatment":
        answer = random.choice(_TREATMENT_REFUSALS)
    else:
        answer = random.choice(_PROGNOSIS_REFUSALS)

    return {
        "question": question,
        "answer": answer,
        "source_type": "safety_refusal",
        "confidence_level": "N/A",
        "category": category,
    }


def run() -> list[dict]:
    """Generate safety refusal training examples."""
    QA_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []

    # Generate diagnosis refusals
    for q in _DIAGNOSIS_QUESTIONS:
        pairs.append(_build_refusal_pair(q, "diagnosis"))

    # Generate treatment refusals
    for q in _TREATMENT_QUESTIONS:
        pairs.append(_build_refusal_pair(q, "treatment"))

    # Generate prognosis refusals
    for q in _PROGNOSIS_QUESTIONS:
        pairs.append(_build_refusal_pair(q, "prognosis"))

    # Shuffle to mix categories
    random.shuffle(pairs)

    out_path = QA_PAIRS_DIR / "_safety_refusals.json"
    out_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Step 3] Generated {len(pairs)} safety refusal examples")
    print(f"  Diagnosis: {len(_DIAGNOSIS_QUESTIONS)}")
    print(f"  Treatment: {len(_TREATMENT_QUESTIONS)}")
    print(f"  Prognosis: {len(_PROGNOSIS_QUESTIONS)}")

    return pairs


if __name__ == "__main__":
    run()
