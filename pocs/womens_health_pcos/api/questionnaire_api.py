from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from medical_assistant import evaluate_pcos

router = APIRouter(prefix="/api/questionnaire", tags=["questionnaire"])

class QuestionnaireRequest(BaseModel):
    age: int
    cycle_regularity: str
    energy_levels: str
    family_history_pcos: Optional[bool] = False
    family_history_diabetes: Optional[bool] = False
    symptoms: List[str] = []
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    primary_health_concern: Optional[str] = None

@router.post("/evaluate")
async def evaluate_questionnaire(data: QuestionnaireRequest):
    try:
        # evaluate_pcos expects a dict
        result = evaluate_pcos(data.model_dump())
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
