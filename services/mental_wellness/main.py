"""
Mental Wellness Service
Mental health support and analysis using MedGemma
"""
from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

app = FastAPI(
    title="Mental Wellness Service",
    description="Mental health support using MedGemma",
    version="1.0.0" 
)

class WellnessRequest(BaseModel):
    message: str
    mood_context: str = "neutral"
    user_id: str = "anonymous"

class WellnessResponse(BaseModel):
    supportive_response: str
    mood_assessment: str
    confidence: float
    resources: List[str]
    follow_up_suggestions: List[str]

@app.get("/")
async def root():
    return {
        "service": "Mental Wellness",
        "status": "healthy",
        "model": "MedGemma",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mental_wellness"}

@app.post("/comfort")
async def provide_support(request: WellnessRequest) -> WellnessResponse:
    """Provide mental health support and comfort"""
    # Mock response - will implement actual MedGemma model
    return WellnessResponse(
        supportive_response="I understand you're going through a difficult time. Your feelings are valid, and it's important to acknowledge them. Taking care of your mental health is just as important as your physical health.",
        mood_assessment="mildly_concerned",
        confidence=0.85,
        resources=[
            "National Suicide Prevention Lifeline: 988",
            "Crisis Text Line: Text HOME to 741741",
            "Mental Health America: mhanational.org"
        ],
        follow_up_suggestions=[
            "Consider speaking with a mental health professional", 
            "Practice mindfulness or meditation",
            "Maintain regular sleep and exercise routines",
            "Connect with supportive friends or family"
        ]
    )

@app.get("/model/info")
async def model_info():
    return {
        "model_name": "MedGemma",
        "model_type": "Language Model for Healthcare",
        "capabilities": ["mental_health_support", "mood_assessment", "resource_recommendations"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
