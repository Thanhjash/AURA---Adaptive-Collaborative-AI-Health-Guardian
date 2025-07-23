"""
ECG Interpreter Service
Specialized service for ECG analysis using OTSN model
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

app = FastAPI(
    title="ECG Interpreter Service", 
    description="ECG analysis using OTSN model",
    version="1.0.0"
)

class ECGAnalysisRequest(BaseModel):
    ecg_data: List[float]
    sampling_rate: int = 500
    patient_id: str = "anonymous"

class ECGAnalysisResponse(BaseModel):
    classification: str
    confidence: float
    interpretability: Dict[str, Any]
    recommendations: List[str]

@app.get("/")
async def root():
    return {
        "service": "ECG Interpreter",
        "status": "healthy", 
        "model": "OTSN",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ecg_interpreter"}

@app.post("/analyze")
async def analyze_ecg(request: ECGAnalysisRequest) -> ECGAnalysisResponse:
    """Analyze ECG data using OTSN model"""
    # Mock response - will implement actual OTSN model
    return ECGAnalysisResponse(
        classification="Normal Sinus Rhythm",
        confidence=0.92,
        interpretability={
            "important_features": ["P-wave", "QRS complex", "T-wave"],
            "attention_weights": [0.3, 0.5, 0.2]
        },
        recommendations=[
            "ECG appears normal",
            "Continue regular monitoring",
            "Consult physician if symptoms persist"
        ]
    )

@app.get("/model/info")
async def model_info():
    return {
        "model_name": "OTSN",
        "model_type": "CNN-Transformer",
        "input_format": "1D ECG signal",
        "classes": ["Normal", "Arrhythmia", "Myocardial Infarction"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
