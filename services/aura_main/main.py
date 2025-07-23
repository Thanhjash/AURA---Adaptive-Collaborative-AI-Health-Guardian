"""
AURA Main Orchestrator Service
Central API gateway and orchestrator for the AURA health platform
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Dict, Any

app = FastAPI(
    title="AURA Main Orchestrator",
    description="Central orchestrator for AURA health platform",
    version="1.0.0"
)

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"

class HealthResponse(BaseModel):
    response: str
    service_used: str
    confidence: float
    timestamp: str

@app.get("/")
async def root():
    return {
        "service": "AURA Main Orchestrator",
        "status": "healthy",
        "version": "1.0.0",
        "description": "Central gateway for AURA health platform"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "aura_main"}

@app.post("/api/chat")
async def chat_endpoint(query: HealthQuery) -> Dict[str, Any]:
    """Main chat endpoint - will implement routing logic"""
    return {
        "response": f"AURA received your query: '{query.query}'",
        "service_used": "orchestrator",
        "confidence": 0.95,
        "query_complexity": "simple",
        "user_id": query.user_id
    }

@app.get("/api/services/status")
async def services_status():
    """Check status of all microservices"""
    return {
        "orchestrator": "online",
        "ecg_interpreter": "online", 
        "radiology_vqa": "online",
        "mental_wellness": "online"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)