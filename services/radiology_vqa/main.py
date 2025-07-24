# services/radiology_vqa/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="AURA Radiology VQA", version="1.0.0")

class VQARequest(BaseModel):
    image_url: str
    question: str

@app.get("/")
async def root():
    return {"service": "AURA Radiology VQA", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/vqa")
async def analyze_medical_image(request: VQARequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://ai_server:9000/ai/vqa",
                json={"image_url": request.image_url, "question": request.question}
            )
            return response.json()
    except Exception as e:
        raise HTTPException(500, f"AI service error: {str(e)}")