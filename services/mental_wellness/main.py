# services/mental_wellness/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="AURA Mental Wellness", version="1.0.0")

class ComfortRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"service": "AURA Mental Wellness", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/comfort")
async def provide_mental_support(request: ComfortRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://ai_server:9000/ai/wellness",
                json={"message": request.message}
            )
            return response.json()
    except Exception as e:
        raise HTTPException(500, f"AI service error: {str(e)}")