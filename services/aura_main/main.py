# File: services/aura_main/main.py (VERSION V12 - FINAL)

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import from models.py to avoid circular dependency
from models import HealthQuery 
from lib.conversation_manager import ConversationManager

# --- INITIALIZATION ---
load_dotenv()
conversation_manager = ConversationManager()

# --- APP CONFIGURATION ---
app = FastAPI(
    title="AURA Main Orchestrator V12",
    description="Final Stream-Only, Conversation-Centric Architecture",
    version="12.0.0",
    lifespan=conversation_manager.lifespan
)

# --- MIDDLEWARE ---
ALLOWED_ORIGINS = [
    os.getenv("FRONTEND_URL_VERCEL"),
    os.getenv("FRONTEND_URL_EC2"), 
    "http://localhost:3000",
]
# Remove None values if env vars not set
ALLOWED_ORIGINS = [origin for origin in ALLOWED_ORIGINS if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MAIN ENDPOINT ---
@app.post("/chat-stream")
async def chat_stream_endpoint(query: HealthQuery):
    """
    Single responsibility: Forward request to ConversationManager and stream response.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive", 
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        conversation_manager.stream_turn(query), 
        headers=headers
    )

# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    """Health check delegates to ConversationManager."""
    return await conversation_manager.health_check()

# --- DEV SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)