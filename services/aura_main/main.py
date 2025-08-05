# File: services/aura_main/main.py (VERSION V12 - FIXED)

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import from models.py to avoid circular dependency
from models import HealthQuery 
from lib.conversation_manager import ConversationManager
from lib.personalization_manager import PersonalizationManager

# --- INITIALIZATION ---
load_dotenv()
conversation_manager = ConversationManager()
personalization_manager = PersonalizationManager()  # 🔍 FIXED: Added missing initialization

# --- APP CONFIGURATION ---
app = FastAPI(
    title="AURA Main Orchestrator V12",
    description="Final Stream-Only, Conversation-Centric Architecture",
    version="12.0.0",
    lifespan=conversation_manager.lifespan
)

# --- CẤU HÌNH CORS MIDDLEWARE (FIXED) ---
# Đây chính là "danh sách khách mời", cho phép Vercel và Localhost truy cập.
ALLOWED_ORIGINS = [
    os.getenv("FRONTEND_URL_VERCEL"),    # Vd: "https://aura-....vercel.app"
    os.getenv("FRONTEND_URL_EC2"),       # Vd: "https://44.217.60.106.sslip.io"
    "http://localhost:3000",             # Cho phép test ở local
]
# Loại bỏ các giá trị None nếu biến môi trường không được đặt
ALLOWED_ORIGINS = [origin for origin in ALLOWED_ORIGINS if origin]

# Nếu danh sách trống (để phòng hờ), cho phép tất cả để debug
if not ALLOWED_ORIGINS:
    print("⚠️ WARNING: No CORS origins specified, allowing all for debug purposes.")
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,       # Chỉ cho phép các địa chỉ trong danh sách
    allow_credentials=True,              # Cho phép gửi cookie/authentication
    allow_methods=["*"],                 # Cho phép tất cả các phương thức (GET, POST, etc.)
    allow_headers=["*"],                 # Cho phép tất cả các tiêu đề
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

@app.get("/session/{session_id}")  # 🔍 FIXED: Added missing /api prefix
async def get_session_history_endpoint(session_id: str):
    """Endpoint for frontend to reload specific session history."""
    try:
        history = await conversation_manager.get_session_history(session_id)
        if "error" in history:
            raise HTTPException(status_code=404, detail=history["error"])
        return history
    except Exception as e:
        print(f"Error in get_session_history endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")

# @app.get("/chat-history/{user_id}")  # 🔍 ADDED: Missing chat history endpoint
# async def get_chat_history_endpoint(user_id: str):
#     """Get chat history for user."""
#     try:
#         # This should be implemented in conversation_manager
#         return await conversation_manager.get_chat_history(user_id) 
#     except Exception as e:
#         print(f"Error getting chat history: {e}")
#         return []

# --- DEV SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)