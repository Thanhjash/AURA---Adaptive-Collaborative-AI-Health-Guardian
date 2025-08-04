# File: services/ai_server/main.py (VERSION V12 - PURE INFERENCE ENGINE)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# GGUF support
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("⚠️ llama-cpp-python not installed. This service will not work.")
    print("Install with: pip install llama-cpp-python")

# --- CONFIGURATION & GLOBALS ---
os.environ['HF_HOME'] = '/app/.hf_cache'

MEDGEMMA_MODEL = None
MODEL_LOADED = False
STARTUP_ERROR = None

# --- MODEL LOADING LOGIC ---
async def load_gguf_model():
    """Load MedGemma GGUF model once during server startup."""
    global MEDGEMMA_MODEL, MODEL_LOADED, STARTUP_ERROR
    
    if not GGUF_AVAILABLE:
        STARTUP_ERROR = "llama-cpp-python is not installed."
        MODEL_LOADED = False
        return

    try:
        print("🚀 [AI-SERVER] Loading MedGemma GGUF (CPU optimized)...")
        model_path = "/app/.hf_cache/medgemma-4b-it-Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
            print(f"📥 [AI-SERVER] Model not found at {model_path}. Downloading...")
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="unsloth/medgemma-4b-it-GGUF",
                filename="medgemma-4b-it-Q4_K_M.gguf",
                cache_dir="/app/.hf_cache"
            )
            print("✅ [AI-SERVER] Model download complete.")
        
        print("🔬 [AI-SERVER] Initializing Llama model...")
        MEDGEMMA_MODEL = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=512,
            n_threads=None,  # Auto-use all CPU cores
            verbose=False,
            use_mmap=True,
            n_gpu_layers=0  # CPU-only
        )
        MODEL_LOADED = True
        print("✅ [AI-SERVER] MedGemma GGUF loaded successfully.")
        
    except Exception as e:
        STARTUP_ERROR = str(e)
        MODEL_LOADED = False
        print(f"❌ [AI-SERVER] CRITICAL ERROR during model loading: {e}")

# --- APP LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_gguf_model()
    yield
    print("🔄 [AI-SERVER] Shutting down.")

# --- APP INITIALIZATION ---
app = FastAPI(
    title="AURA AI Server (Pure GGUF Inference)", 
    version="3.0.0",
    lifespan=lifespan
)

# --- REQUEST MODEL ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250
    temperature: float = 0.7
    stop: list[str] = ["<|im_end|>", "\n\n"]

# --- MAIN ENDPOINT ---
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Pure inference endpoint: receives prompt, returns MedGemma-generated text.
    This is the single 'muscle' endpoint of this server.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {STARTUP_ERROR}"
        )
    
    try:
        print(f"🔍 [AI-SERVER] Generation request: {request.prompt[:100]}...")
        
        response = MEDGEMMA_MODEL(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            echo=False,
            repeat_penalty=1.1
        )
        
        generated_text = response['choices'][0]['text'].strip()
        print("✅ [AI-SERVER] Generation successful.")
        return {"response": generated_text, "model": "MedGemma-GGUF"}

    except Exception as e:
        print(f"❌ [AI-SERVER] Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# --- UTILITY ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "service": "AURA AI Server (Pure GGUF Inference)", 
        "status": "operational" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "inference_device": "CPU",
        "startup_error": STARTUP_ERROR,
    }