# services/ai_server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import asyncio
from contextlib import asynccontextmanager

# Set cache directory
os.environ['HF_HOME'] = '/app/.hf_cache'

# Global model variables
MEDGEMMA_MODEL = None
MEDGEMMA_TOKENIZER = None
MODEL_LOADED = False
STARTUP_ERROR = None

async def load_models():
    global MEDGEMMA_MODEL, MEDGEMMA_TOKENIZER, MODEL_LOADED, STARTUP_ERROR
    
    try:
        print("🚀 Loading MedGemma (shared for all services)...")
        model_name = "unsloth/medgemma-4b-it-bnb-4bit"
        
        # BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("🔬 Loading MedGemma with 4-bit quantization...")
        
        # Load tokenizer
        MEDGEMMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        if MEDGEMMA_TOKENIZER.pad_token is None:
            MEDGEMMA_TOKENIZER.pad_token = MEDGEMMA_TOKENIZER.eos_token
        
        # Load model with quantization
        MEDGEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print("✅ MedGemma loaded with quantization")        
        print(f"✅ Model loaded on {MEDGEMMA_MODEL.device}")
        print(f"📊 Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        MODEL_LOADED = True
        
    except Exception as e:
        print(f"❌ Failed to load MedGemma: {e}")
        STARTUP_ERROR = str(e)
        MODEL_LOADED = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🔄 Starting AI Server...")
    await load_models()
    yield
    # Shutdown
    print("🔄 Shutting down AI Server...")

app = FastAPI(
    title="AURA AI Server", 
    version="1.0.0",
    lifespan=lifespan
)

class VQARequest(BaseModel):
    image_url: str
    question: str

class WellnessRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {
        "service": "AURA AI Server", 
        "status": "operational",
        "model_loaded": MODEL_LOADED
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL_LOADED else "loading",
        "models_loaded": MODEL_LOADED,
        "startup_error": STARTUP_ERROR,
        "gpu_available": torch.cuda.is_available() if torch else False
    }

@app.post("/ai/vqa")
async def medical_vqa(request: VQARequest):
    if not MODEL_LOADED:
        raise HTTPException(503, f"Model not loaded. Error: {STARTUP_ERROR}")
    
    prompt = f"As a medical imaging specialist, analyze this image and answer: {request.question}\n\nProvide a clear, professional medical assessment."
    
    try:
        inputs = MEDGEMMA_TOKENIZER(prompt, return_tensors='pt').to(MEDGEMMA_MODEL.device)
        
        with torch.no_grad():
            outputs = MEDGEMMA_MODEL.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=MEDGEMMA_TOKENIZER.eos_token_id
            )
        
        response = MEDGEMMA_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        return {"answer": answer, "model": "MedGemma"}
        
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.post("/ai/wellness")
async def mental_wellness(request: WellnessRequest):
    if not MODEL_LOADED:
        raise HTTPException(503, f"Model not loaded. Error: {STARTUP_ERROR}")
    
    prompt = f"As a compassionate mental health professional, provide supportive guidance for: {request.message}\n\nRespond with empathy and practical advice."
    
    try:
        inputs = MEDGEMMA_TOKENIZER(prompt, return_tensors='pt').to(MEDGEMMA_MODEL.device)
        
        with torch.no_grad():
            outputs = MEDGEMMA_MODEL.generate(
                **inputs, 
                max_new_tokens=200, 
                temperature=0.8,
                do_sample=True,
                pad_token_id=MEDGEMMA_TOKENIZER.eos_token_id
            )
        
        response = MEDGEMMA_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        return {"response": answer, "model": "MedGemma"}
        
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)