# services/ai_server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set cache directory (same as your test)
os.environ['HF_HOME'] = '/app/.hf_cache'

app = FastAPI(title="AURA AI Server", version="1.0.0")

# Global model (loaded once, shared)
MEDGEMMA_MODEL = None
MEDGEMMA_TOKENIZER = None

class ECGAnalysisRequest(BaseModel):
    ecg_data: str
    
class VQARequest(BaseModel):
    image_url: str
    question: str

class WellnessRequest(BaseModel):
    message: str

@app.on_event("startup")
async def load_models():
    global MEDGEMMA_MODEL, MEDGEMMA_TOKENIZER
    
    print("🚀 Loading MedGemma (shared for all services)...")
    model_name = "unsloth/medgemma-4b-it-bnb-4bit"
    
    try:
        # Try with quantization config first (like your working test)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("🔬 Loading MedGemma with 4-bit quantization...")
        MEDGEMMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        MEDGEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("✅ MedGemma loaded with quantization")
        
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        print("🔄 Falling back to direct loading...")
        
        try:
            MEDGEMMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            MEDGEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            print("✅ MedGemma loaded without quantization")
            
        except Exception as e2:
            print(f"❌ Failed to load MedGemma: {e2}")
            print("🔄 Loading placeholder model...")
            
            # Final fallback
            model_name = "microsoft/DialoGPT-medium"
            MEDGEMMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            MEDGEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            print("⚠️ Using placeholder model")
    
    # Fix tokenizer padding issue
    if MEDGEMMA_TOKENIZER.pad_token is None:
        MEDGEMMA_TOKENIZER.pad_token = MEDGEMMA_TOKENIZER.eos_token
    
    print(f"✅ Model loaded on {MEDGEMMA_MODEL.device}")
    print(f"📊 Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

@app.get("/")
async def root():
    return {"service": "AURA AI Server", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": MEDGEMMA_MODEL is not None}

@app.post("/ai/vqa")
async def medical_vqa(request: VQARequest):
    if MEDGEMMA_MODEL is None:
        raise HTTPException(503, "Model not loaded")
    
    prompt = f"As a medical imaging specialist, analyze this image and answer: {request.question}\n\nProvide a clear, professional medical assessment."
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

@app.post("/ai/wellness")
async def mental_wellness(request: WellnessRequest):
    if MEDGEMMA_MODEL is None:
        raise HTTPException(503, "Model not loaded")
    
    prompt = f"As a compassionate mental health professional, provide supportive guidance for: {request.message}\n\nRespond with empathy and practical advice."
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