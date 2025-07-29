# services/ai_server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import asyncio
import json
import re
from contextlib import asynccontextmanager
import google.generativeai as genai

# Set cache directory
os.environ['HF_HOME'] = '/app/.hf_cache'

# Global model variables
MEDGEMMA_MODEL = None
MEDGEMMA_TOKENIZER = None
MODEL_LOADED = False
STARTUP_ERROR = None
TRIAGE_CLIENT = None

async def load_models():
    global MEDGEMMA_MODEL, MEDGEMMA_TOKENIZER, MODEL_LOADED, STARTUP_ERROR, TRIAGE_CLIENT
    
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
        
        # Initialize Gemini for intelligent triage
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("⚠️ GOOGLE_API_KEY not found - triage will use fallback only")
                TRIAGE_CLIENT = None
            else:
                genai.configure(api_key=api_key)
                TRIAGE_CLIENT = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Gemini triage client initialized")
        except Exception as e:
            print(f"⚠️ Gemini triage setup failed: {e}")
            TRIAGE_CLIENT = None
        
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
    version="2.0.0",
    lifespan=lifespan
)

class VQARequest(BaseModel):
    image_url: str
    question: str

class WellnessRequest(BaseModel):
    message: str

class TriageQuery(BaseModel):
    query: str
    context: str = ""

@app.get("/")
async def root():
    return {
        "service": "AURA AI Server", 
        "status": "operational",
        "model_loaded": MODEL_LOADED,
        "triage_available": TRIAGE_CLIENT is not None,
        "capabilities": ["Medical VQA", "Wellness Analysis", "Intelligent Triage"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL_LOADED else "loading",
        "models_loaded": MODEL_LOADED,
        "triage_client_ready": TRIAGE_CLIENT is not None,
        "startup_error": STARTUP_ERROR,
        "gpu_available": torch.cuda.is_available() if torch else False
    }

@app.post("/ai/triage")
async def intelligent_triage(request: TriageQuery):
    """
    LLM-driven intelligent query classification
    Replaces all fixed keyword-based routing with adaptive AI decision making
    """
    
    if not TRIAGE_CLIENT:
        # Fallback to conservative classification if Gemini unavailable
        return _conservative_fallback_triage(request.query)
    
    # Enhanced triage prompt for medical context understanding
    triage_prompt = f"""You are an expert medical triage specialist with deep understanding of patient communication patterns and healthcare workflow optimization. Analyze this user query with full contextual awareness.

USER QUERY: "{request.query}"

ADDITIONAL CONTEXT: {request.context if request.context else "No additional context provided"}

CLASSIFICATION TASK: Determine the most appropriate category for this query based on medical urgency, complexity, care requirements, and optimal patient flow.

CATEGORIES:
- "simple_chitchat": Non-medical conversation, greetings, general questions unrelated to health
- "medical_query_low_priority": Single symptoms without red flags, general wellness questions, lifestyle advice, non-urgent concerns that benefit from progressive consultation
- "medical_query_high_priority": Complex symptoms, multiple concerning symptoms, specific medical conditions requiring specialist attention, persistent/worsening symptoms
- "medical_emergency": Life-threatening symptoms, severe acute conditions, situations requiring immediate medical intervention

ROUTING PHILOSOPHY - MULTI-PERSPECTIVE ANALYSIS:

**From Patient Experience Perspective:**
- Simple symptoms deserve empathetic progressive consultation, not overwhelming expert analysis
- Complex cases benefit from comprehensive expert review
- Emergency situations need immediate recognition and guidance

**From Clinical Workflow Perspective:**
- Single, mild symptoms (headache, minor pain, general fatigue) → Progressive consultation builds context
- Multiple symptoms, severe symptoms, or red flag indicators → Expert council for comprehensive analysis
- True emergencies → Immediate emergency guidance

**From System Resource Perspective:**
- Reserve expert council for cases that truly benefit from multi-specialist analysis
- Use progressive consultation to gather context before escalating
- Optimize routing based on complexity, not just medical keywords

CLASSIFICATION GUIDELINES:

**medical_query_low_priority** (Progressive Consultation First):
- Single common symptoms without severity indicators: "headache", "tired", "minor pain"
- General wellness questions: "vitamins", "exercise", "nutrition"
- Vague concerns: "not feeling well", "something feels off"
- Lifestyle questions: "sleep better", "stress management"

**medical_query_high_priority** (Expert Council Appropriate):
- Multiple concerning symptoms: "headache + vision changes + nausea"
- Symptoms with red flag descriptors: "severe", "sudden onset", "worst ever"
- Specific medical conditions: "diabetes management", "medication interactions"
- Persistent/worsening symptoms: "pain getting worse", "symptoms for weeks"

**medical_emergency** (Immediate Emergency Guidance):
- Chest pain + breathing difficulty
- Severe neurological symptoms: sudden weakness, severe headache
- Loss of consciousness, seizures
- Severe bleeding, trauma

CRITICAL INSTRUCTIONS:
1. Analyze symptom COMPLEXITY and SEVERITY, not just medical keywords
2. Consider patient journey - simple symptoms should start with progressive consultation
3. Reserve high-priority routing for cases that truly need specialist analysis
4. Be conservative with emergency classification - only clear emergencies
5. Return ONLY a valid JSON object with this exact structure:

{{
    "category": "exact_category_name",
    "confidence": 0.95,
    "reasoning": "Clear explanation focusing on routing rationale and patient flow optimization",
    "urgency_score": 0.85,
    "medical_indicators": ["specific", "clinical", "indicators", "identified"],
    "recommended_flow": "progressive_consultation|expert_council|emergency_referral",
    "routing_rationale": "Why this routing serves the patient best from workflow perspective"
}}

Analyze the query with optimal patient flow in mind and respond with ONLY the JSON object."""

    try:
        # Generate intelligent classification using Gemini
        response = TRIAGE_CLIENT.generate_content(triage_prompt)
        
        # Parse structured response
        triage_result = _parse_triage_response(response.text)
        
        if not triage_result:
            print("⚠️ Failed to parse triage response, using fallback")
            return _conservative_fallback_triage(request.query)
        
        # Add metadata
        triage_result["triage_model"] = "gemini-2.0-flash-exp"
        triage_result["query_length"] = len(request.query)
        triage_result["has_context"] = bool(request.context)
        triage_result["llm_driven"] = True
        
        print(f"🎯 Intelligent triage: {triage_result['category']} (confidence: {triage_result['confidence']:.2f})")
        
        return triage_result
        
    except Exception as e:
        print(f"❌ Triage generation failed: {e}")
        return _conservative_fallback_triage(request.query)

def _parse_triage_response(response_text: str):
    """Parse LLM triage response with multiple parsing strategies"""
    
    if not response_text:
        return None
    
    # Strategy 1: Look for JSON code blocks
    json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Find balanced JSON object
    brace_count = 0
    start_idx = response_text.find('{')
    if start_idx == -1:
        return None
    
    in_string = False
    escape_next = False
    
    for i, char in enumerate(response_text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = response_text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break
    
    # Strategy 3: Extract category if mentioned explicitly
    categories = ["simple_chitchat", "medical_query_low_priority", "medical_query_high_priority", "medical_emergency"]
    for category in categories:
        if category in response_text.lower():
            return {
                "category": category,
                "confidence": 0.7,
                "reasoning": "Extracted category from partial response",
                "urgency_score": 0.5,
                "medical_indicators": [],
                "recommended_flow": "progressive_consultation",
                "semantic_analysis": "Partial parsing recovery"
            }
    
    return None

def _conservative_fallback_triage(query: str):
    """
    Conservative fallback classification when LLM triage unavailable
    Uses basic pattern recognition as last resort
    """
    query_lower = query.lower()
    
    # Emergency patterns (high specificity required)
    emergency_patterns = [
        "can't breathe", "cannot breathe", "difficulty breathing", "severe pain",
        "crushing pain", "heart attack", "stroke", "unconscious", "bleeding heavily",
        "severe chest pain", "can't catch my breath"
    ]
    
    if any(pattern in query_lower for pattern in emergency_patterns):
        return {
            "category": "medical_emergency",
            "confidence": 0.6,
            "reasoning": "Emergency patterns detected - conservative fallback",
            "urgency_score": 0.9,
            "medical_indicators": ["emergency_language_patterns"],
            "recommended_flow": "emergency_referral",
            "semantic_analysis": "Pattern-based emergency detection",
            "fallback": True
        }
    
    # High-priority medical patterns
    medical_patterns = [
        "chest pain", "chest tight", "chest pressure", "shortness of breath",
        "dizzy", "dizziness", "nausea", "fever", "headache", "pain",
        "symptom", "feel sick", "not feeling well"
    ]
    
    if any(pattern in query_lower for pattern in medical_patterns):
        return {
            "category": "medical_query_high_priority", 
            "confidence": 0.5,
            "reasoning": "Medical symptom patterns detected - conservative fallback",
            "urgency_score": 0.7,
            "medical_indicators": ["symptom_language_patterns"],
            "recommended_flow": "expert_council",
            "semantic_analysis": "Pattern-based medical detection",
            "fallback": True
        }
    
    # Greeting and chitchat patterns
    chitchat_patterns = ["hello", "hi", "how are you", "good morning", "good day", "thanks", "thank you"]
    
    if any(pattern in query_lower for pattern in chitchat_patterns) and len(query.split()) <= 5:
        return {
            "category": "simple_chitchat",
            "confidence": 0.8,
            "reasoning": "Greeting patterns detected - conservative fallback", 
            "urgency_score": 0.1,
            "medical_indicators": [],
            "recommended_flow": "simple_response",
            "semantic_analysis": "Pattern-based greeting detection",
            "fallback": True
        }
    
    # Default conservative classification for unknown queries
    return {
        "category": "medical_query_low_priority",
        "confidence": 0.4,
        "reasoning": "Unknown query type - conservative medical classification",
        "urgency_score": 0.5,
        "medical_indicators": [],
        "recommended_flow": "progressive_consultation",
        "semantic_analysis": "Default conservative classification",
        "fallback": True
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