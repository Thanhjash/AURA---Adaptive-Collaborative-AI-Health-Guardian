# services/aura_main/main.py
"""
AURA Main Orchestrator Service - REFACTORED FOR STREAMING with Vercel AI SDK
- Implements "Narrative Stream" architecture with custom events.
- Retains non-streaming endpoint for internal testing.
- Centralized, reusable logic for both streaming and non-streaming modes.
"""
import os
import traceback
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response  
from pydantic import BaseModel
import uvicorn
import httpx
import json
import asyncio 
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from lib.rag_manager import RAGManager
from lib.personalization_manager import PersonalizationManager
from lib.conversation_manager import ConversationManager
from core.expert_council import expert_council

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI(
    title="AURA Main Orchestrator",
    description="LLM-Driven Health Platform with Expert Council Observability and Streaming",
    version="6.0.0-stream"
)

ALLOWED_ORIGINS = [
    "https://44-217-60-106.sslip.io",                     # chính domain SSLIP
    "https://aura-adaptive-collaborative-ai-heal.vercel.app",  # preview Vercel
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
rag_manager = RAGManager(SUPABASE_URL, SUPABASE_KEY)
personalization_manager = PersonalizationManager()
conversation_manager = ConversationManager()

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    image_url: Optional[str] = None
    force_expert_council: bool = False

class SessionContinuation(BaseModel):
    session_id: str
    message: str

class FeedbackRequest(BaseModel):
    user_id: str
    interaction_id: str
    feedback: Dict[str, Any]

# ==================== HELPER FUNCTIONS ====================

# 🔍 ALSO UPDATE: _get_or_create_session function
async def _get_or_create_session(query: HealthQuery, user_context: str) -> str:
    """
    REFACTOR: Centralized session management - ONLY CREATE IF NEEDED
    """
    # 🔍 DEBUG: Check what we received
    print(f"🔍 _get_or_create_session called:")
    print(f"   - query.session_id: {query.session_id}")
    print(f"   - user_id: {query.user_id}")
    
    if query.session_id:
        print(f"🔄 Session already exists: {query.session_id}")
        return query.session_id
    
    print(f"🆕 Creating NEW session for user: {query.user_id}")
    new_session_data = await conversation_manager.start_conversation(
        user_id=query.user_id,
        initial_query=query.query,
        user_context=user_context,
        rag_context=""  # RAG context added later if needed
    )
    
    session_id = new_session_data.get("session_id")
    print(f"✅ NEW Session created: {session_id}")
    return session_id

async def _robust_ai_server_call(endpoint: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """
    REFACTOR: Centralized AI server communication with enhanced error handling
    OPTIMIZED: 30s timeout for MedGemma startup/inference
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(f"http://ai_server:9000{endpoint}", json=payload)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
            return response.json()
            
    except httpx.HTTPStatusError as http_err:
        error_detail = f"HTTP {http_err.response.status_code}"
        try:
            error_body = http_err.response.text
            if error_body:
                error_detail += f": {error_body[:200]}"
        except:
            pass
        print(f"❌ AI Server HTTP Error [{endpoint}]: {error_detail}")
        raise
        
    except httpx.TimeoutException:
        print(f"⏰ AI Server Timeout [{endpoint}]: {timeout}s exceeded (MedGemma startup/inference)")
        raise
        
    except Exception as e:
        print(f"❌ AI Server Connection Error [{endpoint}]: {type(e).__name__} - {e}")
        raise

# ==================== MAIN ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "service": "AURA Main Orchestrator",
        "status": "healthy", 
        "version": "5.1.0",
        "capabilities": [
            "LLM-Driven Routing", 
            "Progressive Consultation", 
            "Structured Expert Council", 
            "RAG", 
            "Personalization",
            "Council Session Observability"
        ],
        "description": "Intelligent medical conversation platform with comprehensive Expert Council analytics",
        "routing_system": "llm_driven_classification",
        "enhancements": [
            "intelligent_triage", 
            "adaptive_routing", 
            "semantic_analysis",
            "council_session_logging",
            "performance_analytics",
            "refactored_session_management"
        ]
    }

@app.get("/health")
async def health_check():
    # Check all systems including council session observability
    rag_health = await rag_manager.health_check()
    personalization_health = await personalization_manager.health_check()
    conversation_health = await conversation_manager.health_check()
    
    # Test intelligent triage system
    triage_health = await _test_triage_health()
    
    return {
        "status": "healthy", 
        "service": "aura_main",
        "version": "5.1.0",
        "systems": {
            "rag_system": rag_health,
            "personalization_system": personalization_health,
            "conversation_system": conversation_health,
            "intelligent_triage": triage_health,
            "expert_council": {
                "status": "ready", 
                "models": ["MedGemma-4B", "Gemini-2.5-Flash", "Gemini-2.0-Flash"],
                "workflow": "medagent_pro_structured_v3",
                "features": [
                    "structured_output", 
                    "interactive_components", 
                    "fail_fast",
                    "session_observability"
                ]
            }
        },
        "routing": {
            "type": "llm_driven_semantic_analysis",
            "fallback": "conservative_pattern_matching",
            "model": "gemini-2.0-flash-exp"
        },
        "observability": {
            "council_sessions_tracked": personalization_health.get("council_sessions_tracked", 0),
            "session_analytics": "enabled",
            "debugging_tools": "available"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/chat")
async def intelligent_chat_endpoint(query: HealthQuery) -> Dict[str, Any]:
    """
    REFACTORED: Enhanced Chat Endpoint with consistent session management
    - Session created/retrieved at the start
    - All handlers guaranteed to have session_id
    - Clean separation of concerns
    """
    
    # Get or create user profile for personalization
    profile = await personalization_manager.get_user_profile(query.user_id)
    if not profile:
        profile = await personalization_manager.create_user_profile(query.user_id)
    
    # Get user context for personalization
    user_context = await personalization_manager.get_user_context(query.user_id)
    print(f"👤 User: {query.user_id} | Context: {len(user_context)} chars")
    
    # ===== REFACTOR: CENTRALIZED SESSION MANAGEMENT =====
    # Ensure session exists BEFORE any processing
    # This eliminates all session-related bugs in handlers
    session_id = await _get_or_create_session(query, user_context)
    query.session_id = session_id  # Update query object for handlers
    
    # Force Expert Council if explicitly requested
    if query.force_expert_council:
        return await _handle_direct_expert_council(query)
    
    # === LLM-DRIVEN INTELLIGENT TRIAGE ===
    print(f"🧠 Semantic analysis: '{query.query[:50]}...'")
    triage_result = await _intelligent_semantic_triage(query.query, user_context)
    
    print(f"🎯 Category: {triage_result['category']} | Confidence: {triage_result['confidence']:.2f} | Indicators: {triage_result.get('medical_indicators', [])}")
    
    # Get RAG context for medical queries
    rag_context = ""
    if triage_result['category'] in ['medical_query_low_priority', 'medical_query_high_priority', 'medical_emergency']:
        rag_context = await rag_manager.get_context_for_query(query.query)
    
    # === INTELLIGENT ROUTING DECISIONS ===
    routing_decision = _determine_intelligent_routing(triage_result, query.session_id)
    print(f"🚦 Route: {routing_decision['strategy']} | Reason: {routing_decision['reason'][:60]}...")
    
    # Execute intelligent routing strategy
    if routing_decision['strategy'] == 'emergency_with_expert_analysis':
        response = await _handle_emergency_with_expert_analysis(query, triage_result, user_context, rag_context)
    elif routing_decision['strategy'] == 'emergency_guidance':
        response = await _handle_emergency_guidance(query, triage_result, user_context)
    elif routing_decision['strategy'] == 'direct_expert_council':
        response = await _handle_direct_expert_council(query, triage_result)
    elif routing_decision['strategy'] == 'progressive_consultation':
        response = await _handle_progressive_consultation(query, user_context, rag_context, triage_result)
    elif routing_decision['strategy'] == 'simple_response':
        response = await _handle_simple_response(query, user_context, triage_result)
    else:
        # Intelligent fallback
        response = await _handle_progressive_consultation(query, user_context, rag_context, triage_result)
    
    # Log interaction for personalization
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        response.get("response", "")[:100] + "...",
        response.get("service_used", "intelligent_orchestrator"),
        response.get("confidence", 0.8),
        session_id=session_id
    )
    
    # ===== REFACTOR: GUARANTEED SESSION_ID IN RESPONSE =====
    # Ensure session_id is always in the response for frontend
    response["session_id"] = session_id
    
    # Enhance response with intelligent routing metadata
    enhanced_response = {
        **response,
        "user_id": query.user_id,
        "personalized": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flow_type": "llm_driven_intelligent_routing",
        "triage_analysis": {
            "category": triage_result['category'],
            "confidence": triage_result['confidence'],
            "reasoning": triage_result['reasoning'],
            "urgency_score": triage_result.get('urgency_score', 0.5),
            "medical_indicators": triage_result.get('medical_indicators', []),
            "semantic_analysis": triage_result.get('semantic_analysis', ''),
            "llm_driven": triage_result.get('llm_driven', False)
        },
        "routing_decision": routing_decision,
        "context_provided": {
            "user_context_length": len(user_context),
            "rag_context_length": len(rag_context),
            "intelligent_triage": True,
            "llm_driven": triage_result.get("llm_driven", False)
        }
    }
    
    return enhanced_response

# ==================== HANDLER FUNCTIONS (REFACTORED) ====================

async def _intelligent_semantic_triage(query: str, user_context: str) -> Dict[str, Any]:
    """
    REFACTORED: LLM-driven semantic query analysis with enhanced error handling
    """
    try:
        result = await _robust_ai_server_call(
            "/ai/triage",
            {"query": query, "context": user_context[:200]}
        )
        print(f"✅ Semantic analysis successful: {result['category']}")
        return result
        
    except Exception as e:
        print(f"❌ Semantic analysis failed, using conservative fallback: {type(e).__name__}")
        return _emergency_conservative_fallback(query)

async def _handle_simple_response(query: HealthQuery, user_context: str, triage_result: Dict) -> Dict[str, Any]:
    """
    REFACTORED: Handle simple chitchat - session management already handled by main endpoint
    """
    # Session guaranteed to exist at this point - no need to create
    
    try:
        response_data = await _robust_ai_server_call(
            "/ai/wellness",
            {"message": f"Respond naturally and helpfully to this non-medical query: {query.query}"}
        )
        
        return {
            "response": response_data.get("response", "Hello! How can I help you today?"),
            "service_used": "llm_simple_conversation",
            "confidence": 0.8,
            "category": "chitchat",
            "triage_category": triage_result['category'],
            "llm_analysis": triage_result.get('semantic_analysis', '')
        }
        
    except Exception as e:
        print(f"❌ Simple response AI call failed: {type(e).__name__} - {e}")
        # Graceful fallback
        return {
            "response": "Hello! I'm AURA, your health companion. How can I help you today?",
            "service_used": "simple_conversation_fallback",
            "confidence": 0.6,
            "category": "chitchat",
            "triage_category": triage_result['category']
        }

async def _handle_emergency_with_expert_analysis(query: HealthQuery, triage_result: Dict, user_context: str, rag_context: str) -> Dict[str, Any]:
    """
    ENHANCED: Handle true emergencies with both immediate guidance AND expert analysis
    Real emergencies need immediate actions + comprehensive expert assessment
    """
    
    medical_indicators = triage_result.get('medical_indicators', ['Emergency symptoms'])
    urgency_score = triage_result.get('urgency_score', 0.9)
    
    # IMMEDIATE EMERGENCY GUIDANCE (for quick action)
    immediate_guidance = f"""🚨 **URGENT MEDICAL ATTENTION NEEDED**

**IMMEDIATE ACTIONS - DO NOW:**
• Call emergency services (911) immediately
• Go to nearest emergency room - do not drive yourself
• If possible, have someone stay with you
• Keep phone nearby and stay calm

**Your situation:** AI analysis indicates {urgency_score:.0%} urgency with symptoms: {', '.join(medical_indicators)}"""

    # PARALLEL: Run Expert Council for detailed analysis
    print("🏥 Running Expert Council for emergency case...")
    enhanced_user_context = f"EMERGENCY CASE - user_id: {query.user_id}\n{user_context}"
    
    try:
        council_result = await expert_council.run_expert_council(
            query=f"EMERGENCY: {query.query}",
            user_context=enhanced_user_context,
            rag_context=rag_context
        )
        
        # Extract structured data if successful
        if council_result.get("success"):
            structured_analysis = council_result.get("structured_analysis", {})
            interactive_components = council_result.get("interactive_components", {})
            expert_response = council_result.get("user_response", "")
            confidence = council_result.get("confidence", 0.8)
            
            # Combined response: Immediate + Expert analysis
            combined_response = f"""{immediate_guidance}

---

**🏥 EXPERT MEDICAL ANALYSIS**

{expert_response}

⚠️ **Critical:** This combines immediate emergency guidance with AI medical analysis. Only medical professionals can provide definitive emergency care."""

            return {
                "response": combined_response,
                "structured_analysis": structured_analysis,  # For frontend components
                "interactive_components": interactive_components,
                "immediate_guidance": immediate_guidance,  # Separate immediate actions
                "expert_analysis": expert_response,  # Separate expert analysis
                "service_used": "emergency_with_expert_analysis",
                "confidence": confidence,
                "category": "emergency",
                "triage_category": triage_result['category'],
                "urgency_level": "critical",
                "immediate_action_required": True,
                "emergency_indicators": medical_indicators,
                "expert_council_session": {
                    "session_id": council_result.get("session_id"),
                    "workflow": "emergency_expert_analysis"
                },
                "reasoning_trace": council_result.get('reasoning_trace', {}),
                "llm_analysis": {
                    "urgency_score": urgency_score,
                    "confidence": triage_result['confidence'],
                    "reasoning": triage_result['reasoning']
                }
            }
        
    except Exception as e:
        print(f"❌ Expert Council failed for emergency, using immediate guidance only: {e}")
    
    # Fallback: Immediate guidance only if Expert Council fails
    return {
        "response": immediate_guidance + "\n\n*Note: Detailed analysis temporarily unavailable. Please seek immediate medical attention.*",
        "immediate_guidance": immediate_guidance,
        "service_used": "emergency_guidance_fallback",
        "confidence": 0.95,
        "category": "emergency",
        "urgency_level": "critical",
        "immediate_action_required": True,
        "emergency_indicators": medical_indicators
    }

async def _handle_emergency_guidance(query: HealthQuery, triage_result: Dict, user_context: str) -> Dict[str, Any]:
    """
    REFACTORED: Handle emergency situations - session management already handled by main endpoint
    """
    # Session guaranteed to exist at this point - no need to create
    
    medical_indicators = triage_result.get('medical_indicators', ['Emergency symptoms'])
    urgency_score = triage_result.get('urgency_score', 0.9)
    
    emergency_response = f"""🚨 **URGENT MEDICAL ATTENTION NEEDED**

Based on AI analysis of your description: "{query.query}", this appears to be a medical emergency requiring immediate professional care.

**LLM Analysis Results:**
• Urgency Score: {urgency_score:.0%}
• Medical Indicators: {', '.join(medical_indicators)}
• Confidence: {triage_result['confidence']:.0%}

**IMMEDIATE ACTIONS:**
• Call emergency services (911) if in the US, or your local emergency number
• Go to the nearest emergency room immediately
• Do not drive yourself - call an ambulance or have someone drive you
• If possible, contact your doctor or healthcare provider

**WHILE WAITING FOR HELP:**
• Stay calm and try to remain still
• If you have prescribed emergency medications, follow your doctor's instructions
• Have someone stay with you if possible
• Keep your phone nearby

⚠️ **IMPORTANT:** This is an AI assessment based on semantic analysis of your description. Only medical professionals can provide definitive emergency care."""

    return {
        "response": emergency_response,
        "service_used": "llm_emergency_guidance_system",
        "confidence": 0.95,
        "category": "emergency",
        "triage_category": triage_result['category'],
        "urgency_level": "critical",
        "immediate_action_required": True,
        "emergency_indicators": medical_indicators,
        "llm_analysis": {
            "urgency_score": urgency_score,
            "confidence": triage_result['confidence'],
            "reasoning": triage_result['reasoning']
        }
    }

async def _handle_progressive_consultation(query: HealthQuery, user_context: str, rag_context: str, triage_result: Dict) -> Dict[str, Any]:
    """
    REFACTORED: Handle medical queries through progressive consultation
    OPTIMIZED: Reduced redundant escalation logic
    """
    # Session guaranteed to exist at this point
    
    # OPTIMIZED: Simplified flow - just continue conversation
    # Let conversation_manager handle escalation logic internally
    try:
        print(f"🔄 Progressive consultation for session: {query.session_id}")
        response = await conversation_manager.continue_conversation(
            session_id=query.session_id,
            user_message=query.query,
            user_context=user_context,
            rag_context=rag_context
        )
    except Exception as e:
        print(f"❌ Progressive consultation error: {e}")
        # Simple fallback
        return {
            "response": "I'm having trouble processing your request. Could you please rephrase your concern?",
            "service_used": "progressive_consultation_fallback",
            "confidence": 0.3
        }
    
    # Add LLM analysis metadata
    response["triage_analysis"] = {
        "category": triage_result['category'],
        "confidence": triage_result['confidence'],
        "reasoning": triage_result['reasoning'],
        "medical_indicators": triage_result.get('medical_indicators', [])
    }
    response["llm_driven_routing"] = True
    
    return response

async def _handle_direct_expert_council(query: HealthQuery, triage_result: Optional[Dict] = None) -> Dict[str, Any]:
    """
    REFACTORED: Enhanced Expert Council - session management already handled by main endpoint
    """
    # Session guaranteed to exist at this point - no need to create
    
    # Get user context and RAG context
    user_context = await personalization_manager.get_user_context(query.user_id)
    rag_context = await rag_manager.get_context_for_query(query.query)
    
    # Enhanced user context with explicit user_id
    enhanced_user_context = f"user_id: {query.user_id}\n{user_context}"
    
    # Run Enhanced Expert Council with LLM analysis context
    council_result = await expert_council.run_expert_council(
        query=query.query,
        user_context=enhanced_user_context,
        rag_context=rag_context
    )
    
    # Handle Expert Council failure
    if not council_result.get("success"):
        error_response = {
            "error": "Expert Council consultation failed",
            "error_type": council_result.get("error_type", "expert_council_failure"),
            "failed_step": council_result.get("failed_step", "unknown_step"),
            "error_message": council_result.get("error_message", "Expert Council encountered an error"),
            "suggestion": council_result.get("suggestion", "Please try again or start with basic consultation"),
            "user_response": council_result.get("user_response", "I'm experiencing technical difficulties. Please try again in a moment."),
            "service_used": "expert_council_error",
            "confidence": 0.1,
            "user_id": query.user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "council_session_id": council_result.get("session_id")
        }
        return error_response
    
    # Extract structured data from Expert Council
    confidence = council_result.get("confidence", 0.7)
    structured_analysis = council_result.get("structured_analysis", {})
    interactive_components = council_result.get("interactive_components", {})
    
    # Create enhanced response
    final_response = f"""**🏥 AURA Expert Council Analysis**

{council_result['user_response']}

**🤝 Expert Consensus:** {confidence:.0%} confidence
**👤 Personalized Analysis**

---
*This analysis was conducted by AURA's Expert Council using MedAgent-Pro methodology with multiple AI specialists.*

⚠️ **Important:** This analysis is for informational purposes. Please consult your healthcare provider for personalized medical advice."""
    
    # Log interaction
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        "Expert Council consultation completed",
        "expert_council_llm_routing",
        confidence,
        session_id=query.session_id
    )
    
    # Return enhanced response with LLM analysis context
    response = {
        "response": final_response,
        "structured_analysis": structured_analysis,
        "interactive_components": interactive_components,
        "service_used": "expert_council_llm_routing",
        "confidence": confidence,
        "query_complexity": "expert_council",
        "user_id": query.user_id,
        "knowledge_used": True,
        "personalized": True,
        "expert_council_session": {
            "session_id": council_result.get("session_id"),
            "experts_consulted": council_result.get("metadata", {}).get("experts_consulted", []),
            "evidence_sources": council_result.get("metadata", {}).get("evidence_sources", []),
            "workflow": council_result.get("metadata", {}).get("workflow", "medagent_pro_structured_v3")
        },
        "reasoning_trace": council_result.get('reasoning_trace', {}),
        "flow_type": "llm_driven_expert_council",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add LLM triage context if available
    if triage_result:
        response["semantic_analysis"] = {
            "category": triage_result['category'],
            "confidence": triage_result['confidence'],
            "reasoning": triage_result['reasoning'],
            "medical_indicators": triage_result.get('medical_indicators', [])
        }
    
    return response

# ==================== STREAMING IMPLEMENTATION (THE CORE NEW LOGIC) ====================
async def run_expert_council_stream(query: HealthQuery, user_context: str, rag_context: str):
    try:
        enhanced_user_context = f"user_id: {query.user_id}\n{user_context}"
        
        async for update in expert_council.run_expert_council_with_progress(
            query=query.query,
            user_context=enhanced_user_context,
            rag_context=rag_context
        ):
            if update["type"] == "progress":
                yield f"event: council_step\ndata: {json.dumps({'step': update['step'], 'status': update['status'], 'description': update['description']})}\n\n"
            
            elif update["type"] == "result":
                council_result = update["data"]
                
                if council_result.get("success"):
                    # Stream response text
                    response_text = council_result.get("user_response", "")
                    for char in response_text:
                        yield f"event: text_token\ndata: {json.dumps({'token': char})}\n\n"
                        await asyncio.sleep(0.01)
                    
                    # Send structured data
                    structured_data = {
                        'structured_analysis': council_result.get('structured_analysis', {}),
                        'interactive_components': council_result.get('interactive_components', {}),
                        'reasoning_trace': council_result.get('reasoning_trace', {}),
                        'confidence': council_result.get('confidence', 0.7)
                    }
                    yield f"event: council_complete\ndata: {json.dumps(structured_data)}\n\n"
                else:
                    # Handle error
                    error_message = council_result.get("user_response", "Expert Council failed")
                    for char in error_message:
                        yield f"event: text_token\ndata: {json.dumps({'token': char})}\n\n"
                        await asyncio.sleep(0.01)
    
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

async def stream_aura_response(query: HealthQuery):
    """
    HÀM GENERATOR CHÍNH CHO STREAMING (PHIÊN BẢN TỐI ƯU NHẤT)
    Điều phối toàn bộ logic và `yield` các gói tin JSON với các events chi tiết.
    """
    session_id = None
    try:
        # === BƯỚC 1: GIỮ ẤM KẾT NỐI (FIX LỖI NETWORK ERROR) ===
        # Gửi ngay một gói tin để trình duyệt và NGINX biết kết nối đang hoạt động
        # trong khi chúng ta chờ các tác vụ nặng ở phía sau.
        yield f"event: task_started\ndata: {json.dumps({'message': 'Request received, preparing analysis...'})}\n\n"
        await asyncio.sleep(0.01) # Đảm bảo gói tin được gửi đi

        # --- Giai đoạn 2: Khởi tạo & Phân tích (Logic cũ) ---
        print(f"🔍 STREAM REQUEST DEBUG:")
        print(f"   - Query: {query.query[:50]}...")
        print(f"   - User ID: {query.user_id}")
        print(f"   - Incoming session_id: {query.session_id}")

        user_context = await personalization_manager.get_user_context(query.user_id)
        
        if query.session_id:
            session_id = query.session_id
            yield f"event: session_ready\ndata: {json.dumps({'session_id': session_id, 'message': 'Existing session restored.'})}\n\n"
        else:
            session_id = await _get_or_create_session(query, user_context)
            query.session_id = session_id # Cập nhật session_id vào query object
            yield f"event: session_ready\ndata: {json.dumps({'session_id': session_id, 'message': 'New session created.'})}\n\n"
        
        print(f"✅ Final session_id for stream: {session_id}")
        
        yield f"event: analysis_start\ndata: {json.dumps({'message': 'Performing semantic triage (can be slow on first load)...'})}\n\n"
        triage_result = await _intelligent_semantic_triage(query.query, user_context)
        routing_decision = _determine_intelligent_routing(triage_result, query.session_id)
        yield f"event: analysis_complete\ndata: {json.dumps({'category': triage_result['category'], 'routing': routing_decision['strategy']})}\n\n"
        
        rag_context = ""
        if triage_result['category'] not in ['simple_chitchat']:
            yield f"event: knowledge_search\ndata: {json.dumps({'message': 'Searching medical knowledge base...'})}\n\n"
            rag_context = await rag_manager.get_context_for_query(query.query)

        # --- Giai đoạn 3: Định tuyến đến luồng xử lý phù hợp ---
        strategy = routing_decision['strategy']
        print(f"🚦 Routing strategy: {strategy}")

        if strategy in ['emergency_with_expert_analysis', 'direct_expert_council'] or query.force_expert_council:
            async for chunk in run_expert_council_stream(query, user_context, rag_context):
                yield chunk
        
        elif strategy == 'simple_response':
            response_data = await _handle_simple_response(query, user_context, triage_result)
            response_text = response_data.get("response", "Hello! How can I help you today?")
            for word in response_text.split():
                yield f"event: text_token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.04)

        else: # Progressive consultation
            consultation_result = await _handle_progressive_consultation(query, user_context, rag_context, triage_result)
            response_text = consultation_result.get("response", "Thank you for your question. Let me look into that for you.")
            for word in response_text.split():
                yield f"event: text_token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.05)
    
    except Exception as e:
        print(f"❌ STREAMING ERROR in main generator: {e}\n{traceback.format_exc()}")
        error_info = {"error": "An unexpected error occurred during the stream.", "detail": str(e)}
        yield f"event: error\ndata: {json.dumps(error_info)}\n\n"

    finally:
        final_data = {'message': 'Stream finished.', 'session_id': session_id}
        yield f"event: stream_end\ndata: {json.dumps(final_data)}\n\n"
        print(f"🏁 Stream completed for session: {session_id}")


# ==================== UTILITY FUNCTIONS ====================

def _emergency_conservative_fallback(query: str) -> Dict[str, Any]:
    """Emergency conservative fallback when all LLM systems unavailable"""
    # Very basic emergency detection as absolute last resort
    query_lower = query.lower()
    
    if any(emergency in query_lower for emergency in ["can't breathe", "severe pain", "unconscious", "heart attack"]):
        urgency = 0.9
        category = "medical_emergency"
    elif any(medical in query_lower for medical in ["pain", "chest", "dizzy", "fever", "symptom"]):
        urgency = 0.7
        category = "medical_query_high_priority"
    else:
        urgency = 0.5
        category = "medical_query_low_priority"
    
    return {
        "category": category,
        "confidence": 0.6,
        "reasoning": "Emergency fallback - LLM triage unavailable",
        "urgency_score": urgency,
        "medical_indicators": [],
        "recommended_flow": "progressive_consultation",
        "semantic_analysis": "Pattern-based emergency fallback",
        "llm_driven": False,
        "fallback": True
    }

def _determine_intelligent_routing(triage_result: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
    """
    Determine routing strategy based on semantic AI analysis
    Uses LLM insights for intelligent decision making
    """
    category = triage_result['category']
    confidence = triage_result['confidence']
    urgency_score = triage_result.get('urgency_score', 0.5)
    medical_indicators = triage_result.get('medical_indicators', [])
    
    # Emergency handling with dual approach
    if category == 'medical_emergency' or urgency_score > 0.85:
        return {
            "strategy": "emergency_with_expert_analysis",  # NEW: Combined approach
            "reason": f"Emergency requiring immediate guidance + expert analysis (urgency: {urgency_score:.2f})",
            "bypass_conversation": False,
            "ai_confidence": confidence,
            "urgency_score": urgency_score,
            "medical_indicators": medical_indicators
        }
    
    # High-priority medical with high AI confidence → Expert Council
    if category == 'medical_query_high_priority' and confidence > 0.75:
        return {
            "strategy": "direct_expert_council", 
            "reason": f"High-priority medical query detected by LLM (confidence: {confidence:.2f})",
            "bypass_conversation": False,
            "ai_confidence": confidence
        }
    
    # Simple chitchat detected by LLM
    if category == 'simple_chitchat':
        return {
            "strategy": "simple_response",
            "reason": f"Non-medical conversation detected by semantic analysis",
            "bypass_conversation": True,
            "ai_confidence": confidence
        }
    
    # Progressive consultation for medical queries with context consideration
    if category in ['medical_query_low_priority', 'medical_query_high_priority']:
        # If continuing session, consider Expert Council escalation
        if session_id and category == 'medical_query_high_priority':
            strategy = "direct_expert_council"
            reason = f"High-priority medical query in existing session - escalating to Expert Council"
        else:
            strategy = "progressive_consultation"
            reason = f"Medical query suitable for progressive consultation (category: {category})"
        
        return {
            "strategy": strategy,
            "reason": reason,
            "bypass_conversation": False,
            "ai_confidence": confidence
        }
    
    # Intelligent fallback
    return {
        "strategy": "progressive_consultation",
        "reason": f"LLM analysis suggests progressive approach (category: {category})",
        "bypass_conversation": False,
        "ai_confidence": confidence
    }

async def _test_triage_health() -> Dict[str, Any]:
    """Test intelligent triage system health"""
    try:
        # Use basic health endpoint instead of heavy triage call
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get("http://ai_server:9000/health")
            response.raise_for_status()
            health_data = response.json()
            
            return {
                "status": "healthy" if health_data.get("status") == "healthy" else "degraded",
                "models_loaded": health_data.get("models_loaded", False),
                "gpu_available": health_data.get("gpu_available", False)
            }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}

# ==================== NEW/UPDATED ENDPOINTS ====================

@app.options("/chat-stream")
async def options_chat_stream():
    """Handle preflight requests for streaming endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# Update your streaming endpoint headers:
@app.post("/chat-stream")
async def chat_stream_endpoint(query: HealthQuery):
    """Enhanced streaming endpoint with proper CORS headers"""
    
    # Enhanced headers for better streaming compatibility
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "*"
    }
    
    return StreamingResponse(
        stream_aura_response(query), 
        media_type="text/event-stream",
        headers=headers
    )

async def handle_non_stream_chat(query: HealthQuery) -> Dict[str, Any]:
    """
    Hàm logic chính được tách ra để endpoint non-streaming có thể tái sử dụng.
    Nó gọi các hàm xử lý cũ của bạn.
    """
    profile = await personalization_manager.get_user_profile(query.user_id)
    if not profile:
        await personalization_manager.create_user_profile(query.user_id)
    
    user_context = await personalization_manager.get_user_context(query.user_id)
    session_id = await _get_or_create_session(query, user_context)
    query.session_id = session_id
    
    if query.force_expert_council:
        return await _handle_direct_expert_council(query)
        
    triage_result = await _intelligent_semantic_triage(query.query, user_context)
    rag_context = ""
    if triage_result['category'] not in ['simple_chitchat']:
        rag_context = await rag_manager.get_context_for_query(query.query)
        
    routing_decision = _determine_intelligent_routing(triage_result, query.session_id)
    
    strategy = routing_decision['strategy']
    response = {}
    if strategy == 'emergency_with_expert_analysis':
        response = await _handle_emergency_with_expert_analysis(query, triage_result, user_context, rag_context)
    elif strategy == 'emergency_guidance':
        response = await _handle_emergency_guidance(query, triage_result, user_context)
    elif strategy == 'direct_expert_council':
        response = await _handle_direct_expert_council(query, triage_result)
    elif strategy == 'progressive_consultation':
        response = await _handle_progressive_consultation(query, user_context, rag_context, triage_result)
    elif strategy == 'simple_response':
        response = await _handle_simple_response(query, user_context, triage_result)
    else:
        response = await _handle_progressive_consultation(query, user_context, rag_context, triage_result)
        
    response["session_id"] = session_id
    # ... (phần code còn lại để thêm metadata vào response)
    return response

# ==================== EXISTING ENDPOINTS (UNCHANGED) ====================

@app.post("/chat/continue")
async def continue_session(continuation: SessionContinuation) -> Dict[str, Any]:
    """Continue an existing conversation session"""
    user_id = continuation.session_id.split("_")[-1] if "_" in continuation.session_id else "anonymous"
    
    # Get contexts for continuation
    user_context = await personalization_manager.get_user_context(user_id)
    rag_context = await rag_manager.get_context_for_query(continuation.message)
    
    return await conversation_manager.continue_conversation(
        continuation.session_id,
        continuation.message,
        user_context,
        rag_context
    )

@app.get("/chat/session/{session_id}/history")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """Get complete session conversation history"""
    return await conversation_manager.get_session_history(session_id)

@app.post("/feedback")
async def log_user_feedback(feedback_request: FeedbackRequest) -> Dict[str, Any]:
    """Log user feedback for Human-in-the-Loop learning"""
    success = await personalization_manager.log_interaction_feedback(
        feedback_request.user_id,
        feedback_request.interaction_id,
        feedback_request.feedback
    )
    
    if success:
        return {
            "status": "success",
            "message": "Feedback logged successfully",
            "interaction_id": feedback_request.interaction_id
        }
    else:
        return {
            "status": "error",
            "message": "Failed to log feedback"
        }

@app.get("/analytics/feedback")
async def get_feedback_analytics() -> Dict[str, Any]:
    """Get system feedback analytics for improvement insights"""
    return await personalization_manager.get_feedback_analytics()

@app.get("/user/{user_id}/export")
async def export_user_data(user_id: str) -> Dict[str, Any]:
    """Export all user data for privacy compliance"""
    return await personalization_manager.get_user_data_export(user_id)

@app.delete("/user/{user_id}")
async def delete_user_account(user_id: str) -> Dict[str, Any]:
    """Delete user account and all associated data"""
    success = await personalization_manager.delete_user_data(user_id)
    
    if success:
        return {
            "status": "success",
            "message": "User data deleted successfully",
            "user_id": user_id
        }
    else:
        return {
            "status": "error",
            "message": "Failed to delete user data"
        }

@app.get("/knowledge/search")
async def search_knowledge(query: str, limit: int = 3):
    """Direct knowledge search endpoint for testing"""
    results = await rag_manager.semantic_search(query, max_results=limit)
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

@app.get("/services/status")
async def services_status():
    """Check status of all microservices"""
    rag_health = await rag_manager.health_check()
    conversation_health = await conversation_manager.health_check()
    triage_health = await _test_triage_health()
    
    return {
        "orchestrator": "online",
        "rag_system": rag_health["status"],
        "personalization": "online",
        "conversation_manager": conversation_health["status"],
        "intelligent_triage": triage_health["status"],
        "expert_council": "ready_structured_observable",
        "ecg_interpreter": "online", 
        "radiology_vqa": "online",
        "mental_wellness": "online",
        "ai_server": "online",
        "knowledge_entries": rag_health.get("knowledge_entries", 0),
        "routing": "llm_driven_semantic_analysis"
    }

# Testing and debugging endpoints for LLM routing
@app.get("/triage/test")
async def test_semantic_triage(query: str, context: str = ""):
    """Test intelligent semantic triage classification"""
    result = await _intelligent_semantic_triage(query, context)
    routing_decision = _determine_intelligent_routing(result, None)
    
    return {
        "test_query": query,
        "triage_analysis": result,
        "routing_decision": routing_decision,
        "llm_driven": result.get("llm_driven", False)
    }

@app.get("/triage/debug")
async def debug_semantic_triage():
    """Debug semantic triage system with multiple test cases"""
    test_cases = [
        "Hello how are you?",
        "I have a persistent headache for 3 days",
        "I can't breathe and have crushing chest pain",
        "What vitamins should I take for energy?",
        "My chest feels tight when I exercise",
        "I feel dizzy and nauseous since this morning",
        "I think I'm having a heart attack",
        "What's the weather like today?",
        "I feel a bit off in my chest area"
    ]
    
    results = []
    for test_query in test_cases:
        semantic_analysis = await _intelligent_semantic_triage(test_query, "")
        routing_decision = _determine_intelligent_routing(semantic_analysis, None)
        
        results.append({
            "query": test_query,
            "triage_analysis": semantic_analysis,
            "routing_decision": routing_decision,
            "llm_driven": semantic_analysis.get("llm_driven", False)
        })
    
    return {
        "test_results": results,
        "summary": {
            "total_tests": len(test_cases),
            "categories": list(set(r["triage_analysis"]["category"] for r in results)),
            "routing_strategies": list(set(r["routing_decision"]["strategy"] for r in results)),
            "llm_driven_count": sum(1 for r in results if r["llm_driven"]),
            "fallback_count": sum(1 for r in results if r["triage_analysis"].get("fallback", False))
        }
    }

# === EXPERT COUNCIL SESSION OBSERVABILITY ENDPOINTS (UNCHANGED) ===

@app.get("/council-sessions/recent")
async def get_recent_council_sessions(
    limit: int = Query(default=10, description="Number of recent sessions to retrieve", ge=1, le=50),
    include_successful_only: bool = Query(default=False, description="Only include successful sessions")
) -> Dict[str, Any]:
    """Get recent Expert Council sessions for monitoring and debugging"""
    try:
        sessions = await personalization_manager.get_recent_council_sessions(
            limit=limit, 
            successful_only=include_successful_only
        )
        
        return {
            "recent_sessions": sessions,
            "total_returned": len(sessions),
            "filters_applied": {
                "limit": limit,
                "successful_only": include_successful_only
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A server error occurred while retrieving recent sessions: {str(e)}")

@app.get("/council-sessions/{session_id}")
async def get_council_session(session_id: str) -> Dict[str, Any]:
    """Retrieve specific Expert Council session for debugging and review"""
    session_data = await personalization_manager.get_council_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Council session {session_id} not found")
    
    return {
        "session_id": session_id,
        "session_data": session_data,
        "retrieved_at": datetime.now(timezone.utc).isoformat()
    }

@app.get("/analytics/council-performance")
async def get_council_performance_analytics(
    days: int = Query(default=30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """Get comprehensive Expert Council performance analytics"""
    analytics = await personalization_manager.get_council_analytics(days=days)
    
    return {
        "analytics_period": f"Last {days} days",
        "performance_metrics": analytics,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

@app.get("/council-sessions/search")
async def search_council_sessions(
    user_id: Optional[str] = Query(default=None, description="Filter by specific user ID"),
    success_only: bool = Query(default=False, description="Only successful sessions"),
    min_confidence: float = Query(default=0.0, description="Minimum confidence score", ge=0.0, le=1.0),
    days: int = Query(default=7, description="Days to search back", ge=1, le=90)
) -> Dict[str, Any]:
    """Search Expert Council sessions with filters"""
    try:
        from google.cloud import firestore
        from datetime import timedelta
        
        # Build query with filters
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = personalization_manager.db.collection('council_sessions')
        query = query.where('timestamp', '>=', cutoff_date)
        
        if user_id:
            query = query.where('user_id', '==', user_id)
        
        if success_only:
            query = query.where('success', '==', True)
        
        # Execute query
        session_docs = await query.get()
        
        # Filter by confidence (Firestore doesn't support this natively)
        filtered_sessions = []
        for doc in session_docs:
            session_data = doc.to_dict()
            
            if session_data.get('confidence', 0.0) >= min_confidence:
                session_summary = {
                    "session_id": session_data.get("session_id"),
                    "timestamp": session_data.get("timestamp"),
                    "user_id": session_data.get("user_id"),
                    "success": session_data.get("success", False),
                    "confidence": session_data.get("confidence", 0.0),
                    "duration_seconds": session_data.get("duration_seconds", 0),
                    "original_query": session_data.get("original_query", "")[:100] + "..." if len(session_data.get("original_query", "")) > 100 else session_data.get("original_query", ""),
                    "primary_assessment": session_data.get("structured_analysis", {}).get("clinical_summary", {}).get("primary_assessment", ""),
                    "error_type": session_data.get("error_info", {}).get("error_type") if not session_data.get("success", False) else None
                }
                
                filtered_sessions.append(session_summary)
        
        # Sort by timestamp (most recent first)
        filtered_sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "matching_sessions": filtered_sessions,
            "total_found": len(filtered_sessions),
            "search_criteria": {
                "user_id": user_id,
                "success_only": success_only,
                "min_confidence": min_confidence,
                "days_searched": days
            },
            "searched_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)