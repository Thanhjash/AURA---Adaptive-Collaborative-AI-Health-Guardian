# services/aura_main/main.py
"""
AURA Main Orchestrator Service - Enhanced with Structured Expert Council
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional
from datetime import datetime
from lib.rag_manager import RAGManager
from lib.personalization_manager import PersonalizationManager
from lib.conversation_manager import ConversationManager
from core.expert_council import expert_council

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI(
    title="AURA Main Orchestrator",
    description="Enhanced platform with structured Expert Council and progressive consultation",
    version="3.2.0"
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

@app.get("/")
async def root():
    return {
        "service": "AURA Main Orchestrator",
        "status": "healthy", 
        "version": "3.2.0",
        "capabilities": ["Progressive Consultation", "Structured Expert Council", "RAG", "Personalization"],
        "description": "Enhanced medical conversation platform with structured AI analysis",
        "conversation_flow": "Triage → Context Building → Smart Escalation → Structured Expert Council",
        "enhancements": ["structured_analysis", "interactive_components", "fail_fast_expert_council"]
    }

@app.get("/health")
async def health_check():
    # Check all systems
    rag_health = await rag_manager.health_check()
    personalization_health = await personalization_manager.health_check()
    conversation_health = await conversation_manager.health_check()
    
    return {
        "status": "healthy", 
        "service": "aura_main",
        "version": "3.2.0",
        "systems": {
            "rag_system": rag_health,
            "personalization_system": personalization_health,
            "conversation_system": conversation_health,
            "expert_council": {
                "status": "ready", 
                "models": ["MedGemma-4B", "Gemini-2.5-Flash", "Gemini-2.0-Flash"],
                "workflow": "medagent_pro_structured_v3",
                "features": ["structured_output", "interactive_components", "fail_fast"]
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/chat")
async def progressive_chat_endpoint(query: HealthQuery) -> Dict[str, Any]:
    """
    Progressive Consultation Chat Endpoint with Enhanced Expert Council
    """
    
    # Get or create user profile for personalization
    profile = await personalization_manager.get_user_profile(query.user_id)
    if not profile:
        profile = await personalization_manager.create_user_profile(query.user_id)
    
    # Force Expert Council if explicitly requested
    if query.force_expert_council:
        return await _handle_direct_expert_council(query)
    
    # === CONTEXT PASSING - ALWAYS FETCH CONTEXTS ===
    print(f"🔍 Fetching contexts for user: {query.user_id}")
    
    # Always get user context (personalization)
    user_context = await personalization_manager.get_user_context(query.user_id)
    print(f"👤 User context: {len(user_context)} chars")
    
    # Always get RAG context (medical knowledge)
    rag_context = await rag_manager.get_context_for_query(query.query)
    print(f"📚 RAG context: {len(rag_context)} chars")
    
    # Route based on session with FULL CONTEXT
    if query.session_id:
        print(f"🔄 Continuing conversation: {query.session_id}")
        # Continue existing conversation WITH CONTEXTS
        response = await conversation_manager.continue_conversation(
            session_id=query.session_id, 
            user_message=query.query,
            user_context=user_context,
            rag_context=rag_context
        )
    else:
        print(f"🆕 Starting new conversation")
        # Start new conversation WITH CONTEXTS
        response = await conversation_manager.start_conversation(
            user_id=query.user_id, 
            initial_query=query.query,
            user_context=user_context,
            rag_context=rag_context
        )
    
    # Log interaction for personalization
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        response.get("response", "")[:100] + "...",
        response.get("service_used", "conversation_manager"),
        response.get("confidence", 0.8)
    )
    
    # Enhance response with additional metadata
    enhanced_response = {
        **response,
        "user_id": query.user_id,
        "personalized": True,
        "timestamp": datetime.utcnow().isoformat(),
        "flow_type": "progressive_consultation",
        "context_provided": {
            "user_context_length": len(user_context),
            "rag_context_length": len(rag_context),
            "context_passing_status": "enabled"
        }
    }
    
    return enhanced_response

@app.post("/api/chat/continue")
async def continue_session(continuation: SessionContinuation) -> Dict[str, Any]:
    """Continue an existing conversation session"""
    # Extract user_id from session (simplified approach)
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

@app.get("/api/chat/session/{session_id}/history")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """Get complete session conversation history"""
    return await conversation_manager.get_session_history(session_id)

@app.post("/api/expert-council/direct")
async def direct_expert_council(query: HealthQuery) -> Dict[str, Any]:
    """Direct Expert Council access with structured output"""
    return await _handle_direct_expert_council(query)

async def _handle_direct_expert_council(query: HealthQuery) -> Dict[str, Any]:
    """
    ENHANCED: Handle direct Expert Council with structured output
    """
    
    # Get user context and RAG context
    user_context = await personalization_manager.get_user_context(query.user_id)
    rag_context = ""
    knowledge_chunks = []
    
    # Get medical knowledge if query seems medical
    if await _is_medical_query(query.query):
        rag_context = await rag_manager.get_context_for_query(query.query)
        knowledge_chunks = await rag_manager.semantic_search(query.query)
    
    # Run Enhanced Expert Council
    council_result = await expert_council.run_expert_council(
        query=query.query,
        user_context=user_context,
        rag_context=rag_context
    )
    
    # Enhanced: Handle Expert Council failure with detailed error info
    if not council_result.get("success"):
        error_response = {
            "error": "Expert Council consultation failed",
            "error_type": council_result.get("error_type", "expert_council_failure"),
            "failed_step": council_result.get("failed_step", "unknown_step"),
            "error_message": council_result.get("error_message", "Expert Council encountered an error"),
            "error_details": council_result.get("error_details", "No additional details available"),
            "suggestion": council_result.get("suggestion", "Please try again or start with basic consultation"),
            "user_response": council_result.get("user_response", "I'm experiencing technical difficulties with our Expert Council. Please try again in a moment."),
            "service_used": "expert_council_error",
            "confidence": 0.1,
            "user_id": query.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "debug_info": {
                "original_error": council_result.get("error", "Unknown"),
                "council_success": council_result.get("success", False),
                "available_keys": list(council_result.keys())
            }
        }
        
        # Log detailed error for debugging
        print(f"❌ Expert Council Error: {error_response['error_type']} at {error_response['failed_step']}")
        print(f"🔍 Council result keys: {list(council_result.keys())}")
        
        return error_response
    
    # ENHANCED: Extract structured data from Expert Council
    confidence = council_result.get("confidence", 0.7)
    sources = [chunk.get('title', 'Medical Knowledge') for chunk in knowledge_chunks]
    structured_analysis = council_result.get("structured_analysis", {})
    interactive_components = council_result.get("interactive_components", {})
    
    # Create enhanced response format
    final_response = f"""**🏥 AURA Expert Council Analysis**

{council_result['user_response']}

**📚 Evidence Base:** {len(sources)} medical sources consulted
**🤝 Expert Consensus:** {confidence:.0%} confidence
**👤 Personalized for:** {user_context.split(':')[0] if ':' in user_context else 'Your profile'}

---
*This analysis was conducted by AURA's Expert Council using MedAgent-Pro methodology with multiple AI specialists.*

⚠️ **Important:** This analysis is for informational purposes. Please consult your healthcare provider for personalized medical advice."""
    
    # Log interaction
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        "Expert Council consultation completed",
        "expert_council_direct_structured",
        confidence
    )
    
    # ENHANCED: Return response with structured data
    return {
        "response": final_response,
        "structured_analysis": structured_analysis,  # NEW: Full structured medical data
        "interactive_components": interactive_components,  # NEW: UI component data
        "service_used": "expert_council_direct_structured",
        "confidence": confidence,
        "query_complexity": "expert_council",
        "user_id": query.user_id,
        "knowledge_used": True,
        "sources": sources,
        "personalized": True,
        "expert_council_session": {
            "session_id": council_result.get("session_id"),
            "experts_consulted": council_result.get("metadata", {}).get("experts_consulted", []),
            "evidence_sources": council_result.get("metadata", {}).get("evidence_sources", []),
            "workflow": council_result.get("metadata", {}).get("workflow", "medagent_pro_structured_v3")
        },
        "reasoning_trace": council_result.get('reasoning_trace', {}),
        "flow_type": "direct_expert_council_structured",
        "timestamp": datetime.utcnow().isoformat(),
        "enhancements": {
            "structured_output": bool(structured_analysis),
            "interactive_components": bool(interactive_components),
            "fail_fast_enabled": True
        }
    }

@app.get("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 3):
    """Direct knowledge search endpoint for testing"""
    results = await rag_manager.semantic_search(query, max_results=limit)
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

@app.get("/api/services/status")
async def services_status():
    """Check status of all microservices"""
    rag_health = await rag_manager.health_check()
    conversation_health = await conversation_manager.health_check()
    
    return {
        "orchestrator": "online",
        "rag_system": rag_health["status"],
        "personalization": "online",
        "conversation_manager": conversation_health["status"],
        "expert_council": "ready_structured",
        "ecg_interpreter": "online", 
        "radiology_vqa": "online",
        "mental_wellness": "online",
        "ai_server": "online",
        "knowledge_entries": rag_health.get("knowledge_entries", 0),
        "enhanced_features": ["structured_analysis", "interactive_components", "fail_fast"]
    }

@app.get("/api/conversation/debug/{session_id}")
async def debug_conversation(session_id: str):
    """Debug endpoint for conversation state"""
    return await conversation_manager.get_session_history(session_id)

# Enhanced debugging endpoints
@app.get("/api/expert-council/debug")
async def debug_expert_council(query: str, user_id: str = "debug_user"):
    """Debug endpoint for Expert Council testing with structured output"""
    query_obj = HealthQuery(query=query, user_id=user_id, force_expert_council=True)
    result = await _handle_direct_expert_council(query_obj)
    
    # Return debug info
    return {
        "debug_info": {
            "structured_analysis_present": "structured_analysis" in result,
            "interactive_components_present": "interactive_components" in result,
            "expert_council_session": result.get("expert_council_session", {}),
            "enhancements": result.get("enhancements", {})
        },
        "full_result": result
    }

@app.get("/api/expert-council/test-structure")
async def test_structure():
    """Test endpoint to verify structured output capability"""
    test_query = HealthQuery(
        query="Test query for structured output",
        user_id="test_structure_user",
        force_expert_council=True
    )
    
    result = await _handle_direct_expert_council(test_query)
    
    return {
        "test_result": "structured_output",
        "has_structured_analysis": "structured_analysis" in result and result["structured_analysis"] is not None,
        "has_interactive_components": "interactive_components" in result and result["interactive_components"] is not None,
        "structured_analysis_keys": list(result.get("structured_analysis", {}).keys()) if result.get("structured_analysis") else [],
        "interactive_components_keys": list(result.get("interactive_components", {}).keys()) if result.get("interactive_components") else [],
        "service_version": "3.2.0_enhanced"
    }

# Utility functions
async def _is_medical_query(query: str) -> bool:
    """Simple medical query detection"""
    medical_keywords = [
        "pain", "symptom", "diagnosis", "treatment", "condition", "arrhythmia",
        "heart", "cardiac", "chest", "shortness", "breath", "ECG",
        "blood pressure", "medication", "side effects", "disease", "health",
        "headache", "fever", "cough", "nausea", "dizziness", "fatigue"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in medical_keywords)

async def _assess_emergency_signals(query: str) -> bool:
    """Detect emergency signals that should skip progressive consultation"""
    emergency_keywords = [
        "emergency", "urgent", "chest pain", "difficulty breathing", 
        "severe pain", "heart attack", "stroke", "unconscious", "bleeding"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in emergency_keywords)

# Development and testing endpoints
@app.post("/api/test/conversation-flow")
async def test_conversation_flow(query: str, user_id: str = "test_user"):
    """Test endpoint for conversation flow development"""
    test_query = HealthQuery(query=query, user_id=user_id)
    return await progressive_chat_endpoint(test_query)

@app.get("/api/system/conversation-stats")
async def conversation_stats():
    """System statistics for conversation management"""
    try:
        # Get basic stats from Firebase
        from firebase_admin import db
        conversations_ref = db.reference('conversations')
        all_sessions = conversations_ref.get()
        
        if not all_sessions:
            return {"total_sessions": 0, "active_sessions": 0}
        
        stats = {
            "total_sessions": len(all_sessions),
            "active_sessions": len([s for s in all_sessions.values() 
                                  if s.get("state") not in ["completed"]]),
            "states_distribution": {},
            "average_turns": 0
        }
        
        # Calculate state distribution
        for session in all_sessions.values():
            state = session.get("state", "unknown")
            stats["states_distribution"][state] = stats["states_distribution"].get(state, 0) + 1
        
        # Calculate average turns
        total_turns = sum(len(s.get("messages", [])) for s in all_sessions.values())
        stats["average_turns"] = total_turns / len(all_sessions) if all_sessions else 0
        
        return stats
        
    except Exception as e:
        return {"error": str(e), "total_sessions": 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)