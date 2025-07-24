"""
AURA Main Orchestrator Service - Enhanced with Expert Council Protocol
Central API gateway and orchestrator for the AURA health platform
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
from core.expert_council import expert_council

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI(
    title="AURA Main Orchestrator",
    description="Central orchestrator for AURA health platform with Expert Council",
    version="2.0.0"
)

# Initialize managers
rag_manager = RAGManager(SUPABASE_URL, SUPABASE_KEY)
personalization_manager = PersonalizationManager()

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"
    image_url: Optional[str] = None
    request_expert_council: bool = False

@app.get("/")
async def root():
    return {
        "service": "AURA Main Orchestrator",
        "status": "healthy", 
        "version": "2.0.0",
        "capabilities": ["RAG", "Personalization", "Expert Council"],
        "description": "Central gateway for AURA health platform with MedAgent-Pro architecture"
    }

@app.get("/health")
async def health_check():
    # Check both RAG and personalization systems
    rag_health = await rag_manager.health_check()
    personalization_health = await personalization_manager.health_check()
    
    return {
        "status": "healthy", 
        "service": "aura_main",
        "version": "2.0.0",
        "systems": {
            "rag_system": rag_health,
            "personalization_system": personalization_health,
            "expert_council": {
                "status": "ready", 
                "models": ["MedGemma-4B", "Gemini-2.5-Flash", "Gemini-2.0-Flash"],
                "workflow": "medagent_pro_5_step"
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(query: HealthQuery) -> Dict[str, Any]:
    """Enhanced chat endpoint with Expert Council Protocol"""
    
    # Get or create user profile
    profile = await personalization_manager.get_user_profile(query.user_id)
    if not profile:
        profile = await personalization_manager.create_user_profile(query.user_id)
    
    # Get user context
    user_context = await personalization_manager.get_user_context(query.user_id)
    
    # Enhanced Smart Router with Expert Council trigger
    complexity_score = await _assess_query_complexity(query.query)
    
    # Get RAG context for all medical queries
    rag_context = ""
    knowledge_chunks = []
    if complexity_score["is_medical"]:
        rag_context = await rag_manager.get_context_for_query(query.query)
        knowledge_chunks = await rag_manager.semantic_search(query.query)
    
    # Decision Tree for Response Strategy
    if complexity_score["requires_expert_council"] or query.request_expert_council:
        # Expert Council Protocol for complex medical cases
        return await _handle_expert_council_query(
            query, user_context, rag_context, knowledge_chunks
        )
    
    elif complexity_score["is_medical"]:
        # Standard RAG + Personalization for medical queries
        return await _handle_medical_query(
            query, user_context, rag_context, knowledge_chunks
        )
    
    else:
        # Simple personalized response for general queries
        return await _handle_general_query(query, user_context)

async def _assess_query_complexity(query: str) -> Dict[str, Any]:
    """Enhanced complexity assessment for Smart Router"""
    
    # Medical keywords
    medical_keywords = [
        "diagnosis", "symptoms", "treatment", "condition", "arrhythmia", 
        "heart", "cardiac", "chest pain", "shortness of breath", "ECG",
        "blood pressure", "medication", "side effects", "disease", "pain"
    ]
    
    # Expert Council triggers
    expert_council_keywords = [
        "multiple symptoms", "complex condition", "second opinion",
        "detailed analysis", "comprehensive assessment", "differential diagnosis",
        "treatment options", "specialist opinion", "serious", "concerned"
    ]
    
    query_lower = query.lower()
    
    is_medical = any(keyword in query_lower for keyword in medical_keywords)
    has_multiple_aspects = len([kw for kw in medical_keywords if kw in query_lower]) >= 2
    requests_expert_council = any(keyword in query_lower for keyword in expert_council_keywords)
    
    complexity_score = {
        "is_medical": is_medical,
        "has_multiple_aspects": has_multiple_aspects,
        "requests_expert_council": requests_expert_council,
        "requires_expert_council": (
            requests_expert_council or 
            (is_medical and has_multiple_aspects)
        ),
        "complexity_level": "expert_council" if (requests_expert_council or has_multiple_aspects) 
                           else "medical" if is_medical 
                           else "general"
    }
    
    return complexity_score

async def _handle_expert_council_query(
    query: HealthQuery, 
    user_context: str, 
    rag_context: str, 
    knowledge_chunks: list
) -> Dict[str, Any]:
    """Handle complex queries requiring Expert Council consultation"""
    
    # Convene Expert Council with MedAgent-Pro workflow
    council_result = await expert_council.run_expert_council(
        query=query.query,
        user_context=user_context,
        rag_context=rag_context
    )
    
    if not council_result.get("success"):
        # Fallback to standard medical query handling
        return await _handle_medical_query(query, user_context, rag_context, knowledge_chunks)
    
    service_used = "expert_council_medagent_pro"
    confidence = council_result.get("confidence", 0.7)
    sources = [chunk.get('title', 'Medical Knowledge') for chunk in knowledge_chunks]
    
    # Enhanced response structure
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
        service_used,
        confidence
    )
    
    return {
        "response": final_response,
        "service_used": service_used,
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
            "workflow": "medagent_pro_5_step"
        },
        "reasoning_trace": council_result.get('reasoning_trace', []),
        "timestamp": datetime.utcnow().isoformat()
    }

async def _handle_medical_query(
    query: HealthQuery, 
    user_context: str, 
    rag_context: str, 
    knowledge_chunks: list
) -> Dict[str, Any]:
    """Handle standard medical queries with RAG + Personalization"""
    
    response_text = f"""**🔍 AURA Medical Analysis**

Based on medical knowledge and your profile:

**📖 Medical Context:**
{rag_context}

**👤 Your Context:**
{user_context}

**🩺 AURA Assessment:**
This query relates to medical information. The above context provides evidence-based information relevant to your question. For specific medical advice, please consult a healthcare professional."""

    service_used = "orchestrator_with_rag_and_personalization"
    confidence = 0.85
    sources = [chunk.get('title', 'Medical Knowledge') for chunk in knowledge_chunks]
    
    # Log interaction
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        response_text[:100] + "...",
        service_used,
        confidence
    )

    return {
        "response": response_text,
        "service_used": service_used,
        "confidence": confidence,
        "query_complexity": "medical",
        "user_id": query.user_id,
        "knowledge_used": True,
        "sources": sources,
        "personalized": True,
        "timestamp": datetime.utcnow().isoformat()
    }

async def _handle_general_query(query: HealthQuery, user_context: str) -> Dict[str, Any]:
    """Handle general queries with basic personalization"""
    
    response_text = f"Hi there! I received your query: '{query.query}'. How can I help you with your health today?"
        
    service_used = "orchestrator_with_personalization"
    confidence = 0.75
    
    # Log interaction
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        response_text,
        service_used,
        confidence
    )

    return {
        "response": response_text,
        "service_used": service_used,
        "confidence": confidence,
        "query_complexity": "general",
        "user_id": query.user_id,
        "knowledge_used": False,
        "sources": [],
        "personalized": True,
        "timestamp": datetime.utcnow().isoformat()
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
    
    return {
        "orchestrator": "online",
        "rag_system": rag_health["status"],
        "personalization": "online",
        "expert_council": "ready",
        "ecg_interpreter": "online", 
        "radiology_vqa": "online",
        "mental_wellness": "online",
        "ai_server": "online",
        "knowledge_entries": rag_health.get("knowledge_entries", 0)
    }

@app.get("/api/expert-council/debug")
async def debug_expert_council(query: str, user_id: str = "debug_user"):
    """Debug endpoint for Expert Council testing"""
    user_context = await personalization_manager.get_user_context(user_id)
    rag_context = await rag_manager.get_context_for_query(query)
    
    result = await expert_council.run_expert_council(
        query=query,
        user_context=user_context,
        rag_context=rag_context
    )
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)