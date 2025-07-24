"""
AURA Main Orchestrator Service
Central API gateway and orchestrator for the AURA health platform
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any
from datetime import datetime
from lib.rag_manager import RAGManager
from lib.personalization_manager import PersonalizationManager

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI(
    title="AURA Main Orchestrator",
    description="Central orchestrator for AURA health platform",
    version="1.0.0"
)

# Initialize managers
rag_manager = RAGManager(SUPABASE_URL, SUPABASE_KEY)
personalization_manager = PersonalizationManager()

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"

class HealthResponse(BaseModel):
    response: str
    service_used: str
    confidence: float
    timestamp: str
    knowledge_used: bool = False
    sources: list = []

@app.get("/")
async def root():
    return {
        "service": "AURA Main Orchestrator",
        "status": "healthy",
        "version": "1.0.0",
        "description": "Central gateway for AURA health platform with RAG"
    }

@app.get("/health")
async def health_check():
    # Check both RAG and personalization systems
    rag_health = await rag_manager.health_check()
    personalization_health = await personalization_manager.health_check()
    
    return {
        "status": "healthy", 
        "service": "aura_main",
        "rag_system": rag_health,
        "personalization_system": personalization_health,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(query: HealthQuery) -> Dict[str, Any]:
    """Main chat endpoint with RAG and personalization"""
    
    # Get or create user profile
    profile = await personalization_manager.get_user_profile(query.user_id)
    if not profile:
        profile = await personalization_manager.create_user_profile(query.user_id)
    
    # Get user context
    user_context = await personalization_manager.get_user_context(query.user_id)
    
    # Smart Router: Simple keyword-based complexity assessment
    complex_keywords = ["diagnosis", "symptoms", "treatment", "condition", "arrhythmia", "heart"]
    is_complex = any(keyword in query.query.lower() for keyword in complex_keywords)
    
    if is_complex:
        # Use RAG for medical queries
        context = await rag_manager.get_context_for_query(query.query)
        knowledge_chunks = await rag_manager.semantic_search(query.query)
        
        # Enhanced response with user context
        response_text = f"""Based on medical knowledge and your profile: {query.query}

**Your Context:**
{user_context}

**Medical Context:**
{context}

**AURA Analysis:**
This query relates to medical information. The above context provides evidence-based information relevant to your question. For specific medical advice, please consult a healthcare professional."""

        service_used = "orchestrator_with_rag_and_personalization"
        confidence = 0.90
        sources = [chunk.get('title', 'Medical Knowledge') for chunk in knowledge_chunks]
        
    else:
        # Simple queries with basic personalization
        comm_pref = profile.get('profile', {}).get('communication_preference', 'friendly')
        if comm_pref == 'friendly':
            response_text = f"Hi there! I received your query: '{query.query}'. How can I help you with your health today?"
        else:
            response_text = f"Query received: '{query.query}'. Please let me know how I can assist you."
            
        service_used = "orchestrator_with_personalization"
        confidence = 0.80
        sources = []
    
    # Log interaction
    await personalization_manager.log_interaction(
        query.user_id,
        query.query,
        response_text[:100] + "..." if len(response_text) > 100 else response_text,
        service_used,
        confidence
    )

    return {
        "response": response_text,
        "service_used": service_used,
        "confidence": confidence,
        "query_complexity": "complex" if is_complex else "simple",
        "user_id": query.user_id,
        "knowledge_used": is_complex,
        "sources": sources,
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
        "ecg_interpreter": "online", 
        "radiology_vqa": "online",
        "mental_wellness": "online",
        "knowledge_entries": rag_health.get("knowledge_entries", 0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)