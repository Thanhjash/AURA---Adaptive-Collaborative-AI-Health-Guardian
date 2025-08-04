# File: services/aura_main/lib/conversation_manager.py (VERSION V12.1 - BULLETPROOF)

import os
import json
import asyncio
import traceback
import uuid
import re
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import httpx
from datetime import datetime, timezone

# Google Gemini SDK for reliable API calls
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. pip install google-generativeai")

# Import architecture components
from models import HealthQuery
from .rag_manager import RAGManager
from .personalization_manager import PersonalizationManager
from core.expert_council import expert_council

class ConversationManager:
    """
    AURA's Central Brain V12.1 - Bulletproof with proper Gemini integration
    """
    def __init__(self):
        # Initialize sub-systems
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.rag_manager = RAGManager(supabase_url, supabase_key)
        self.personalization_manager = PersonalizationManager()
        self.expert_council = expert_council
        
        # Initialize Gemini client properly
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_client = None
        
        if self.gemini_api_key and GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash-latest')
                print("✅ [CONVO-MGR] Gemini client initialized successfully.")
            except Exception as e:
                print(f"⚠️ [CONVO-MGR] Gemini configuration failed: {e}")
                self.gemini_client = None
        else:
            print("⚠️ [CONVO-MGR] Gemini unavailable. Using fallback logic.")
        
        # Simple in-memory cache
        self._context_cache = {} 
        self._triage_cache = {}

    @asynccontextmanager
    async def lifespan(self, app):
        print("ConversationManager V12.1 starting up...")
        yield
        print("ConversationManager V12.1 shutting down...")

    # --- MAIN STREAM FUNCTION ---
    async def stream_turn(self, query: HealthQuery) -> AsyncGenerator[str, None]:
        """Central brain manages complete conversation turn."""
        session_state = None
        try:
            # 1. Load or Create Session State
            yield self._sse_event("progress", {"step": "session", "status": "Loading session..."})
            session_state = await self._load_or_create_session_state(query)
            
            # 2. Load Context with Cache
            yield self._sse_event("progress", {"step": "context", "status": "Loading context..."})
            user_context, rag_context = await self._load_contexts_cached(query)
            
            # 3. Triage using Gemini or fallback
            yield self._sse_event("progress", {"step": "triage", "status": "Analyzing query..."})
            routing = await self._decide_routing_strategy(query, session_state, user_context, rag_context)

            # 4. Execute strategy
            strategy = routing.get("strategy", "clarify")
            yield self._sse_event("progress", {"step": "action", "status": f"Strategy: {strategy}"})

            if strategy == "clarify":
                async for chunk in self._stream_clarifying_question(query, session_state, user_context, rag_context):
                    yield chunk
            elif strategy == "guide":
                async for chunk in self._stream_direct_guidance(query, session_state, user_context, rag_context):
                    yield chunk
            elif strategy == "suggest_council":
                async for chunk in self._stream_expert_council_suggestion(query):
                    yield chunk
            elif strategy == "run_council":
                async for chunk in self._stream_expert_council_once(query, user_context, rag_context):
                    yield chunk
            else:
                yield self._sse_event("text_token", {"token": "I'm not sure how to proceed. Can you rephrase your question?"})

        except Exception as e:
            traceback.print_exc()
            yield self._sse_event("error", {"message": f"Critical error: {str(e)}"})
        finally:
            if session_state:
                await self._save_session_state(session_state, query)
            yield self._sse_event("stream_end", {"session_id": query.session_id})

    # --- STATE MANAGEMENT ---
    async def _load_or_create_session_state(self, query: HealthQuery) -> dict:
        if not query.session_id:
            query.session_id = f"session_{uuid.uuid4()}"
        
        try:
            doc_ref = self.personalization_manager.db.collection('sessions').document(query.session_id)
            doc = await doc_ref.get()

            if doc.exists:
                print(f"SESSION FOUND: {query.session_id}")
                return doc.to_dict()
            else:
                print(f"SESSION CREATED: {query.session_id}")
                new_state = {
                    "session_id": query.session_id,
                    "user_id": query.user_id,
                    "conversation_state": "INITIAL_GREETING",
                    "turn_count": 0,
                    "symptom_profile": {},
                    "message_history": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                await doc_ref.set(new_state)
                return new_state
        except Exception as e:
            print(f"Session state error: {e}")
            return {
                "session_id": query.session_id,
                "user_id": query.user_id,
                "conversation_state": "INITIAL_GREETING",
                "turn_count": 0,
                "symptom_profile": {},
                "message_history": []
            }

    async def _save_session_state(self, session_state: dict, query: HealthQuery):
        try:
            session_id = session_state["session_id"]
            doc_ref = self.personalization_manager.db.collection('sessions').document(session_id)
            
            session_state["last_updated"] = datetime.now(timezone.utc).isoformat()
            session_state["turn_count"] = session_state.get("turn_count", 0) + 1
            
            session_state["message_history"].append({
                "query": query.query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "turn": session_state["turn_count"]
            })
            
            await doc_ref.update(session_state)
            print(f"SESSION SAVED: {session_id}")
        except Exception as e:
            print(f"Session save error: {e}")

    # --- CONTEXT LOADING ---
    async def _load_contexts_cached(self, query: HealthQuery) -> tuple:
        cache_key = f"{query.user_id}:{hash(query.query) % 10000}"
        if cache_key in self._context_cache:
            print("CONTEXT CACHE HIT!")
            return self._context_cache[cache_key]

        print("CONTEXT CACHE MISS - Fetching...")
        try:
            user_task = asyncio.create_task(self.personalization_manager.get_user_context(query.user_id))
            rag_task = asyncio.create_task(self.rag_manager.get_context_for_query(query.query))
            contexts = await asyncio.gather(user_task, rag_task)
            self._context_cache[cache_key] = contexts
            return contexts
        except Exception as e:
            print(f"Context loading error: {e}")
            return ("No user context available.", "No relevant medical knowledge found.")

    # --- ROUTING STRATEGY ---
    async def _decide_routing_strategy(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> dict:
        """
        Routing logic V12.2 - Fixed to always suggest Expert Council for complex cases.
        """
        try:
            triage_result = await self._run_triage_with_gemini(query, user_context, rag_context)
            
            enable_council = os.getenv("ENABLE_EXPERT_COUNCIL", "false").lower() == "true"
            category = triage_result.get("category", "medical_query_low_priority")
            
            # PRIORITY 1: Force council if requested and enabled
            if query.force_expert_council and enable_council:
                print("Routing: Council forced by user and enabled.")
                return {"strategy": "run_council"}
            
            # PRIORITY 2: Handle complex/emergency cases
            if category in ["medical_emergency", "medical_query_high_priority"]:
                # Emergency: run council immediately if enabled
                if category == "medical_emergency" and enable_council:
                    print("Routing: Emergency detected, running council.")
                    return {"strategy": "run_council"}
                
                # High priority: ALWAYS suggest council (regardless of enabled status)
                print("Routing: High priority query, suggesting council.")
                return {"strategy": "suggest_council"}
            
            # PRIORITY 3: Simple cases
            if category in ["simple_chitchat", "general_health"]:
                print("Routing: Simple query, providing direct guidance.")
                return {"strategy": "guide"}
            
            # PRIORITY 4: Medical interview logic (for low priority)
            symptom_completeness = len(session_state.get("symptom_profile", {}))
            if symptom_completeness < 3:
                print("Routing: Low priority with incomplete profile, clarifying.")
                return {"strategy": "clarify"}
            else:
                print("Routing: Low priority with complete profile, providing guidance.")
                return {"strategy": "guide"}
                
        except Exception as e:
            print(f"CRITICAL Routing decision error: {e}")
            return {"strategy": "guide"}

    async def _run_triage_with_gemini(self, query: HealthQuery, user_context: str, rag_context: str) -> dict:
        # Check cache first
        cache_key = f"triage:{query.session_id}:{hash(query.query) % 1000}"
        cached_entry = self._triage_cache.get(cache_key)
        
        current_time = asyncio.get_running_loop().time()
        if cached_entry and (current_time - cached_entry['timestamp']) < 90:
            print("TRIAGE CACHE HIT!")
            return cached_entry['data']

        # Use Gemini if available, otherwise smart fallback
        if not self.gemini_client:
            print("TRIAGE: Using smart fallback")
            return self._smart_fallback_triage(query, user_context)

        try:
            triage_prompt = f"""Analyze this health query and categorize it:

User Query: {query.query}
User Context: {user_context}
Medical Knowledge: {rag_context[:500]}

Categorize as one of:
- medical_emergency (urgent, dangerous symptoms)
- medical_query_high_priority (complex, needs expert analysis)  
- medical_query_low_priority (routine question)
- simple_chitchat (greeting, general wellness)

Respond in JSON format only:
{{"category": "category_name", "confidence": 0.8, "reasoning": "brief explanation"}}"""

            response_text = await self._call_gemini_api(triage_prompt)
            
            # Parse JSON response
            try:
                triage_result = json.loads(response_text.strip())
            except json.JSONDecodeError:
                print("Failed to parse triage JSON, using fallback")
                triage_result = self._smart_fallback_triage(query, user_context)
            
            self._triage_cache[cache_key] = {'timestamp': current_time, 'data': triage_result}
            return triage_result
            
        except Exception as e:
            print(f"Triage with Gemini failed: {e}")
            return self._smart_fallback_triage(query, user_context)

    def _smart_fallback_triage(self, query: HealthQuery, user_context: str) -> dict:
        """Smart fallback triage using keyword analysis."""
        query_lower = query.query.lower()
        
        # Emergency keywords
        emergency_words = ['chest pain', 'can\'t breathe', 'severe pain', 'bleeding heavily', 'unconscious']
        if any(word in query_lower for word in emergency_words):
            return {"category": "medical_emergency", "confidence": 0.9, "reasoning": "Emergency keywords detected"}
        
        # High priority medical keywords
        high_priority_words = ['pain', 'headache', 'fever', 'nausea', 'dizzy', 'rash', 'swelling']
        if any(word in query_lower for word in high_priority_words):
            return {"category": "medical_query_high_priority", "confidence": 0.7, "reasoning": "Medical symptoms detected"}
        
        # Chitchat keywords
        chitchat_words = ['hello', 'hi', 'how are you', 'good morning', 'thanks', 'thank you']
        if any(word in query_lower for word in chitchat_words):
            return {"category": "simple_chitchat", "confidence": 0.9, "reasoning": "Greeting detected"}
        
        # Default
        return {"category": "medical_query_low_priority", "confidence": 0.5, "reasoning": "Default classification"}

    # --- STREAMING GENERATORS ---
    async def _stream_clarifying_question(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        symptom_profile = session_state.get("symptom_profile", {})
        
        prompt = f"""You are AURA, an empathetic AI health assistant. Generate ONE caring, specific follow-up question to better understand the user's health concern.

User Query: {query.query}
User Context: {user_context}
Known Symptoms: {symptom_profile}

Ask like a caring doctor. Be specific but gentle. Keep it to one question only."""

        try:
            if not self.gemini_client:
                raise ValueError("Gemini client not available")

            response_text = await self._call_gemini_api(prompt)
            async for chunk in self._stream_text(response_text):
                yield chunk
        except Exception as e:
            print(f"Clarification question failed: {e}")
            fallback = "I'd like to help you better. Could you tell me more about what you're experiencing?"
            async for chunk in self._stream_text(fallback):
                yield chunk

    async def _stream_direct_guidance(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        symptom_profile = session_state.get("symptom_profile", {})
        
        # For medical guidance, use MedGemma
        medgemma_prompt = f"""<|im_start|>system
You are AURA, a knowledgeable AI health assistant. Provide helpful, evidence-based medical guidance while always recommending professional consultation for serious concerns.
<|im_end|>
<|im_start|>user
Query: {query.query}
User Context: {user_context}
Symptoms: {symptom_profile}
Medical Knowledge: {rag_context}

Please provide clear, actionable guidance for this health concern.
<|im_end|>
<|im_start|>assistant"""

        try:
            payload = {
                "prompt": medgemma_prompt,
                "max_tokens": 300,
                "temperature": 0.7,
                "stop": ["<|im_end|>"]
            }
            result = await self._call_ai_server("/generate", payload)
            response_text = result.get("response", "I'm having trouble accessing my knowledge right now.")
            
            async for chunk in self._stream_text(response_text):
                yield chunk
        except Exception as e:
            print(f"MedGemma call failed: {e}")
            fallback = "I'm having trouble accessing my medical knowledge. Please consult with a healthcare professional for personalized advice."
            async for chunk in self._stream_text(fallback):
                yield chunk

    async def _stream_expert_council_suggestion(self, query: HealthQuery) -> AsyncGenerator[str, None]:
        suggestion_text = """Your query involves complex medical considerations. I can convene an AI Expert Council for comprehensive multi-perspective analysis. This takes 10-15 seconds but provides more thorough insights. Would you like me to proceed?"""
        
        async for chunk in self._stream_text(suggestion_text):
            yield chunk

    async def _stream_expert_council_once(self, query: HealthQuery, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        try:
            yield self._sse_event("council_started", {"message": "Convening Expert Council..."})
            
            async for update in self.expert_council.run_expert_council_with_progress(query.query, user_context, rag_context):
                if update["type"] == "progress":
                    yield self._sse_event("council_step", update)
                elif update["type"] == "result":
                    council_result = update["data"]
                    response_text = council_result.get("user_response", "Expert Council analysis complete.")
                    async for chunk in self._stream_text(response_text):
                        yield chunk
                    break
        except Exception as e:
            print(f"Expert Council failed: {e}")
            fallback = "The Expert Council is temporarily unavailable. Let me provide standard guidance."
            async for chunk in self._stream_text(fallback):
                yield chunk

    async def _stream_text(self, text: str) -> AsyncGenerator[str, None]:
        words = text.split()
        for i, word in enumerate(words):
            yield self._sse_event("text_token", {"token": word + " "})
            delay = 0.03 if i % 5 == 0 else 0.05
            await asyncio.sleep(delay)

    # --- API CALLS ---
    async def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using official SDK."""
        if not self.gemini_client:
            raise ConnectionError("Gemini client not initialized")
            
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API failed: {e}")
            raise e

    async def _call_ai_server(self, endpoint: str, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Call ai_server with retry logic."""
        last_exception = None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(f"http://ai_server:9000{endpoint}", json=payload)
                    response.raise_for_status()
                    return response.json()
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    last_exception = e
                    print(f"AI server call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"AI server failed after {max_retries} attempts: {last_exception}")

    def _sse_event(self, event_name: str, data: Dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, default=str)}\n\n"

    async def health_check(self):
        return {
            "status": "healthy",
            "architecture": "Conversation-Centric V12.1 Bulletproof",
            "features": {
                "session_persistence": True,
                "context_caching": True,
                "gemini_integration": bool(self.gemini_client),
                "expert_council_enabled": os.getenv("ENABLE_EXPERT_COUNCIL", "false").lower() == "true"
            }
        }