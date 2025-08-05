# services/aura_main/lib/conversation_manager.py - COMPLETE VERSION WITH STATE MANAGEMENT

import json
import re
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, AsyncGenerator, Optional
from contextlib import asynccontextmanager
import google.generativeai as genai
import os
from google.cloud import firestore
from models import HealthQuery
from lib.personalization_manager import PersonalizationManager
from lib.rag_manager import RAGManager
from core.expert_council import expert_council

class ConversationManager:
    def __init__(self):
        self.personalization_manager = PersonalizationManager()
        self.rag_manager = RAGManager(
            os.getenv("SUPABASE_URL"), 
            os.getenv("SUPABASE_KEY")
        )
        
        # Initialize Gemini
        self.gemini_client = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel('gemini-2.0-flash')
        
        self._context_cache = {}
        self._captured_text = ""  # For capturing streamed text
        
    @asynccontextmanager
    async def lifespan(self, app):
        """FastAPI lifespan management"""
        print("🚀 ConversationManager initialized")
        yield
        print("🔄 ConversationManager shutting down")

    # === CORE STREAMING METHOD ===
    async def stream_turn(self, query: HealthQuery) -> AsyncGenerator[str, None]:
        """Central brain manages complete conversation turn with memory"""
        session_state = None
        assistant_response_content = ""
        
        try:
            # 1. Load or Create Session State
            yield self._sse_event("progress", {"step": "session", "status": "Loading session..."})
            session_state = await self._load_or_create_session_state(query)
            
            # 2. 🧠 LISTEN & UPDATE SYMPTOM PROFILE (New!)
            yield self._sse_event("progress", {"step": "understanding", "status": "Understanding your message..."})
            updated_profile = await self._update_symptom_profile(query, session_state)
            session_state['symptom_profile'] = updated_profile
            
            # 3. 🔍 PRIORITY CHECK: Force Expert Council
            if query.force_expert_council:
                yield self._sse_event("council_started", {"message": "Expert Council activated by user request"})
                
                user_context, rag_context = await self._load_contexts_cached(query)
                
                async for chunk in self._stream_expert_council_once(query, user_context, rag_context):
                    if chunk.startswith('data: {"token":'):
                        # Extract token for building full response
                        try:
                            token_data = json.loads(chunk.split('data: ')[1])
                            assistant_response_content += token_data.get('token', '')
                        except:
                            pass
                    yield chunk
                
                yield self._sse_event("stream_end", {"session_id": query.session_id, "message": "Expert Council Complete"})
                return
            
            # 4. Load Context with Cache
            yield self._sse_event("progress", {"step": "context", "status": "Loading context..."})
            user_context, rag_context = await self._load_contexts_cached(query)
            
            # 5. 🎯 INTELLIGENT ROUTING with State Awareness
            yield self._sse_event("progress", {"step": "triage", "status": "Analyzing query..."})
            routing = await self._decide_routing_strategy(query, session_state, user_context, rag_context)

            # 6. Execute strategy and capture response
            strategy = routing.get("strategy", "clarify")
            yield self._sse_event("progress", {"step": "action", "status": f"Strategy: {strategy}"})

            stream_generator = None
            if strategy == "clarify":
                stream_generator = self._stream_clarifying_question(query, session_state, user_context, rag_context)
            elif strategy == "guide":
                stream_generator = self._stream_direct_guidance(query, session_state, user_context, rag_context)
            elif strategy == "suggest_council":
                stream_generator = self._stream_expert_council_suggestion(query, session_state)
            elif strategy == "run_council":
                stream_generator = self._stream_expert_council_once(query, user_context, rag_context)
            elif strategy == "clarify_council":
                stream_generator = self._stream_clarify_council(query, session_state)
            else:
                error_msg = "I'm not sure how to proceed. Can you rephrase your question?"
                yield self._sse_event("text_token", {"token": error_msg})
                assistant_response_content = error_msg

            # Stream and capture response
            if stream_generator:
                assistant_response_content = ""
                async for chunk in stream_generator:
                    yield chunk  # Forward to client
                    
                    # Capture tokens for saving
                    if 'data: {"token":' in chunk:
                        try:
                            data_str = chunk.split('data: ')[1]
                            token_data = json.loads(data_str)
                            assistant_response_content += token_data.get('token', '')
                        except (json.JSONDecodeError, IndexError):
                            pass

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Critical error: {str(e)}"
            yield self._sse_event("error", {"message": error_msg})
            assistant_response_content = error_msg
        finally:
            if session_state:
                await self._save_session_state(session_state, query, assistant_response_content)
            yield self._sse_event("stream_end", {"session_id": query.session_id})

    # === NEW: SYMPTOM PROFILE MEMORY ===
    async def _update_symptom_profile(self, query: HealthQuery, session_state: dict) -> dict:
        """
        Use Gemini to extract symptoms from latest message and update state.
        This is AURA's "listening" step.
        """
        if not self.gemini_client:
            return session_state.get('symptom_profile', {})

        history = "\n".join([
            f"  - {msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in session_state.get('message_history', [])[-4:]
        ])
        
        prompt = f"""You are a medical information extraction system.
Analyze the 'LATEST USER MESSAGE' based on the 'CURRENT PROFILE' and recent 'CONVERSATION HISTORY'.
Extract or update any medical symptoms, duration, severity, location, or context.

CURRENT PROFILE (JSON):
{json.dumps(session_state.get('symptom_profile', {}), indent=2)}

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
"{query.query}"

TASK: Return ONLY the updated JSON object of the symptom profile.
- If a field already exists, update it with more specific information if available.
- If no new medical information is present, return the original profile JSON exactly as it was.
- Do not add fields that are not mentioned by the user.
- The JSON should be compact and on a single line.
"""

        try:
            response_text = await self._call_gemini_api(prompt)
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                updated_profile = json.loads(match.group(0))
                print(f"✅ SYMPTOM PROFILE UPDATED: {updated_profile}")
                return updated_profile
            else:
                return session_state.get('symptom_profile', {})
        except Exception as e:
            print(f"❌ Failed to update symptom profile: {e}")
            return session_state.get('symptom_profile', {})

    # === ROUTING STRATEGY WITH CORRECT PRIORITY ORDER ===
    async def _decide_routing_strategy(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> dict:
        """
        FIXED: Proper priority order - Triage BEFORE symptom completeness check
        """
        conversation_state = session_state.get("conversation_state", "ACTIVE")
        user_intent = query.query.lower().strip()

        # --- PRIORITY 1: Handle responses to Expert Council suggestions ---
        if conversation_state == "AWAITING_COUNCIL_CONFIRMATION":
            print(f"🔍 STATE CHECK: Awaiting council confirmation, user said: '{user_intent}'")
            
            if any(word in user_intent for word in ["yes", "yep", "ok", "proceed", "yess", "sure", "please"]):
                print("✅ User confirmed Expert Council")
                session_state["conversation_state"] = "ACTIVE"  # Reset state
                return {"strategy": "run_council"}
            else:
                print("❌ User declined Expert Council")
                session_state["conversation_state"] = "ACTIVE"  # Reset state
                return {"strategy": "guide"}

        # --- PRIORITY 2: TRIAGE FIRST (CRITICAL FOR SAFETY) ---
        try:
            triage_result = await self._run_triage_with_gemini(query, user_context, rag_context)
            category = triage_result.get('category', 'unknown')
            
            print(f"🎯 Triage: {category}")
            
            # Emergency cases bypass symptom completeness check
            if category in ["medical_emergency", "medical_query_high_priority"]:
                session_state["conversation_state"] = "AWAITING_COUNCIL_CONFIRMATION"
                print("🏥 High priority - suggesting Expert Council")
                return {"strategy": "suggest_council"}
                
        except Exception as e:
            print(f"Triage error: {e}")
            # Continue to other checks if triage fails

        # --- PRIORITY 3: Check symptom completeness (ONLY for non-emergencies) ---
        symptom_profile = session_state.get("symptom_profile", {})
        symptom_completeness = len(symptom_profile)
        print(f"🩺 Symptom completeness: {symptom_completeness} fields - {symptom_profile}")
        
        if symptom_completeness < 3:
            print("❓ Incomplete profile - asking clarification")
            return {"strategy": "clarify"}

        # --- PRIORITY 4: Provide guidance if sufficient information ---
        print("📋 Profile complete - providing guidance")
        return {"strategy": "guide"}

    # === STREAMING METHODS ===
    async def _stream_expert_council_suggestion(self, query: HealthQuery, session_state: dict) -> AsyncGenerator[str, None]:
        """Stream Expert Council suggestion and set state"""
        session_state["conversation_state"] = "AWAITING_COUNCIL_CONFIRMATION"
        print("🔄 STATE SET: AWAITING_COUNCIL_CONFIRMATION")
        
        suggestion_text = """Your query involves complex medical considerations. I can convene an AI Expert Council for comprehensive multi-perspective analysis. This takes 10-15 seconds but provides more thorough insights. Would you like me to proceed with the Expert Council?"""
        
        async for chunk in self._stream_text(suggestion_text):
            yield chunk

    async def _stream_clarify_council(self, query: HealthQuery, session_state: dict) -> AsyncGenerator[str, None]:
        """Handle unclear response to Expert Council suggestion"""
        clarification_text = """I'm not sure if you want me to proceed with the Expert Council or not. Please respond with 'yes' to activate the Expert Council, or 'no' to continue with standard guidance."""
        
        async for chunk in self._stream_text(clarification_text):
            yield chunk

    async def _stream_clarifying_question(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        """Generate contextual clarifying questions"""
        if not self.gemini_client:
            fallback = "Can you tell me more about your symptoms, including when they started and how severe they are?"
            async for chunk in self._stream_text(fallback):
                yield chunk
            return

        symptom_profile = session_state.get('symptom_profile', {})
        prompt = f"""Based on this symptom profile: {json.dumps(symptom_profile)}
Latest query: "{query.query}"

Generate ONE specific clarifying question to better understand their health concern. Ask about missing details like:
- Duration/timing
- Severity (1-10 scale)  
- Location/specific area
- Triggers or patterns
- Associated symptoms

Respond naturally as a caring health assistant."""

        try:
            response = await self._call_gemini_api(prompt)
            async for chunk in self._stream_text(response):
                yield chunk
        except Exception as e:
            print(f"Clarification error: {e}")
            fallback = "Can you provide more details about your symptoms?"
            async for chunk in self._stream_text(fallback):
                yield chunk

    async def _stream_direct_guidance(self, query: HealthQuery, session_state: dict, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        """Provide direct health guidance"""
        if not self.gemini_client:
            fallback = "Based on your symptoms, I recommend consulting with a healthcare professional for proper evaluation."
            async for chunk in self._stream_text(fallback):
                yield chunk
            return

        symptom_profile = session_state.get('symptom_profile', {})
        prompt = f"""Provide health guidance for this profile:
Symptoms: {json.dumps(symptom_profile)}
Context: {rag_context[:500]}

Give practical, helpful advice including:
1. Immediate steps they can take
2. When to seek medical care
3. General wellness recommendations
4. Important disclaimers

Be supportive and informative."""

        try:
            response = await self._call_gemini_api(prompt)
            async for chunk in self._stream_text(response):
                yield chunk
        except Exception as e:
            print(f"Guidance error: {e}")
            fallback = "I recommend consulting with a healthcare professional for personalized advice."
            async for chunk in self._stream_text(fallback):
                yield chunk

    async def _stream_expert_council_once(self, query: HealthQuery, user_context: str, rag_context: str) -> AsyncGenerator[str, None]:
        """Stream Expert Council analysis with proper async generator handling"""
        yield self._sse_event("council_started", {"message": "Convening Expert Council..."})
        
        try:
            # FIXED: Use async generator correctly
            council_result = None
            
            async for event in expert_council.run_expert_council_with_progress(query.query, user_context, rag_context):
                event_type = event.get("type")
                
                if event_type == "progress":
                    # Stream progress updates
                    yield self._sse_event("council_step", {
                        "step": event.get("step", ""),
                        "status": event.get("status", ""),
                        "description": event.get("description", "")
                    })
                elif event_type == "result":
                    # Final result
                    council_result = event.get("data")
                    break
            
            # Stream the response text
            if council_result and council_result.get('user_response'):
                response_text = council_result['user_response']
                async for chunk in self._stream_text(response_text):
                    yield chunk
            else:
                fallback_text = "Expert Council analysis completed, but no detailed response was generated."
                async for chunk in self._stream_text(fallback_text):
                    yield chunk
                    
        except Exception as e:
            print(f"Expert Council error: {e}")
            import traceback
            traceback.print_exc()
            
            error_text = "Expert Council encountered a technical issue. Providing standard medical guidance instead."
            async for chunk in self._stream_text(error_text):
                yield chunk



    async def _stream_text(self, text: str) -> AsyncGenerator[str, None]:
        """Stream text token by token"""
        words = text.split()
        for i in range(0, len(words), 3):  # Stream 3 words at a time
            chunk = " ".join(words[i:i+3])
            if i + 3 < len(words):
                chunk += " "
            yield self._sse_event("text_token", {"token": chunk})
            await asyncio.sleep(0.05)  # Small delay for streaming effect

    # === HELPER METHODS ===
    def _sse_event(self, event: str, data: dict) -> str:
        """Format Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def _load_or_create_session_state(self, query: HealthQuery) -> dict:
        """Load existing session or create new one"""
        if query.session_id:
            try:
                existing_state = await self.personalization_manager.get_session_data(query.session_id)
                if existing_state:
                    return existing_state
            except Exception as e:
                print(f"Failed to load session {query.session_id}: {e}")
        
        # Create new session
        new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        query.session_id = new_session_id
        
        return {
            "session_id": new_session_id,
            "user_id": query.user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conversation_state": "ACTIVE",
            "symptom_profile": {},
            "message_history": []
        }

    async def _load_contexts_cached(self, query: HealthQuery) -> tuple[str, str]:
        """Load user and RAG contexts with caching"""
        cache_key = f"{query.user_id}_{hash(query.query[:50])}"
        
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        try:
            user_context = await self.personalization_manager.get_user_context(query.user_id)
            rag_context = await self.rag_manager.get_context_for_query(query.query)
            
            self._context_cache[cache_key] = (user_context, rag_context)
            return (user_context, rag_context)
        except Exception as e:
            print(f"Context loading error: {e}")
            return ("", "")

    async def _save_session_state(self, session_state: dict, query: HealthQuery, assistant_response: str):
        """Save complete session state with full memory persistence"""
        try:
            # Add user message
            session_state["message_history"].append({
                "role": "user",
                "content": query.query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            # Add assistant response if available
            if assistant_response:
                session_state["message_history"].append({
                    "role": "assistant", 
                    "content": assistant_response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            
            # CRITICAL FIX: Save ALL state fields to preserve memory
            session_state["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save complete session state including symptom_profile and conversation_state
            await self.personalization_manager.save_session_data(query.session_id, {
                "session_id": session_state["session_id"],
                "user_id": session_state["user_id"],
                "created_at": session_state["created_at"],
                "last_updated": session_state["last_updated"],
                "conversation_state": session_state.get("conversation_state", "ACTIVE"),
                "symptom_profile": session_state.get("symptom_profile", {}),
                "message_history": session_state["message_history"]
            })
            
            print(f"💾 Complete session saved: {query.session_id} (symptom_profile: {len(session_state.get('symptom_profile', {}))} fields)")
            
        except Exception as e:
            print(f"Failed to save session: {e}")

    async def _run_triage_with_gemini(self, query: HealthQuery, user_context: str, rag_context: str) -> dict:
        """Run medical triage using Gemini"""
        if not self.gemini_client:
            return {"category": "medical_query_low_priority", "confidence": 0.5}

        prompt = f"""Analyze this health query for medical priority:
Query: "{query.query}"
Context: {user_context[:200]}

Classify as:
- medical_emergency: Severe/life-threatening 
- medical_query_high_priority: Complex, needs expert analysis
- medical_query_low_priority: General health questions
- simple_chitchat: Non-medical conversation

Return JSON: {{"category": "...", "confidence": 0.0-1.0}}"""

        try:
            response = await self._call_gemini_api(prompt)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"Triage error: {e}")
        
        return {"category": "medical_query_low_priority", "confidence": 0.5}

    async def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API with error handling"""
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise

    # === HEALTH CHECK ===
    async def health_check(self) -> dict:
        """Health check for ConversationManager"""
        return {
            "status": "healthy",
            "gemini_available": bool(self.gemini_client),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # === SESSION HISTORY ===
    async def get_session_history(self, session_id: str) -> dict:
        """Get session history for frontend"""
        try:
            session_data = await self.personalization_manager.get_session_data(session_id)
            if session_data:
                return {
                    "message_history": session_data.get("message_history", []),  # FIXED: Use message_history not messages
                    "session_id": session_id,
                    "symptom_profile": session_data.get("symptom_profile", {}),
                    "created_at": session_data.get("created_at"),
                    "last_updated": session_data.get("last_updated")
                }
            else:
                return {"message_history": []}  # FIXED: Return empty message_history for consistency
        except Exception as e:
            print(f"Failed to get session history: {e}")
            return {"message_history": []}  # FIXED: Return empty message_history on error

    async def get_chat_history(self, user_id: str) -> list:
        """Get chat history list for user"""
        try:
            # This would need to be implemented in personalization_manager
            return await self.personalization_manager.get_user_sessions(user_id)
        except Exception as e:
            print(f"Failed to get chat history: {e}")
            return []