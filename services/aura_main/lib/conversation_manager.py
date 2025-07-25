# services/aura_main/lib/conversation_manager.py
"""
AURA Robust Conversational Inference Engine
Architecture: Trust LLM + Bulletproof Communication + Honest Failures
Enhanced: Structured Expert Council Integration
"""
import os
import json
import asyncio
import httpx
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import firebase_admin
from firebase_admin import db

# Import bulletproof utilities
from lib.utils import (
    parse_llm_json_response, 
    clean_llm_response, 
    robust_http_call,
    ServiceHealthChecker
)

class ConversationState(Enum):
    """States of medical conversation progression"""
    INITIAL_TRIAGE = "initial_triage"
    CONTEXT_BUILDING = "context_building"
    READY_FOR_ASSESSMENT = "ready_for_assessment"
    EXPERT_COUNCIL_TRIGGERED = "expert_council_triggered"
    COMPLETED = "completed"

class ConversationManager:
    """
    Robust LLM-Driven Conversational Inference Engine with bulletproof communication
    Enhanced with structured Expert Council integration
    """
    
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = firebase_admin.credentials.Certificate("config/secrets/aura-77fdb-firebase-adminsdk-fbsvc-6c88e2a1fd.json")
                firebase_admin.initialize_app(cred, {
                    'databaseURL': 'https://aura-77fdb-default-rtdb.asia-southeast1.firebasedatabase.app/'
                })
            self.db = db
        except Exception as e:
            print(f"Firebase initialization warning: {e}")
            self.db = db
            
        self.timeout = httpx.Timeout(60.0)
        self.health_checker = ServiceHealthChecker()
        self._ai_server_ready = False
        
    async def _ensure_ai_server_ready(self):
        """Ensure AI server is ready before making calls"""
        if not self._ai_server_ready:
            self._ai_server_ready = await self.health_checker.wait_for_service("http://ai_server:9000", max_wait=30)
        return self._ai_server_ready
        
    async def start_conversation(self, user_id: str, initial_query: str, 
                               user_context: str, rag_context: str) -> Dict[str, Any]:
        """Start conversation with robust entity extraction"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        session_id = f"session_{timestamp}_{user_id.replace('.', '_')}"
        
        # Robust entity extraction from initial query
        symptom_profile = await self._extract_entities_robust(
            initial_query, "", user_context, rag_context
        )
        
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "state": ConversationState.INITIAL_TRIAGE.value,
            "messages": [
                {
                    "role": "user",
                    "content": initial_query,
                    "timestamp": datetime.utcnow().isoformat(),
                    "index": 0
                }
            ],
            "symptom_profile": symptom_profile,
            "extraction_failures": 0,
            "communication_errors": 0,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        sessions_ref = self.db.reference(f'conversations/{session_id}')
        sessions_ref.set(session_data)
        
        return await self._handle_conversation_turn(session_id, initial_query, user_context, rag_context)
    
    async def continue_conversation(self, session_id: str, user_message: str,
                                  user_context: str, rag_context: str) -> Dict[str, Any]:
        """Continue conversation with robust progression"""
        try:
            session_ref = self.db.reference(f'conversations/{session_id}')
            session_data = session_ref.get()
            
            if not session_data:
                return {
                    "error": "Session not found",
                    "suggestion": "Please start a new conversation",
                    "service_used": "conversation_manager_error"
                }
            
            # Update symptom profile with robust extraction
            conversation_history = self._format_conversation_history(session_data.get("messages", []))
            updated_profile = await self._extract_entities_robust(
                user_message, conversation_history, user_context, rag_context
            )
            
            # Intelligent merge: LLM-extracted data takes precedence
            current_profile = session_data.get("symptom_profile", {})
            if updated_profile:  # Only merge if extraction succeeded
                for key, value in updated_profile.items():
                    if value is not None and value != []:
                        current_profile[key] = value
            else:
                # Track extraction failures
                session_data["extraction_failures"] = session_data.get("extraction_failures", 0) + 1
            
            message_index = len(session_data.get("messages", []))
            session_data["messages"].append({
                "role": "user", 
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "index": message_index
            })
            
            session_data["symptom_profile"] = current_profile
            
            session_ref.update({
                "messages": session_data["messages"],
                "symptom_profile": current_profile,
                "extraction_failures": session_data.get("extraction_failures", 0),
                "communication_errors": session_data.get("communication_errors", 0),
                "last_updated": datetime.utcnow().isoformat()
            })
            
            return await self._handle_conversation_turn(session_id, user_message, user_context, rag_context)
            
        except Exception as e:
            return {
                "error": f"Session error: {str(e)}",
                "suggestion": "Please start a new conversation",
                "service_used": "conversation_manager_error"
            }

    async def _extract_entities_robust(self, current_message: str, conversation_history: str, 
                                     user_context: str, rag_context: str) -> Dict[str, Any]:
        """
        Robust entity extraction with bulletproof communication and parsing
        """
        extraction_prompt = self._build_extraction_prompt(
            current_message, conversation_history, user_context, rag_context
        )
        
        for attempt in range(2):  # Self-correction: 2 attempts max
            try:
                # Ensure AI server is ready
                if not await self._ensure_ai_server_ready():
                    print("🚨 AI server not ready, returning empty profile")
                    return {}
                
                # Robust HTTP call with exponential backoff
                response_data = await self._call_ai_server_robust("/ai/wellness", {"message": extraction_prompt})
                response_text = response_data.get("response", "")
                
                if not response_text:
                    raise ValueError("Empty response from AI server")

                # Bulletproof JSON parsing
                parsed_data = parse_llm_json_response(response_text)
                if parsed_data:
                    print(f"✅ Entity extraction successful (attempt {attempt + 1})")
                    return parsed_data
                else:
                    raise ValueError("No valid JSON object found in LLM response")

            except Exception as e:
                print(f"❌ Extraction attempt {attempt + 1} failed: {e}")
                
                if attempt == 0:  # Self-correction on first failure
                    extraction_prompt = self._build_correction_prompt(
                        response_text if 'response_text' in locals() else "",
                        current_message, conversation_history, user_context
                    )
                else:
                    print("🚨 Both extraction attempts failed - returning empty profile")
                    return {}  # Honest failure
        
        return {}  # Final failure

    async def _call_ai_server_robust(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Robust AI server communication with exponential backoff and health checking
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                return await robust_http_call(
                    client, 
                    'POST', 
                    f"http://ai_server:9000{endpoint}", 
                    payload,
                    max_retries=4
                )
            except Exception as e:
                print(f"❌ AI server communication failed after all retries: {e}")
                raise

    def _build_extraction_prompt(self, message: str, history: str, user_context: str, rag_context: str) -> str:
        """Build robust extraction prompt with clear JSON instructions"""
        return f"""You are a medical information extraction specialist. Extract structured symptom data from this conversation.

MEDICAL KNOWLEDGE CONTEXT:
{rag_context[:800]}

USER PROFILE:
{user_context}

CONVERSATION HISTORY:
{history}

CURRENT PATIENT MESSAGE: "{message}"

TASK: Extract medical information into a valid JSON object. Only extract what is explicitly mentioned. Use null for missing information.

CRITICAL INSTRUCTIONS:
1. Your response MUST contain exactly one JSON object
2. Do NOT include any explanations, text, or comments outside the JSON
3. Do NOT generate example conversations or patient dialogue
4. Respond directly as the extraction system, not as a conversational agent
5. Use this exact structure:

{{
    "primary_symptom": "main complaint (string or null)",
    "onset_time": "when started (string or null)",
    "pain_type": "pain character (string or null)", 
    "associated_symptoms": ["list", "of", "additional", "symptoms"],
    "severity": "pain level or severity (string or null)",
    "triggers": "what makes it worse/better (string or null)",
    "duration": "how long episodes last (string or null)",
    "frequency": "how often occurs (string or null)",
    "emergency_indicators": ["severe", "symptoms", "requiring", "immediate", "attention"]
}}

Respond with ONLY the JSON object."""

    def _build_correction_prompt(self, failed_response: str, message: str, history: str, user_context: str) -> str:
        """Build self-correction prompt for failed extraction"""
        return f"""The previous medical information extraction failed. The response was:
"{failed_response[:200]}..."

ORIGINAL CONTEXT:
Patient Message: "{message}"
User Profile: {user_context}
Recent History: {history[-300:]}

CRITICAL: You must return ONLY a valid JSON object with the exact structure specified earlier. 
No explanations, no text before or after the JSON. Just the JSON object.

Extract medical information and return the corrected JSON object:"""

    async def _handle_conversation_turn(self, session_id: str, user_message: str,
                                      user_context: str, rag_context: str) -> Dict[str, Any]:
        """Handle conversation turn with robust routing"""
        session_data = self._get_session(session_id)
        
        # Robust escalation assessment
        escalation_decision = await self._assess_escalation_robust(session_data, user_context, rag_context)
        
        if escalation_decision.get("should_escalate", False):
            print("🏥 LLM triggered Expert Council escalation")
            return await self._handle_expert_council_escalation(session_id, user_message, user_context, rag_context)
        
        # Continue progressive consultation with robust guidance
        current_state = ConversationState(session_data["state"])
        
        if current_state in [ConversationState.INITIAL_TRIAGE, ConversationState.CONTEXT_BUILDING]:
            return await self._handle_progressive_consultation(session_id, user_message, user_context, rag_context)
        
        # Default to assessment if state unclear
        return await self._handle_assessment_phase(session_id, user_message, user_context, rag_context)

    async def _assess_escalation_robust(self, session_data: Dict, user_context: str, rag_context: str) -> Dict[str, Any]:
        """Robust LLM-driven escalation assessment"""
        symptom_profile = session_data.get("symptom_profile", {})
        messages = session_data.get("messages", [])
        conversation_history = self._format_conversation_history(messages)
        
        assessment_prompt = f"""You are a medical triage specialist. Assess if this conversation requires Expert Council escalation.

PATIENT PROFILE: {user_context}

MEDICAL KNOWLEDGE: {rag_context[:500]}

CONVERSATION SUMMARY:
{conversation_history}

EXTRACTED SYMPTOMS:
{json.dumps(symptom_profile, indent=2)}

CONVERSATION STATS:
- Total messages: {len(messages)}
- Extraction failures: {session_data.get('extraction_failures', 0)}
- Communication errors: {session_data.get('communication_errors', 0)}

ASSESSMENT TASK: Should this case be escalated to our Expert Council for comprehensive analysis?

CRITICAL: Return ONLY this JSON object:
{{
    "should_escalate": true/false,
    "confidence": 0.0-1.0,
    "primary_reason": "brief explanation",
    "urgency_level": "low/medium/high"
}}"""

        try:
            if not await self._ensure_ai_server_ready():
                # Conservative fallback: escalate on long conversations
                return {
                    "should_escalate": len(messages) >= 8,
                    "confidence": 0.5,
                    "primary_reason": "AI server not ready - defaulting to message count",
                    "urgency_level": "medium"
                }

            response_data = await self._call_ai_server_robust("/ai/wellness", {"message": assessment_prompt})
            response_text = response_data.get("response", "")

            decision = parse_llm_json_response(response_text)
            if decision:
                print(f"🎯 LLM escalation decision: {decision.get('should_escalate')} ({decision.get('primary_reason', 'No reason')})")
                return decision

        except Exception as e:
            print(f"❌ Escalation assessment failed: {e}")
        
        # Conservative fallback: escalate on long conversations
        return {
            "should_escalate": len(messages) >= 8,
            "confidence": 0.5,
            "primary_reason": "Assessment failed - defaulting to message count",
            "urgency_level": "medium"
        }

    async def _handle_progressive_consultation(self, session_id: str, user_message: str,
                                             user_context: str, rag_context: str) -> Dict[str, Any]:
        """Handle progressive consultation with robust communication"""
        session_data = self._get_session(session_id)
        
        consultation_prompt = self._build_progressive_prompt(session_data, user_context, rag_context)
        
        try:
            if not await self._ensure_ai_server_ready():
                return self._create_fallback_response(session_id, "AI service temporarily unavailable")
            
            response_data = await self._call_ai_server_robust("/ai/wellness", {"message": consultation_prompt})
            response_text = response_data.get("response", "")
            
            clean_response = clean_llm_response(response_text)
            new_state = self._determine_next_state(session_data)
            
            await self._save_conversation_turn(
                session_id,
                "medgemma_progressive_consultation_robust",
                clean_response,
                new_state,
                {"robust_communication": True, "extraction_working": bool(session_data.get("symptom_profile"))}
            )
            
            return {
                "response": clean_response,
                "session_id": session_id,
                "conversation_state": new_state.value,
                "service_used": "medgemma_progressive_consultation_robust",
                "confidence": 0.85,
                "symptom_profile": session_data.get("symptom_profile", {}),
                "next_action": self._get_next_action_guidance(new_state),
                "system_health": {
                    "ai_server_ready": self._ai_server_ready,
                    "extraction_failures": session_data.get("extraction_failures", 0),
                    "communication_errors": session_data.get("communication_errors", 0)
                }
            }
            
        except Exception as e:
            print(f"❌ Progressive consultation failed: {e}")
            return self._create_fallback_response(session_id, "I'm having difficulty processing your information. Could you rephrase your concern?")

    def _create_fallback_response(self, session_id: str, message: str) -> Dict[str, Any]:
        """Create fallback response for system errors"""
        return {
            "response": message,
            "session_id": session_id,
            "conversation_state": "system_recovery",
            "service_used": "fallback_response",
            "confidence": 0.3,
            "symptom_profile": {},
            "next_action": "Please try rephrasing your concern or start a new conversation."
        }

    def _build_progressive_prompt(self, session_data: Dict, user_context: str, rag_context: str) -> str:
        """Build context-aware prompt for progressive consultation"""
        messages = session_data.get("messages", [])
        symptom_profile = session_data.get("symptom_profile", {})
        conversation_history = self._format_conversation_history(messages[-6:])  # Last 6 messages
        
        return f"""You are AURA's medical consultation specialist. Guide this conversation efficiently toward complete understanding.

PATIENT PROFILE:
{user_context}

RELEVANT MEDICAL KNOWLEDGE:
{rag_context[:600]}

CONVERSATION HISTORY:
{conversation_history}

CURRENT SYMPTOM UNDERSTANDING:
{json.dumps(symptom_profile, indent=2)}

CONVERSATION STATS: {len(messages)} messages, {session_data.get('extraction_failures', 0)} extraction failures

TASK: Ask ONE focused, intelligent question to progress this consultation.

CRITICAL INSTRUCTIONS:
1. Respond directly as AURA - do NOT generate example conversations
2. Do NOT include "Patient:" or dialogue examples in your response
3. Ask only ONE clear question 
4. Be empathetic and professional
5. Do not include any code blocks or JSON in your response

GUIDELINES:
- Don't repeat questions already answered
- Focus on the most important missing information
- If sufficient information gathered, acknowledge and indicate readiness for analysis
- Prioritize: onset timing → severity → associated symptoms → character/triggers

Respond with ONE question or acknowledgment only."""

    async def _handle_expert_council_escalation(self, session_id: str, user_message: str,
                                              user_context: str, rag_context: str) -> Dict[str, Any]:
        """
        ENHANCED: Handle Expert Council escalation with structured output support
        """
        session_data = self._get_session(session_id)
        expert_response = await self._trigger_expert_council(session_id, user_context, rag_context)
        
        await self._save_conversation_turn(
            session_id,
            "expert_council_medagent_pro_structured",
            expert_response["user_response"],
            ConversationState.COMPLETED,
            {"robust_escalation": True, "confidence": expert_response.get("confidence", 0.7)}
        )
        
        # ENHANCED: Return structured data from Expert Council
        return {
            "response": expert_response["user_response"],
            "structured_analysis": expert_response.get("structured_analysis"),  # NEW
            "interactive_components": expert_response.get("interactive_components"),  # NEW
            "session_id": session_id,
            "conversation_state": "expert_council_active",
            "service_used": "expert_council_medagent_pro_structured",
            "expert_council_session": expert_response.get("expert_council_session", {}),
            "confidence": expert_response.get("confidence", 0.7),
            "escalation_reason": "LLM-driven intelligent escalation (structured)",
            "symptom_profile": session_data.get("symptom_profile", {}),
            "reasoning_trace": expert_response.get("reasoning_trace", {}),  # NEW
            "enhancements": {
                "structured_output": bool(expert_response.get("structured_analysis")),
                "interactive_components": bool(expert_response.get("interactive_components")),
                "fail_fast_enabled": True
            }
        }

    async def _handle_assessment_phase(self, session_id: str, user_message: str,
                                     user_context: str, rag_context: str) -> Dict[str, Any]:
        """Final assessment phase - typically escalates to Expert Council"""
        return await self._handle_expert_council_escalation(session_id, user_message, user_context, rag_context)

    def _determine_next_state(self, session_data: Dict) -> ConversationState:
        """Intelligent state progression based on conversation quality"""
        current_state = ConversationState(session_data["state"])
        messages = session_data.get("messages", [])
        symptom_profile = session_data.get("symptom_profile", {})
        
        # Count meaningful symptom data
        meaningful_data = len([v for v in symptom_profile.values() if v not in [None, [], ""]])
        
        # Progressive state transitions based on information completeness
        if current_state == ConversationState.INITIAL_TRIAGE:
            if meaningful_data >= 2 or len(messages) >= 4:
                return ConversationState.CONTEXT_BUILDING
            return ConversationState.INITIAL_TRIAGE
        
        elif current_state == ConversationState.CONTEXT_BUILDING:
            if meaningful_data >= 4 or len(messages) >= 6:
                return ConversationState.READY_FOR_ASSESSMENT
            return ConversationState.CONTEXT_BUILDING
        
        else:
            return ConversationState.READY_FOR_ASSESSMENT

    async def _trigger_expert_council(self, session_id: str, user_context: str = "", rag_context: str = "") -> Dict[str, Any]:
        """Trigger Expert Council with full conversation context"""
        from core.expert_council import expert_council
        
        session_data = self._get_session(session_id)
        conversation_summary = self._build_conversation_summary(session_data)
        
        return await expert_council.run_expert_council(
            query=conversation_summary,
            user_context=user_context or f"User from session {session_id}",
            rag_context=rag_context or "Medical knowledge context"
        )
    
    def _build_conversation_summary(self, session_data: Dict) -> str:
        """Build comprehensive conversation summary for Expert Council"""
        messages = session_data.get("messages", [])
        symptom_profile = session_data.get("symptom_profile", {})
        
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        
        summary = f"Progressive consultation summary: {' | '.join(user_messages)}"
        
        if symptom_profile:
            summary += f"\n\nStructured symptom profile:\n{json.dumps(symptom_profile, indent=2)}"
        
        return summary

    def _format_conversation_history(self, messages: List[Dict]) -> str:
        """Format conversation history for prompts"""
        formatted = []
        for msg in messages:
            role = "Patient" if msg.get("role") == "user" else "AURA"
            formatted.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(formatted)
    
    def _get_next_action_guidance(self, state: ConversationState) -> str:
        """Get next action guidance for UI"""
        guidance = {
            ConversationState.INITIAL_TRIAGE: "Please provide more details to help me understand your concern.",
            ConversationState.CONTEXT_BUILDING: "I'm building a comprehensive picture. Please continue sharing relevant details.",
            ConversationState.READY_FOR_ASSESSMENT: "Gathering final information for comprehensive analysis.",
            ConversationState.EXPERT_COUNCIL_TRIGGERED: "Consulting our medical expert team...",
            ConversationState.COMPLETED: "Consultation completed. Please follow up with healthcare provider as needed."
        }
        return guidance.get(state, "Continue the conversation")

    async def _save_conversation_turn(self, session_id: str, service_used: str, response: str, 
                                    new_state: ConversationState, metadata: Dict):
        """Save conversation turn with enhanced tracking"""
        try:
            session_ref = self.db.reference(f'conversations/{session_id}')
            session_data = session_ref.get()
            
            if not session_data:
                return
            
            message_index = len(session_data.get("messages", []))
            session_data["messages"].append({
                "role": "assistant",
                "content": response,
                "service": service_used,
                "timestamp": datetime.utcnow().isoformat(),
                "index": message_index,
                "metadata": metadata
            })
            
            updates = {
                "messages": session_data["messages"],
                "state": new_state.value,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            session_ref.update(updates)
            
        except Exception as e:
            print(f"Firebase save error: {str(e)}")
    
    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data from Firebase"""
        session_ref = self.db.reference(f'conversations/{session_id}')
        return session_ref.get() or {}
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get complete session history"""
        session_data = self._get_session(session_id)
        if not session_data:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "conversation_state": session_data.get("state"),
            "message_count": len(session_data.get("messages", [])),
            "messages": session_data.get("messages", []),
            "symptom_profile": session_data.get("symptom_profile", {}),
            "extraction_failures": session_data.get("extraction_failures", 0),
            "communication_errors": session_data.get("communication_errors", 0),
            "ai_server_ready": self._ai_server_ready,
            "created_at": session_data.get("created_at"),
            "last_updated": session_data.get("last_updated")
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with communication status"""
        try:
            test_ref = self.db.reference('system/health_check')
            test_ref.set({"timestamp": datetime.utcnow().isoformat()})
            
            ai_server_ready = await self.health_checker.wait_for_service("http://ai_server:9000", max_wait=5)
            
            return {
                "status": "healthy",
                "firebase_connection": "ok",
                "ai_server_connection": "ready" if ai_server_ready else "unavailable",
                "conversation_states": [state.value for state in ConversationState],
                "features": [
                    "bulletproof_json_parsing",
                    "robust_communication",
                    "exponential_backoff_retry",
                    "health_checking",
                    "honest_failure_handling",
                    "context_aware_prompting",
                    "structured_expert_council_integration"  # NEW
                ],
                "architecture": "robust_trust_llm_philosophy",
                "version": "inference_engine_v3_bulletproof_structured"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }