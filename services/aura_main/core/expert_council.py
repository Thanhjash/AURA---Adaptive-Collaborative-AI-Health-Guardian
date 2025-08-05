# File: services/aura_main/core/expert_council.py (VERSION V12.1 - OPTIMIZED)

import os
import json
import asyncio
import traceback
import uuid
import re
from typing import Dict, Any, AsyncGenerator

# Expert Council uses Gemini for specialist roles
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class ExpertCouncil:
    """
    Expert Council V12.1 - Pure reasoning brain.
    Receives full context and focuses on generating analysis via Gemini.
    """
    def __init__(self):
        # Initialize Gemini clients for expert roles
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_client_flash = None
        self.gemini_client_pro = None

        if self.gemini_api_key and GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client_flash = genai.GenerativeModel('gemini-2.0-flash')
                self.gemini_client_pro = genai.GenerativeModel('gemini-2.5-flash')
                print("✅ [EXPERT-COUNCIL] Gemini clients initialized.")
            except Exception as e:
                print(f"⚠️ [EXPERT-COUNCIL] Gemini configuration failed: {e}")
        else:
            print("⚠️ [EXPERT-COUNCIL] Gemini unavailable. Council will not function.")
            
    # --- MAIN WORKFLOW ---
    async def run_expert_council_with_progress(self, query: str, user_context: str = "", rag_context: str = "") -> AsyncGenerator[Dict, None]:
        session_id = f"council_{uuid.uuid4()}"
        start_time = asyncio.get_running_loop().time()
        
        try:
            # Step 1: System Health Check
            yield {"type": "progress", "step": "health_check", "status": "Checking system health..."}
            if not self.gemini_client_flash or not self.gemini_client_pro:
                raise ConnectionError("Expert Council Gemini clients are not available.")

            # Step 2: Parallel Analysis (Coordinator, Reasoner, Critic)
            yield {"type": "progress", "step": "parallel_analysis", "status": "Convening specialists..."}
            
            planning_task = asyncio.create_task(self._coordinator_planning(query, user_context, rag_context))
            reasoning_task = asyncio.create_task(self._complex_reasoning(query, user_context, rag_context))
            critique_task = asyncio.create_task(self._safety_critique(query, user_context, rag_context))
            
            coordinator_analysis, complex_analysis, safety_critique = await asyncio.gather(
                planning_task, reasoning_task, critique_task
            )

            # Step 3: Consensus Synthesis
            yield {"type": "progress", "step": "synthesis", "status": "Synthesizing opinions..."}
            consensus_result = await self._consensus_synthesis(
                query, coordinator_analysis, complex_analysis, safety_critique, rag_context
            )
            
            # Step 4: Final Formatting
            yield {"type": "progress", "step": "formatting", "status": "Formatting final response..."}
            final_response = self._format_user_response(consensus_result)

            end_time = asyncio.get_running_loop().time()
            
            # Package final result
            final_result_package = {
                "user_response": final_response,
                "confidence": consensus_result.get("confidence_level", 0.7),
                "reasoning_trace": {
                    "coordinator": coordinator_analysis,
                    "reasoner": complex_analysis,
                    "critic": safety_critique,
                    "synthesis": consensus_result,
                },
                "duration_seconds": round(end_time - start_time, 2),
                "session_id": session_id,
            }
            yield {"type": "result", "data": final_result_package}

        except Exception as e:
            traceback.print_exc()
            error_response = {
                "user_response": "I apologize, but the Expert Council encountered an unexpected issue. We will use standard guidance for now.",
                "error": str(e)
            }
            yield {"type": "result", "data": error_response}

    # --- SPECIALIST FUNCTIONS ---
    async def _call_gemini_api(self, client, prompt: str) -> str:
        try:
            response = await client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"[EXPERT-COUNCIL] Gemini call failed: {e}")
            raise e

    async def _coordinator_planning(self, query, user_context, rag_context):
        prompt = f"""As Lead Medical Coordinator, provide a strategic analysis for this query. Focus on priority, required specialties, and initial hypothesis.

Query: {query}
Context: {user_context}
Knowledge: {rag_context[:500]}

Provide a structured analysis focusing on medical priority and specialty requirements."""
        return await self._call_gemini_api(self.gemini_client_flash, prompt)

    async def _complex_reasoning(self, query, user_context, rag_context):
        prompt = f"""As an Advanced Medical Reasoning Specialist, provide a differential diagnosis, pattern recognition, and risk stratification. Use comprehensive medical reasoning.

Query: {query}
Context: {user_context}
Knowledge: {rag_context[:1000]}

Provide detailed differential diagnosis and risk assessment."""
        return await self._call_gemini_api(self.gemini_client_pro, prompt)

    async def _safety_critique(self, query, user_context, rag_context):
        prompt = f"""As Chief Medical Safety Officer, review this case for safety concerns, red flags, and clinical blind spots.

Query: {query}
Context: {user_context}
Knowledge: {rag_context[:500]}

Focus on safety concerns and potential risks that must be addressed."""
        return await self._call_gemini_api(self.gemini_client_flash, prompt)
        
    async def _consensus_synthesis(self, query, coordinator_analysis, complex_analysis, safety_critique, rag_context):
        prompt = f"""Synthesize these expert opinions into a unified medical consensus.

Coordinator Analysis: {coordinator_analysis}
Complex Reasoning: {complex_analysis}
Safety Critique: {safety_critique}

Task: Return ONLY a valid JSON object with these exact keys:
- "primary_assessment": string describing main finding
- "confidence_level": number from 0.0 to 1.0
- "key_recommendations": array of recommendation strings
- "safety_priorities": array of safety concern strings

Ensure valid JSON format."""

        response_text = await self._call_gemini_api(self.gemini_client_flash, prompt)
        
        try:
            # Extract JSON from response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback parsing
                return json.loads(response_text.strip())
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"JSON parsing failed: {e}")
            return {
                "primary_assessment": "Expert analysis completed with synthesis challenges.",
                "confidence_level": 0.6,
                "key_recommendations": ["Consult healthcare provider for personalized advice"],
                "safety_priorities": ["Professional medical evaluation recommended"]
            }

    def _format_user_response(self, consensus_result: Dict) -> str:
        assessment = consensus_result.get("primary_assessment", "Assessment complete.")
        recommendations = consensus_result.get("key_recommendations", [])
        safety_priorities = consensus_result.get("safety_priorities", [])
        
        response = f"**Expert Council Assessment:**\n\n**Primary Finding:** {assessment}\n\n"
        
        if recommendations:
            response += "**Key Recommendations:**\n"
            for i, rec in enumerate(recommendations, 1):
                response += f"{i}. {rec}\n"
        
        if safety_priorities:
            response += "\n**Important Safety Notes:**\n"
            for i, priority in enumerate(safety_priorities, 1):
                response += f"⚠️ {priority}\n"
                
        response += "\n*Always consult with qualified healthcare professionals for medical decisions.*"
        return response

# Initialize singleton instance for import
expert_council = ExpertCouncil()