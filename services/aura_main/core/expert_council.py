# File: services/aura_main/core/expert_council.py
# VERSION: 12.2 - PROMPT OPTIMIZED

import os
import json
import asyncio
import traceback
import uuid
import re
from typing import Dict, Any, AsyncGenerator

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class ExpertCouncil:
    """
    Expert Council V12.2 - Pure reasoning brain with optimized, context-aware prompts.
    Receives a full patient case and focuses on generating high-quality analysis via Gemini.
    """
    def __init__(self):
        # Initialize Gemini clients for expert roles
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_client_flash = None
        self.gemini_client_pro = None

        if self.gemini_api_key and GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # FIXED: Use standard, publicly available model names
                self.gemini_client_flash = genai.GenerativeModel('gemini-1.5-flash-latest')
                self.gemini_client_pro = genai.GenerativeModel('gemini-1.5-pro-latest')
                print("✅ [EXPERT-COUNCIL] Gemini clients initialized.")
            except Exception as e:
                print(f"⚠️ [EXPERT-COUNCIL] Gemini configuration failed: {e}")
        else:
            print("⚠️ [EXPERT-COUNCIL] Gemini unavailable. Council will not function.")
            
    # --- MAIN WORKFLOW ---
    async def run_expert_council_with_progress(self, patient_case: str, user_context: str = "", rag_context: str = "") -> AsyncGenerator[Dict, None]:
        session_id = f"council_{uuid.uuid4()}"
        start_time = asyncio.get_running_loop().time()
        
        try:
            # Step 1: System Health Check
            yield {"type": "progress", "step": "health_check", "status": "Checking system health..."}
            if not self.gemini_client_flash or not self.gemini_client_pro:
                raise ConnectionError("Expert Council Gemini clients are not available.")

            # Step 2: Parallel Analysis (Coordinator, Reasoner, Critic)
            yield {"type": "progress", "step": "parallel_analysis", "status": "Convening specialists..."}
            
            planning_task = asyncio.create_task(self._coordinator_planning(patient_case, user_context, rag_context))
            reasoning_task = asyncio.create_task(self._complex_reasoning(patient_case, user_context, rag_context))
            critique_task = asyncio.create_task(self._safety_critique(patient_case, user_context, rag_context))
            
            coordinator_analysis, complex_analysis, safety_critique = await asyncio.gather(
                planning_task, reasoning_task, critique_task
            )

            # Step 3: Consensus Synthesis
            yield {"type": "progress", "step": "synthesis", "status": "Synthesizing opinions..."}
            consensus_result = await self._consensus_synthesis(
                patient_case, coordinator_analysis, complex_analysis, safety_critique
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

    # --- OPTIMIZED SPECIALIST PROMPTS ---
    async def _coordinator_planning(self, patient_case, user_context, rag_context):
        prompt = f"""As a Lead Medical Coordinator, your task is to analyze the provided patient case and create a strategic plan.

**PATIENT CASE:**
{patient_case}

**ADDITIONAL CONTEXT:**
- User Profile: {user_context}
- Related Medical Knowledge (RAG): {rag_context[:500]}

**YOUR ANALYSIS (Focus on):**
1.  **Urgency Level:** (e.g., Critical, High, Medium, Low)
2.  **Initial Hypothesis:** What is the most likely issue?
3.  **Required Information:** What key questions should be asked next?
4.  **Specialties to Consult:** (e.g., Gastroenterology, Neurology)
"""
        return await self._call_gemini_api(self.gemini_client_flash, prompt)

    async def _complex_reasoning(self, patient_case, user_context, rag_context):
        prompt = f"""As an Advanced Medical Reasoning Specialist, your task is to perform a deep analysis of the patient case using your comprehensive medical knowledge.

**PATIENT CASE (Source of Truth):**
{patient_case}

**ADDITIONAL CONTEXT:**
- User Profile: {user_context}
- Related Medical Knowledge (RAG): {rag_context[:1000]}

**YOUR DETAILED ANALYSIS:**
1.  **Differential Diagnoses:** List at least 3 potential conditions, from most to least likely.
2.  **Pattern Recognition:** Are there any recognizable patterns or syndromes?
3.  **Risk Stratification:** What are the immediate and long-term risks to the patient?
"""
        return await self._call_gemini_api(self.gemini_client_pro, prompt)

    async def _safety_critique(self, patient_case, user_context, rag_context):
        prompt = f"""As the Chief Medical Safety Officer, your role is to identify all potential risks and safety concerns. Be critical and cautious.

**PATIENT CASE:**
{patient_case}

**ADDITIONAL CONTEXT:**
- User Profile: {user_context}
- Related Medical Knowledge (RAG): {rag_context[:500]}

**YOUR SAFETY REVIEW (Identify):**
1.  **Red Flags:** List any symptoms that require immediate emergency attention.
2.  **Potential Misinterpretations:** What could be misinterpreted from the user's description?
3.  **Clinical Blind Spots:** What potential issues might be overlooked?
4.  **Worst-Case Scenarios:** What are the most dangerous possibilities if the condition is left untreated?
"""
        return await self._call_gemini_api(self.gemini_client_flash, prompt)
        
    async def _consensus_synthesis(self, patient_case, coordinator_analysis, complex_analysis, safety_critique):
        prompt = f"""You are the Synthesis AI. Your task is to create a unified, actionable medical consensus based on the original patient case and the opinions of three specialists.

**ORIGINAL PATIENT CASE:**
{patient_case}

---
**SPECIALIST OPINIONS:**

**1. Coordinator's Plan:**
{coordinator_analysis}

**2. Reasoner's Diagnosis:**
{complex_analysis}

**3. Safety Officer's Critique:**
{safety_critique}
---

**TASK:**
Based on ALL the information above, return ONLY a valid JSON object with these exact keys:
- "primary_assessment": A single, clear sentence summarizing the most likely medical situation.
- "confidence_level": A number from 0.0 to 1.0 representing your confidence in the primary assessment.
- "key_recommendations": An array of 3-5 clear, actionable steps for the user. Start with the most important one.
- "safety_priorities": An array of the most critical "red flag" warnings from the Safety Officer's critique.

Ensure the output is a single, valid JSON object and nothing else.
"""
        response_text = await self._call_gemini_api(self.gemini_client_flash, prompt)
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
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