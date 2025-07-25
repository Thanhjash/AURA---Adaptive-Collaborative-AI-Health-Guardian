# services/aura_main/core/expert_council.py
"""
AURA Expert Council - Enhanced MedAgent-Pro with Structured Output
Philosophy: Fail Fast & Honestly + Structured JSON Output
"""
import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import google.generativeai as genai
from lib.utils import parse_llm_json_response, clean_llm_response

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

class ExpertCouncilError(Exception):
    """Custom exception for Expert Council failures"""
    def __init__(self, step: str, message: str, details: str = ""):
        self.step = step
        self.message = message
        self.details = details
        super().__init__(f"Step {step} failed: {message}")

class ExpertCouncil:
    """
    Enhanced Expert Council with structured output and fail-fast philosophy
    """
    
    def __init__(self):
        self.timeout = httpx.Timeout(60.0)
        
        # Initialize Gemini models
        self.coordinator_model = genai.GenerativeModel('gemini-2.0-flash')
        self.complex_reasoner_model = genai.GenerativeModel('gemini-2.5-flash')
        self.critique_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Evidence gathering tools
        self.available_tools = {
            "ecg_analysis": "http://ecg_interpreter:8001/analyze",
            "vqa_analysis": "http://ai_server:9000/ai/vqa", 
            "knowledge_base_lookup": "rag_system",
            "wellness_analysis": "http://ai_server:9000/ai/wellness"
        }
        
    async def run_expert_council(self, query: str, user_context: str, rag_context: str, 
                               status_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute MedAgent-Pro workflow with fail-fast philosophy and structured output
        """
        session_id = f"council_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        def update_status(step: str, message: str):
            if status_callback:
                status_callback({"step": step, "message": message, "session_id": session_id})
            print(f"🔍 {step}: {message}")
        
        try:
            # FAIL FAST: Check AI server availability first
            update_status("system_check", "Verifying AI server availability...")
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                try:
                    response = await client.get("http://ai_server:9000/health")
                    if response.status_code != 200:
                        raise ExpertCouncilError(
                            "system_check", 
                            "AI server health check failed",
                            f"Health endpoint returned status {response.status_code}"
                        )
                except Exception as e:
                    raise ExpertCouncilError(
                        "system_check",
                        "AI server unavailable", 
                        f"Cannot connect to MedGemma service: {str(e)}"
                    )
            # Step 1: Planning (FAIL FAST - NO FALLBACKS)
            update_status("diagnostic_planning", "Creating evidence-based diagnostic plan...")
            evidence_plan = await self._create_diagnostic_plan_strict(query)
            
            # Step 2: Evidence Gathering  
            update_status("evidence_gathering", "Gathering clinical evidence...")
            evidence = await self._gather_evidence(evidence_plan, query, user_context, rag_context)
            
            # Step 3: Expert Analysis (Parallel)
            update_status("expert_analysis", "Multiple experts analyzing case...")
            expert_analyses = await self._run_independent_analysis(query, evidence, user_context)
            
            # Step 4: Structured Synthesis (JSON OUTPUT)
            update_status("synthesis", "Synthesizing expert opinions into structured report...")
            structured_report = await self._synthesize_structured_report(expert_analyses, evidence)
            
            # Step 5: Critical Review (STRUCTURED)
            update_status("critical_review", "Running safety and accuracy review...")
            structured_critique = await self._run_structured_critique(structured_report, evidence)
            
            # Step 6: Final Refinement (STRUCTURED)
            update_status("refinement", "Refining analysis based on critique...")
            final_structured_analysis = await self._refine_structured_analysis(structured_report, structured_critique)
            
            # Step 7: User Response Generation
            update_status("response_generation", "Generating personalized response...")
            user_response = await self._generate_user_response_from_structure(final_structured_analysis, query, user_context)
            
            update_status("completed", "Expert Council analysis complete")
            
            return {
                "session_id": session_id,
                "success": True,
                "user_response": user_response,
                "structured_analysis": final_structured_analysis,  # NEW: Full structured data
                "confidence": self._calculate_confidence_from_structure(final_structured_analysis),
                "interactive_components": self._generate_ui_components(final_structured_analysis),  # NEW: UI data
                "reasoning_trace": {
                    "diagnostic_plan": evidence_plan,
                    "evidence_gathered": self._sanitize_evidence(evidence),
                    "expert_analyses": {k: v[:200] + "..." for k, v in expert_analyses.items()},
                    "structured_synthesis": structured_report,
                    "structured_critique": structured_critique,
                    "final_analysis": final_structured_analysis
                },
                "metadata": {
                    "experts_consulted": ["medgemma_4b", "gemini_2.5_reasoner", "gemini_critique"],
                    "evidence_sources": list(evidence.keys()),
                    "workflow": "medagent_pro_structured_v3",
                    "processing_steps": 7
                }
            }
            
        except ExpertCouncilError as ece:
            # Specific Expert Council errors with detailed feedback
            return {
                "session_id": session_id,
                "success": False,
                "error_type": "expert_council_error",
                "failed_step": ece.step,
                "error_message": ece.message,
                "error_details": ece.details,
                "user_response": self._generate_error_response(ece, query),
                "confidence": 0.1,
                "suggestion": self._get_error_suggestion(ece.step)
            }
            
        except Exception as e:
            # Unexpected system errors
            return {
                "session_id": session_id,
                "success": False,
                "error_type": "system_error",
                "error_message": str(e),
                "user_response": "I'm experiencing technical difficulties. Please try again in a moment.",
                "confidence": 0.1
            }

    async def _create_diagnostic_plan_strict(self, query: str) -> Dict[str, Any]:
        """
        ENHANCED: Strict diagnostic planning - NO FALLBACKS, FAIL FAST
        """
        planning_prompt = f"""You are a medical diagnostic coordinator. Create an evidence-based diagnostic plan.

PATIENT QUERY: "{query}"

Available tools:
- ecg_analysis: Heart rhythm/electrical analysis
- vqa_analysis: Medical image interpretation  
- knowledge_base_lookup: Medical literature
- wellness_analysis: Mental health/lifestyle factors

CRITICAL: Return ONLY this exact JSON structure:
{{
    "required_tools": ["tool1", "tool2"],
    "priority": "high|medium|low",
    "complexity": "simple|moderate|complex", 
    "reasoning": "Brief diagnostic approach explanation",
    "expected_evidence_types": ["symptom_analysis", "differential_diagnosis", "risk_assessment"]
}}

Select only tools actually needed for this specific query."""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, planning_prompt
            )
            
            # STRICT JSON PARSING - NO FALLBACKS
            plan_data = parse_llm_json_response(response.text)
            if not plan_data:
                raise ExpertCouncilError(
                    "diagnostic_planning", 
                    "Failed to generate valid diagnostic plan",
                    f"LLM response was not valid JSON: {response.text[:200]}..."
                )
            
            # Validate required fields
            required_fields = ["required_tools", "priority", "complexity", "reasoning"]
            missing_fields = [field for field in required_fields if field not in plan_data]
            if missing_fields:
                raise ExpertCouncilError(
                    "diagnostic_planning",
                    f"Diagnostic plan missing required fields: {missing_fields}",
                    f"Received plan: {plan_data}"
                )
                
            return plan_data
            
        except ExpertCouncilError:
            raise  # Re-raise our specific errors
        except Exception as e:
            raise ExpertCouncilError(
                "diagnostic_planning",
                "Diagnostic planning system failure", 
                f"Unexpected error: {str(e)}"
            )

    async def _synthesize_structured_report(self, analyses: Dict, evidence: Dict) -> Dict[str, Any]:
        """
        NEW: Synthesize into structured JSON format (NO TEXT BLOBS)
        """
        synthesis_prompt = f"""Synthesize expert medical analyses into structured format.

EXPERT ANALYSES:
Physician Assessment: {analyses.get('medgemma_physician', 'Unavailable')}
Complex Reasoning: {analyses.get('gemini_reasoner', 'Unavailable')}

CLINICAL EVIDENCE:
{self._format_evidence_for_experts(evidence)}

TASK: Create structured medical assessment in this EXACT JSON format:

{{
    "clinical_summary": {{
        "primary_concern": "main medical issue identified",
        "urgency_level": "low|medium|high|critical",
        "overall_assessment": "brief clinical overview (max 2 sentences)"
    }},
    "differential_diagnoses": [
        {{
            "condition": "medical condition name",
            "probability": "high|medium|low", 
            "supporting_evidence": ["evidence point 1", "evidence point 2"],
            "risk_level": "low|medium|high",
            "next_steps": "recommended evaluation/testing"
        }}
    ],
    "recommendations": {{
        "immediate_actions": ["specific action 1", "specific action 2"],
        "follow_up_care": ["followup item 1", "followup item 2"], 
        "monitoring": ["what to monitor 1", "what to monitor 2"],
        "lifestyle_modifications": ["modification 1", "modification 2"]
    }},
    "safety_considerations": {{
        "red_flags": ["warning sign 1", "warning sign 2"],
        "emergency_indicators": ["emergency sign 1", "emergency sign 2"],
        "contraindications": ["avoid this", "be careful with that"]
    }},
    "confidence_assessment": {{
        "overall_confidence": 0.75,
        "data_completeness": 0.8,
        "expert_consensus": 0.9,
        "uncertainty_areas": ["area of uncertainty 1", "area of uncertainty 2"]
    }}
}}

CRITICAL: Return ONLY the JSON object. No explanations."""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, synthesis_prompt
            )
            
            structured_data = parse_llm_json_response(response.text)
            if not structured_data:
                raise ExpertCouncilError(
                    "synthesis",
                    "Failed to generate structured medical report",
                    f"Could not parse JSON from synthesis response: {response.text[:200]}..."
                )
                
            # Validate structure
            required_sections = ["clinical_summary", "differential_diagnoses", "recommendations", "safety_considerations"]
            missing_sections = [section for section in required_sections if section not in structured_data]
            if missing_sections:
                raise ExpertCouncilError(
                    "synthesis",
                    f"Structured report missing sections: {missing_sections}",
                    f"Received structure keys: {list(structured_data.keys())}"
                )
                
            return structured_data
            
        except ExpertCouncilError:
            raise
        except Exception as e:
            raise ExpertCouncilError(
                "synthesis",
                "Report synthesis system failure",
                f"Unexpected error: {str(e)}"
            )

    async def _run_structured_critique(self, structured_report: Dict, evidence: Dict) -> Dict[str, Any]:
        """
        NEW: Run structured critique (JSON OUTPUT)
        """
        critique_prompt = f"""Review this structured medical report for safety and accuracy.

STRUCTURED REPORT:
{json.dumps(structured_report, indent=2)}

ORIGINAL EVIDENCE:
{self._format_evidence_for_experts(evidence)}

TASK: Provide structured critique in this EXACT JSON format:

{{
    "safety_assessment": {{
        "critical_safety_issues": ["issue 1", "issue 2"],
        "potential_missed_diagnoses": ["diagnosis 1", "diagnosis 2"],
        "risk_underestimation": ["underestimated risk 1", "underestimated risk 2"]
    }},
    "accuracy_review": {{
        "logical_inconsistencies": ["inconsistency 1", "inconsistency 2"],
        "unsupported_conclusions": ["conclusion 1", "conclusion 2"],
        "evidence_gaps": ["missing evidence 1", "missing evidence 2"]
    }},
    "recommendation_review": {{
        "inappropriate_recommendations": ["recommendation 1", "recommendation 2"],
        "missing_critical_actions": ["missing action 1", "missing action 2"],
        "timing_concerns": ["timing issue 1", "timing issue 2"]
    }},
    "confidence_calibration": {{
        "overconfident_areas": ["area 1", "area 2"],
        "underconfident_areas": ["area 1", "area 2"],
        "uncertainty_not_acknowledged": ["unacknowledged uncertainty 1"]
    }},
    "overall_critique": {{
        "safety_score": 0.85,
        "accuracy_score": 0.9,
        "completeness_score": 0.8,
        "major_concerns": ["major concern 1", "major concern 2"],
        "minor_improvements": ["minor improvement 1", "minor improvement 2"]
    }}
}}

CRITICAL: Return ONLY the JSON object."""

        try:
            response = await asyncio.to_thread(
                self.critique_model.generate_content, critique_prompt
            )
            
            critique_data = parse_llm_json_response(response.text)
            if not critique_data:
                raise ExpertCouncilError(
                    "critical_review",
                    "Failed to generate structured critique",
                    f"Could not parse JSON from critique: {response.text[:200]}..."
                )
                
            return critique_data
            
        except ExpertCouncilError:
            raise
        except Exception as e:
            raise ExpertCouncilError(
                "critical_review", 
                "Critical review system failure",
                f"Unexpected error: {str(e)}"
            )

    async def _refine_structured_analysis(self, original_analysis: Dict, critique: Dict) -> Dict[str, Any]:
        """
        NEW: Refine analysis based on structured critique
        """
        refinement_prompt = f"""Refine medical analysis based on structured critique.

ORIGINAL ANALYSIS:
{json.dumps(original_analysis, indent=2)}

STRUCTURED CRITIQUE:
{json.dumps(critique, indent=2)}

TASK: Create improved analysis addressing critique issues. Use SAME JSON structure as original but with improvements.

Address:
- Safety concerns from critique
- Accuracy issues identified  
- Missing recommendations
- Confidence calibration problems

Return refined analysis with identical JSON structure but improved content.

CRITICAL: Return ONLY the improved JSON object."""

        try:
            response = await asyncio.to_thread(
                self.complex_reasoner_model.generate_content, refinement_prompt
            )
            
            refined_data = parse_llm_json_response(response.text)
            if not refined_data:
                # If refinement fails, return original with critique notes
                original_analysis["refinement_note"] = "Refinement failed - using original analysis"
                original_analysis["critique_summary"] = critique.get("overall_critique", {})
                return original_analysis
                
            # Add refinement metadata
            refined_data["refinement_applied"] = True
            refined_data["critique_addressed"] = True
            
            return refined_data
            
        except Exception as e:
            # Graceful degradation - return original with error note
            original_analysis["refinement_error"] = str(e)
            return original_analysis

    async def _generate_user_response_from_structure(self, structured_analysis: Dict, query: str, user_context: str) -> str:
        """
        Generate natural response from structured analysis
        """
        comm_style = "friendly and empathetic"
        if "formal" in user_context.lower():
            comm_style = "professional and formal"
        
        response_prompt = f"""Transform structured medical analysis into natural patient response.

PATIENT QUESTION: {query}
COMMUNICATION STYLE: {comm_style}

STRUCTURED ANALYSIS:
{json.dumps(structured_analysis, indent=2)}

Create natural, conversational response that:
1. Directly addresses patient's question using {comm_style} tone
2. Explains key findings from structured analysis clearly
3. Highlights important next steps and safety considerations
4. Includes appropriate medical disclaimers
5. Maintains empathy and support
6. Avoids medical jargon

Focus on most important elements from the structured data."""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, response_prompt
            )
            return clean_llm_response(response.text)
        except Exception as e:
            # Fallback to basic structured response
            primary_concern = structured_analysis.get("clinical_summary", {}).get("primary_concern", "your health concern")
            urgency = structured_analysis.get("clinical_summary", {}).get("urgency_level", "medium")
            
            return f"Based on our expert analysis of {primary_concern}, this appears to be a {urgency} priority situation. I recommend consulting with a healthcare professional for proper evaluation and personalized care."

    def _calculate_confidence_from_structure(self, structured_analysis: Dict) -> float:
        """Calculate confidence from structured metrics"""
        confidence_data = structured_analysis.get("confidence_assessment", {})
        return confidence_data.get("overall_confidence", 0.7)

    def _generate_ui_components(self, structured_analysis: Dict) -> Dict[str, Any]:
        """
        Generate interactive UI components for frontend
        """
        clinical_summary = structured_analysis.get("clinical_summary", {})
        
        return {
            "summary_card": {
                "primary_concern": clinical_summary.get("primary_concern", "Health consultation completed"),
                "urgency_level": clinical_summary.get("urgency_level", "medium"),
                "confidence": structured_analysis.get("confidence_assessment", {}).get("overall_confidence", 0.7)
            },
            "diagnoses_panel": {
                "differential_diagnoses": structured_analysis.get("differential_diagnoses", []),
                "expandable": True
            },
            "action_checklist": {
                "immediate": structured_analysis.get("recommendations", {}).get("immediate_actions", []),
                "follow_up": structured_analysis.get("recommendations", {}).get("follow_up_care", []),
                "monitoring": structured_analysis.get("recommendations", {}).get("monitoring", [])
            },
            "safety_alerts": {
                "red_flags": structured_analysis.get("safety_considerations", {}).get("red_flags", []),
                "emergency_signs": structured_analysis.get("safety_considerations", {}).get("emergency_indicators", []),
                "priority": "high" if clinical_summary.get("urgency_level") in ["high", "critical"] else "medium"
            }
        }

    def _generate_error_response(self, error: ExpertCouncilError, query: str) -> str:
        """Generate helpful error response based on failed step"""
        error_responses = {
            "diagnostic_planning": f"I'm having trouble creating a diagnostic plan for your question: '{query}'. Could you try rephrasing it or providing more specific details?",
            "evidence_gathering": "I encountered issues gathering medical evidence. Some analysis tools may be temporarily unavailable. Please try again in a moment.",
            "expert_analysis": "Our expert analysis system is experiencing difficulties. Please try a simpler query or try again later.",
            "synthesis": "I'm having trouble synthesizing the expert opinions into a coherent analysis. The case may be too complex for current processing.",
            "critical_review": "Our safety review system encountered an issue. For safety, I recommend consulting directly with a healthcare professional.",
            "refinement": "The analysis refinement process encountered an issue, but I can provide the preliminary assessment.",
            "response_generation": "I completed the medical analysis but had trouble generating the final response. Please consult a healthcare professional about your concern."
        }
        
        return error_responses.get(error.step, f"I encountered an issue during {error.step}. Please try again or consult a healthcare professional.")

    def _get_error_suggestion(self, failed_step: str) -> str:
        """Get specific suggestion based on failed step"""
        suggestions = {
            "diagnostic_planning": "Try rephrasing your question with more specific symptoms or details",
            "evidence_gathering": "Check if you can provide additional context or try again in a few minutes",
            "expert_analysis": "Consider using the progressive consultation for simpler queries",
            "synthesis": "This query may be too complex - try breaking it into simpler questions",
            "critical_review": "For safety concerns, always consult a healthcare professional directly",
            "refinement": "The preliminary analysis may still be helpful",
            "response_generation": "The analysis was completed - consult a healthcare professional for interpretation"
        }
        
        return suggestions.get(failed_step, "Please try again or start a new conversation")

    # Utility methods from original implementation
    async def _gather_evidence(self, plan: Dict, query: str, user_context: str, rag_context: str) -> Dict[str, Any]:
        """Gather evidence from required tools (implementation unchanged)"""
        # ... existing implementation from original code
        evidence = {
            "query": query,
            "user_context": user_context,
            "knowledge_base": rag_context
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = []
            
            for tool in plan.get("required_tools", []):
                if tool == "knowledge_base_lookup":
                    evidence["knowledge_base"] = rag_context
                elif tool == "ecg_analysis":
                    evidence["ecg_note"] = "ECG analysis requested but no ECG data provided"
                elif tool == "vqa_analysis":
                    evidence["vqa_note"] = "Medical image analysis requested but no image provided"
                elif tool == "wellness_analysis":
                    task = self._call_wellness_tool(client, query, user_context)
                    tasks.append(("wellness_analysis", task))
            
            # Execute tool calls
            if tasks:
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for i, (tool_name, _) in enumerate(tasks):
                    result = results[i]
                    if not isinstance(result, Exception):
                        evidence[tool_name] = result
                    else:
                        evidence[f"{tool_name}_error"] = str(result)
        
        return evidence

    async def _call_wellness_tool(self, client: httpx.AsyncClient, query: str, context: str) -> Dict:
        """Call wellness analysis tool (implementation unchanged)"""
        try:
            response = await client.post(
                self.available_tools["wellness_analysis"],
                json={"message": f"Medical wellness analysis for: {query}. Patient context: {context}"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "tool": "wellness_analysis"}

    async def _run_independent_analysis(self, query: str, evidence: Dict, user_context: str) -> Dict[str, Any]:
        """Run parallel expert analysis (implementation unchanged)"""
        evidence_summary = self._format_evidence_for_experts(evidence)
        
        tasks = [
            self._medgemma_analysis(query, evidence_summary, user_context),
            self._gemini_complex_reasoning(query, evidence_summary, user_context)
        ]
        
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "medgemma_physician": analyses[0] if not isinstance(analyses[0], Exception) else f"Error: {analyses[0]}",
            "gemini_reasoner": analyses[1] if not isinstance(analyses[1], Exception) else f"Error: {analyses[1]}"
        }

    def _format_evidence_for_experts(self, evidence: Dict) -> str:
        """Format evidence for expert analysis (implementation unchanged)"""
        formatted = "=== CLINICAL EVIDENCE ===\n\n"
        
        for key, value in evidence.items():
            if key in ["query", "user_context"]:
                continue
            formatted += f"{key.upper()}:\n{str(value)[:300]}...\n\n"
        
        return formatted

    async def _medgemma_analysis(self, query: str, evidence: str, context: str) -> str:
        """MedGemma analysis (implementation unchanged)"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                prompt = f"""
You are an experienced physician with comprehensive medical training. Provide a thorough medical analysis.

PATIENT QUERY: {query}
PATIENT CONTEXT: {context}

{evidence}

Please provide:
1. Clinical assessment
2. Preliminary diagnosis/differential diagnoses  
3. Recommended next steps
4. Risk factors and warnings

Respond as a medical professional would to a colleague.
"""
                
                response = await client.post(
                    "http://ai_server:9000/ai/wellness",
                    json={"message": prompt}
                )
                response.raise_for_status()
                return response.json().get("response", "MedGemma analysis unavailable")
                
        except Exception as e:
            return f"MedGemma consultation failed: {str(e)}"

    async def _gemini_complex_reasoning(self, query: str, evidence: str, context: str) -> str:
        """Gemini complex reasoning (implementation unchanged)"""
        reasoning_prompt = f"""
You are a medical reasoning specialist focused on complex pattern recognition and differential diagnosis.

PATIENT CASE: {query}
PATIENT BACKGROUND: {context}

{evidence}

Analyze this case focusing on:
1. Hidden connections and patterns in the evidence
2. Multiple differential diagnoses with likelihood assessment
3. Potential complications or overlooked risk factors
4. Logical inconsistencies or gaps in information
5. Evidence-based reasoning for each conclusion

Provide systematic, analytical reasoning like a medical detective.
"""

        try:
            response = await asyncio.to_thread(
                self.complex_reasoner_model.generate_content, reasoning_prompt
            )
            return response.text
        except Exception as e:
            return f"Complex reasoning analysis failed: {str(e)}"

    def _sanitize_evidence(self, evidence: Dict) -> Dict:
        """Sanitize evidence for reasoning trace (implementation unchanged)"""
        sanitized = {}
        for key, value in evidence.items():
            if isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        return sanitized

# Initialize singleton
expert_council = ExpertCouncil()