# services/aura_main/core/expert_council.py
"""
AURA Expert Council - MedAgent-Pro Hybrid Architecture
Implements 5-step collaborative medical reasoning with evidence-based debate
"""
import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

class ExpertCouncil:
    """
    MedAgent-Pro inspired Expert Council for collaborative medical reasoning
    Implements evidence-first approach with structured debate and critique
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
        
    async def run_expert_council(self, query: str, user_context: str, rag_context: str) -> Dict[str, Any]:
        """
        Execute the 5-step MedAgent-Pro workflow for complex medical reasoning
        """
        print(f"🔍 Starting Expert Council for: {query}")
        session_id = f"council_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Step 1: Planning & Evidence Gathering
            print("📋 Step 1: Creating diagnostic plan...")
            evidence_plan = await self._create_diagnostic_plan(query)
            print(f"✅ Plan created: {evidence_plan}")
            
            print("🔬 Step 2: Gathering evidence...")
            evidence = await self._gather_evidence(evidence_plan, query, user_context, rag_context)
            print(f"✅ Evidence gathered: {list(evidence.keys())}")

            # Step 2: Independent Analysis (Parallel Expert Consultation)
            print("👥 Step 3: Running independent analysis...")
            expert_analyses = await self._run_independent_analysis(query, evidence, user_context)
            print(f"✅ Expert analyses complete: {list(expert_analyses.keys())}")
            
            # Step 3: Report Synthesis
            print("📊 Step 4: Synthesizing report...")
            preliminary_report = await self._synthesize_report(expert_analyses, evidence)
            print(f"✅ Preliminary report created: {len(preliminary_report)} chars")

            # Step 4: Critical Review & Refinement
            print("🔍 Step 5: Running critical review...")
            critique = await self._run_critical_review(preliminary_report, evidence)
            print(f"✅ Critique complete: {len(critique)} chars")
            
            print("🔧 Step 6: Refining report...")
            final_report = await self._refine_report(preliminary_report, critique)
            print(f"✅ Final report created: {len(final_report)} chars")
            
            # Step 5: Final Decision & User Response
            print("💬 Step 7: Generating final response...")
            final_user_response = await self._generate_final_response(final_report, query, user_context)
            print("✅ Expert Council session complete!")

            return {
                "session_id": session_id,
                "success": True,
                "user_response": final_user_response,
                "confidence": self._calculate_confidence(expert_analyses, critique),
                "reasoning_trace": {
                    "step_1_planning": evidence_plan,
                    "step_2_evidence": self._sanitize_evidence(evidence),
                    "step_3_expert_analyses": expert_analyses,
                    "step_4_synthesis": preliminary_report[:200] + "...",
                    "step_5_critique": critique[:200] + "...",
                    "final_report": final_report[:200] + "..."
                },
                "metadata": {
                    "experts_consulted": ["medgemma_4b", "gemini_2.5_reasoner", "gemini_critique"],
                    "evidence_sources": list(evidence.keys()),
                    "session_duration": "estimated",
                    "workflow": "medagent_pro_5_step"
                }
            }

            return {
                "session_id": session_id,
                "success": True,
                "user_response": final_user_response,
                "confidence": self._calculate_confidence(expert_analyses, critique),
                "reasoning_trace": {
                    "step_1_planning": evidence_plan,
                    "step_2_evidence": self._sanitize_evidence(evidence),
                    "step_3_expert_analyses": expert_analyses,
                    "step_4_synthesis": preliminary_report[:200] + "...",
                    "step_5_critique": critique[:200] + "...",
                    "final_report": final_report[:200] + "..."
                },
                "metadata": {
                    "experts_consulted": ["medgemma_4b", "gemini_2.5_reasoner", "gemini_critique"],
                    "evidence_sources": list(evidence.keys()),
                    "session_duration": "estimated",
                    "workflow": "medagent_pro_5_step"
                }
            }
            
        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "user_response": "Expert Council consultation temporarily unavailable. Please try again.",
                "confidence": 0.1
            }

    async def _create_diagnostic_plan(self, query: str) -> Dict[str, Any]:
        """
        Step 1A: Create diagnostic plan using Coordinator Agent
        """
        planning_prompt = f"""
You are a medical diagnostic coordinator. Analyze this patient query and determine what evidence is needed.

Patient Query: "{query}"

Available diagnostic tools:
- ecg_analysis: For heart rhythm and cardiac electrical activity analysis
- vqa_analysis: For medical image interpretation (X-rays, scans, photos)  
- knowledge_base_lookup: For medical literature and evidence-based guidelines
- wellness_analysis: For mental health and lifestyle factors

Return a JSON object with:
{{
    "required_tools": ["tool1", "tool2"],
    "priority": "high|medium|low", 
    "complexity": "simple|moderate|complex",
    "reasoning": "Brief explanation of diagnostic approach"
}}

Only include tools that are actually needed for this specific query.
"""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, planning_prompt
            )
            
            # Parse JSON response
            plan_text = response.text.strip()
            if plan_text.startswith('```json'):
                plan_text = plan_text.split('```json')[1].split('```')[0]
            elif plan_text.startswith('```'):
                plan_text = plan_text.split('```')[1].split('```')[0]
                
            plan = json.loads(plan_text)
            return plan
            
        except Exception as e:
            # Fallback plan
            return {
                "required_tools": ["knowledge_base_lookup", "wellness_analysis"],
                "priority": "medium",
                "complexity": "moderate", 
                "reasoning": f"Fallback plan due to parsing error: {str(e)}"
            }

    async def _gather_evidence(self, plan: Dict, query: str, user_context: str, rag_context: str) -> Dict[str, Any]:
        """
        Step 1B: Gather evidence from required tools
        """
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
                    # Note: ECG requires actual ECG data - skip if not provided
                    evidence["ecg_note"] = "ECG analysis requested but no ECG data provided"
                elif tool == "vqa_analysis":
                    # Note: VQA requires image - skip if not provided  
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
        """Call wellness analysis tool"""
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
        """
        Step 2: Run parallel expert analysis
        """
        # Prepare evidence summary for experts
        evidence_summary = self._format_evidence_for_experts(evidence)
        
        # Run parallel expert consultations
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
        """Format evidence for expert analysis"""
        formatted = "=== CLINICAL EVIDENCE ===\n\n"
        
        for key, value in evidence.items():
            if key in ["query", "user_context"]:
                continue
            formatted += f"{key.upper()}:\n{str(value)[:300]}...\n\n"
        
        return formatted

    async def _medgemma_analysis(self, query: str, evidence: str, context: str) -> str:
        """MedGemma 4B medical expert analysis"""
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
        """Gemini 2.5 complex reasoning analysis"""
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

    async def _synthesize_report(self, analyses: Dict, evidence: Dict) -> str:
        """
        Step 3: Synthesize expert analyses into preliminary report
        """
        synthesis_prompt = f"""
You are a medical coordinator synthesizing expert opinions into a cohesive clinical report.

EXPERT ANALYSES:

PHYSICIAN ASSESSMENT:
{analyses.get('medgemma_physician', 'Unavailable')}

COMPLEX REASONING ANALYSIS: 
{analyses.get('gemini_reasoner', 'Unavailable')}

Synthesize these into a single, comprehensive medical report including:
1. Consensus findings
2. Areas of agreement and disagreement
3. Integrated diagnosis/assessment
4. Unified recommendations
5. Confidence level and limitations

Create a structured, professional medical report.
"""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, synthesis_prompt
            )
            return response.text
        except Exception as e:
            return f"Report synthesis failed: {str(e)}"

    async def _run_critical_review(self, preliminary_report: str, evidence: Dict) -> str:
        """
        Step 4A: Critical safety review by dedicated critique agent
        """
        critique_prompt = f"""
You are a medical safety specialist and critical reviewer. Your role is to find flaws, inconsistencies, and safety concerns.

PRELIMINARY MEDICAL REPORT:
{preliminary_report}

ORIGINAL EVIDENCE:
{self._format_evidence_for_experts(evidence)}

Critically analyze this report for:
1. **Logical inconsistencies** or contradictions
2. **Unsupported assumptions** or conclusions
3. **Missing considerations** or overlooked factors
4. **Safety concerns** or potential harm
5. **Overconfident statements** requiring hedging
6. **Alternative interpretations** of the evidence

Provide structured criticism with specific recommendations for improvement.
Be skeptical and thorough - patient safety depends on catching errors.
"""

        try:
            response = await asyncio.to_thread(
                self.critique_model.generate_content, critique_prompt
            )
            return response.text
        except Exception as e:
            return f"Critical review failed: {str(e)}"

    async def _refine_report(self, preliminary_report: str, critique: str) -> str:
        """
        Step 4B: Refine report based on critique
        """
        refinement_prompt = f"""
You are refining a medical report based on critical feedback to improve accuracy and safety.

ORIGINAL REPORT:
{preliminary_report}

CRITICAL REVIEW:
{critique}

Create an improved version that addresses the critique by:
1. Correcting identified errors or inconsistencies
2. Adding appropriate hedging and uncertainty statements
3. Including missing considerations
4. Strengthening evidence-based reasoning
5. Enhancing safety warnings where needed

Maintain medical professionalism while incorporating the feedback.
"""

        try:
            response = await asyncio.to_thread(
                self.complex_reasoner_model.generate_content, refinement_prompt
            )
            return response.text
        except Exception as e:
            return f"Report refinement failed: {str(e)}. Using original report."

    async def _generate_final_response(self, final_report: str, query: str, user_context: str) -> str:
        """
        Step 5: Generate empathetic user-facing response
        """
        # Extract communication preference from user context
        comm_style = "friendly and empathetic"
        if "formal" in user_context.lower():
            comm_style = "professional and formal"
        
        response_prompt = f"""
You are AURA, a compassionate AI Health Guardian. Transform this medical report into a helpful response for the patient.

PATIENT QUESTION: {query}
PATIENT PREFERENCES: {comm_style} communication

EXPERT COUNCIL ANALYSIS:
{final_report}

Create a response that:
1. Directly addresses the patient's question
2. Uses {comm_style} language appropriate for the patient
3. Explains medical concepts clearly without jargon
4. Includes appropriate disclaimers about consulting healthcare professionals
5. Maintains empathy and support
6. Highlights key takeaways and next steps

Remember: You are a health guardian, not a replacement for medical care.
"""

        try:
            response = await asyncio.to_thread(
                self.coordinator_model.generate_content, response_prompt
            )
            return response.text
        except Exception as e:
            return f"I encountered an issue generating my response. The expert council has completed its analysis, but I recommend consulting with a healthcare professional about your concern: {query}"

    def _calculate_confidence(self, analyses: Dict, critique: str) -> float:
        """Calculate overall confidence based on expert agreement and critique severity"""
        base_confidence = 0.7
        
        # Check for expert agreement indicators
        medgemma_text = str(analyses.get('medgemma_physician', ''))
        gemini_text = str(analyses.get('gemini_reasoner', ''))
        
        # Simple agreement check (could be enhanced with semantic similarity)
        if "uncertain" in medgemma_text.lower() or "unclear" in gemini_text.lower():
            base_confidence -= 0.2
            
        # Critique severity check
        if "safety concern" in critique.lower() or "error" in critique.lower():
            base_confidence -= 0.3
        elif "minor" in critique.lower() or "generally sound" in critique.lower():
            base_confidence += 0.1
            
        return max(0.1, min(0.9, base_confidence))  # Cap between 10% and 90%

    def _sanitize_evidence(self, evidence: Dict) -> Dict:
        """Sanitize evidence for reasoning trace"""
        sanitized = {}
        for key, value in evidence.items():
            if isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        return sanitized

# Initialize singleton
expert_council = ExpertCouncil()