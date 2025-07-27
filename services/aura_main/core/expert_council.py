# services/aura_main/core/expert_council.py
"""
Enhanced Expert Council with Session Observability
Implements complete session logging with try/finally pattern for guaranteed observability
"""
import asyncio
import httpx
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import google.generativeai as genai
import os

# Import for session logging
from lib.personalization_manager import PersonalizationManager

class ExpertCouncilError(Exception):
    """Custom exception for Expert Council errors with step tracking"""
    def __init__(self, message: str, step: str, error_type: str = "expert_council_error"):
        self.message = message
        self.step = step
        self.error_type = error_type
        super().__init__(self.message)

@dataclass
class ExpertCouncilStep:
    """Data class for tracking individual council steps"""
    step_name: str
    description: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

class ExpertCouncil:
    """
    Enhanced Expert Council with guaranteed session observability
    Implements MedAgent-Pro methodology with comprehensive logging
    """
    
    def __init__(self):
        # Initialize Gemini clients
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ GOOGLE_API_KEY not found - Expert Council will fail")
            self.gemini_enabled = False
        else:
            genai.configure(api_key=api_key)
            self.coordinator_client = genai.GenerativeModel('gemini-2.0-flash')
            self.reasoner_client = genai.GenerativeModel('gemini-2.5-flash')
            self.critic_client = genai.GenerativeModel('gemini-2.0-flash')
            self.gemini_enabled = True
        
        # Initialize session logger
        self.pm = PersonalizationManager()
        
        # HTTP client for microservice calls
        self.timeout = httpx.Timeout(60.0)
    
    async def run_expert_council(
        self, 
        query: str, 
        user_context: str = "", 
        rag_context: str = ""
    ) -> Dict[str, Any]:
        """
        Enhanced Expert Council with guaranteed session observability
        Uses try/finally to ensure session is ALWAYS logged regardless of success/failure
        """
        
        # Generate unique session ID
        session_id = f"council_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        
        # Initialize council result and step tracking
        council_result = {}
        steps_completed = []
        current_step = "initialization"
        
        # Extract user_id from query context for logging (if available)
        user_id = self._extract_user_id_from_context(user_context) or "anonymous"
        
        try:
            print(f"🏥 Expert Council Session {session_id} starting...")
            
            # === SYSTEM CHECK (FAIL-FAST) ===
            current_step = "system_check"
            system_check_step = ExpertCouncilStep("system_check", "Verify all required services are operational")
            system_check_step.started_at = datetime.utcnow()
            
            if not await self._system_health_check():
                system_check_step.success = False
                system_check_step.error = "Critical services unavailable"
                system_check_step.completed_at = datetime.utcnow()
                steps_completed.append(system_check_step)
                
                raise ExpertCouncilError(
                    "System health check failed - critical services unavailable", 
                    "system_check",
                    "system_unavailable"
                )
            
            system_check_step.success = True
            system_check_step.completed_at = datetime.utcnow()
            system_check_step.duration_seconds = (system_check_step.completed_at - system_check_step.started_at).total_seconds()
            steps_completed.append(system_check_step)
            
            # === STEP 1: COORDINATOR PLANNING ===
            current_step = "coordinator_planning"
            planning_step = ExpertCouncilStep("coordinator_planning", "Strategic analysis and expert consultation planning")
            planning_step.started_at = datetime.utcnow()
            
            coordinator_analysis = await self._coordinator_planning(query, user_context, rag_context)
            
            planning_step.success = True
            planning_step.output = coordinator_analysis[:200] + "..." if len(coordinator_analysis) > 200 else coordinator_analysis
            planning_step.completed_at = datetime.utcnow()
            planning_step.duration_seconds = (planning_step.completed_at - planning_step.started_at).total_seconds()
            steps_completed.append(planning_step)
            
            # === STEP 2: EVIDENCE GATHERING ===
            current_step = "evidence_gathering"
            evidence_step = ExpertCouncilStep("evidence_gathering", "Collect medical evidence and specialist insights")
            evidence_step.started_at = datetime.utcnow()
            
            medical_evidence = await self._gather_medical_evidence(query, user_context, rag_context)
            
            evidence_step.success = True
            evidence_step.output = f"Gathered {len(medical_evidence.get('sources', []))} evidence sources"
            evidence_step.completed_at = datetime.utcnow()
            evidence_step.duration_seconds = (evidence_step.completed_at - evidence_step.started_at).total_seconds()
            steps_completed.append(evidence_step)
            
            # === STEP 3: COMPLEX REASONING ===
            current_step = "complex_reasoning"
            reasoning_step = ExpertCouncilStep("complex_reasoning", "Advanced diagnostic reasoning and pattern analysis")
            reasoning_step.started_at = datetime.utcnow()
            
            complex_analysis = await self._complex_reasoning(query, coordinator_analysis, medical_evidence)
            
            reasoning_step.success = True
            reasoning_step.output = complex_analysis[:200] + "..." if len(complex_analysis) > 200 else complex_analysis
            reasoning_step.completed_at = datetime.utcnow()
            reasoning_step.duration_seconds = (reasoning_step.completed_at - reasoning_step.started_at).total_seconds()
            steps_completed.append(reasoning_step)
            
            # === STEP 4: SAFETY CRITIQUE ===
            current_step = "safety_critique"
            critique_step = ExpertCouncilStep("safety_critique", "Clinical safety review and risk assessment")
            critique_step.started_at = datetime.utcnow()
            
            safety_critique = await self._safety_critique(query, complex_analysis, medical_evidence)
            
            critique_step.success = True
            critique_step.output = safety_critique[:200] + "..." if len(safety_critique) > 200 else safety_critique
            critique_step.completed_at = datetime.utcnow()
            critique_step.duration_seconds = (critique_step.completed_at - critique_step.started_at).total_seconds()
            steps_completed.append(critique_step)
            
            # === STEP 5: CONSENSUS SYNTHESIS ===
            current_step = "consensus_synthesis"
            synthesis_step = ExpertCouncilStep("consensus_synthesis", "Synthesize expert opinions into unified assessment")
            synthesis_step.started_at = datetime.utcnow()
            
            consensus_result = await self._consensus_synthesis(
                query, coordinator_analysis, complex_analysis, safety_critique, medical_evidence
            )
            
            synthesis_step.success = True
            synthesis_step.output = "Consensus reached with confidence assessment"
            synthesis_step.completed_at = datetime.utcnow()
            synthesis_step.duration_seconds = (synthesis_step.completed_at - synthesis_step.started_at).total_seconds()
            steps_completed.append(synthesis_step)
            
            # === STEP 6: STRUCTURED OUTPUT GENERATION ===
            current_step = "structured_output"
            output_step = ExpertCouncilStep("structured_output", "Generate structured analysis and interactive components")
            output_step.started_at = datetime.utcnow()
            
            structured_output = await self._generate_structured_output(consensus_result, medical_evidence)
            
            output_step.success = True
            output_step.output = "Generated structured analysis with interactive components"
            output_step.completed_at = datetime.utcnow()
            output_step.duration_seconds = (output_step.completed_at - output_step.started_at).total_seconds()
            steps_completed.append(output_step)
            
            # === STEP 7: USER RESPONSE FORMATTING ===
            current_step = "user_response"
            response_step = ExpertCouncilStep("user_response", "Format final response for user presentation")
            response_step.started_at = datetime.utcnow()
            
            user_response = await self._format_user_response(consensus_result, structured_output)
            
            response_step.success = True
            response_step.output = "User response formatted successfully"
            response_step.completed_at = datetime.utcnow()
            response_step.duration_seconds = (response_step.completed_at - response_step.started_at).total_seconds()
            steps_completed.append(response_step)
            
            # === SUCCESS: BUILD COMPLETE RESULT ===
            end_time = time.time()
            total_duration = end_time - start_time
            
            council_result = {
                "session_id": session_id,
                "success": True,
                "user_response": user_response,
                "confidence": structured_output.get("confidence_assessment", {}).get("overall_confidence", 0.7),
                "structured_analysis": structured_output.get("structured_analysis", {}),
                "interactive_components": structured_output.get("interactive_components", {}),
                "reasoning_trace": {
                    "coordinator_analysis": coordinator_analysis,
                    "medical_evidence": medical_evidence,
                    "complex_analysis": complex_analysis,
                    "safety_critique": safety_critique,
                    "consensus_result": consensus_result
                },
                "metadata": {
                    "session_id": session_id,
                    "experts_consulted": ["coordinator", "medical_expert", "complex_reasoner", "safety_critic"],
                    "evidence_sources": medical_evidence.get("sources", []),
                    "workflow": "medagent_pro_structured_v3",
                    "models_used": ["gemini-2.0-flash-exp", "gemini-2.5-flash-exp", "medgemma-4b"]
                },
                "duration_seconds": total_duration,
                "step_breakdown": {step.step_name: step.duration_seconds for step in steps_completed}
            }
            
            print(f"✅ Expert Council Session {session_id} completed successfully in {total_duration:.2f}s")
            
        except ExpertCouncilError as ece:
            # Handle known Expert Council errors
            end_time = time.time()
            total_duration = end_time - start_time
            
            print(f"❌ Expert Council Session {session_id} failed at step: {ece.step}")
            
            council_result = {
                "session_id": session_id,
                "success": False,
                "error_type": ece.error_type,
                "failed_step": ece.step,
                "error_message": ece.message,
                "suggestion": self._get_error_suggestion(ece.error_type),
                "user_response": self._generate_error_response(ece),
                "duration_seconds": total_duration,
                "step_breakdown": {step.step_name: step.duration_seconds for step in steps_completed},
                "metadata": {
                    "session_id": session_id,
                    "workflow": "medagent_pro_structured_v3",
                    "failed_at_step": current_step
                }
            }
            
        except Exception as e:
            # Handle unexpected errors
            end_time = time.time()
            total_duration = end_time - start_time
            
            print(f"💥 Expert Council Session {session_id} crashed with unexpected error: {str(e)}")
            
            council_result = {
                "session_id": session_id,
                "success": False,
                "error_type": "unexpected_error",
                "failed_step": current_step,
                "error_message": f"Unexpected error: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support.",
                "user_response": "I apologize, but I encountered an unexpected technical issue. Please try your question again.",
                "duration_seconds": total_duration,
                "step_breakdown": {step.step_name: step.duration_seconds for step in steps_completed},
                "metadata": {
                    "session_id": session_id,
                    "workflow": "medagent_pro_structured_v3",
                    "crashed_at_step": current_step
                }
            }
            
        finally:
            # === GUARANTEED SESSION LOGGING ===
            # This block ALWAYS executes regardless of success or failure
            if council_result:  # Only log if we have some result
                try:
                    success = await self.pm.log_council_session(
                        session_id=session_id,
                        user_id=user_id,
                        original_query=query,
                        council_result=council_result
                    )
                    
                    if success:
                        print(f"📊 Session {session_id} logged successfully to Firestore")
                    else:
                        print(f"⚠️ Failed to log session {session_id} to Firestore")
                        
                except Exception as log_error:
                    print(f"🚨 Critical: Session logging failed for {session_id}: {log_error}")
                    # Note: We don't re-raise here to avoid masking the original error
            else:
                print(f"⚠️ No council result to log for session {session_id}")
        
        return council_result
    

    def _extract_user_id_from_context(self, user_context: str) -> Optional[str]:
        """Extract user ID from context string - improved extraction"""
        if not user_context:
            return None
            
        try:
            # Method 1: Look for user_id in the context string
            if "user_id:" in user_context.lower():
                lines = user_context.split('\n')
                for line in lines:
                    if "user_id:" in line.lower():
                        return line.split(':')[1].strip()
            
            # Method 2: Extract from "User: user123" format
            if "User:" in user_context:
                lines = user_context.split('\n')
                for line in lines:
                    if line.strip().startswith('User:'):
                        return line.split(':')[1].strip()
            
            # Method 3: Look for any user identifier patterns
            import re
            user_patterns = [
                r'user_id[:\s]+([^\s\n]+)',
                r'User[:\s]+([^\s\n]+)',
                r'user[:\s]+([^\s\n]+)'
            ]
            
            for pattern in user_patterns:
                match = re.search(pattern, user_context, re.IGNORECASE)
                if match:
                    return match.group(1)
                    
        except Exception as e:
            print(f"Error extracting user_id: {e}")
        
        return None
    
    async def _system_health_check(self) -> bool:
        """Check if all required services are operational"""
        try:
            # Check AI server availability
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                ai_server_response = await client.get("http://ai_server:9000/health")
                if ai_server_response.status_code != 200:
                    print("❌ AI server health check failed")
                    return False
            
            # Check Gemini clients
            if not self.gemini_enabled:
                print("❌ Gemini clients not available")
                return False
            
            print("✅ System health check passed")
            return True
            
        except Exception as e:
            print(f"❌ System health check failed: {e}")
            return False
    
    async def _coordinator_planning(self, query: str, user_context: str, rag_context: str) -> str:
        """Step 1: Strategic analysis and planning by coordinator"""
        
        planning_prompt = f"""You are the Lead Medical Coordinator for AURA's Expert Council. Your role is to provide strategic analysis and guide the consultation process.

PATIENT QUERY: {query}
USER CONTEXT: {user_context}
MEDICAL KNOWLEDGE: {rag_context[:800]}

TASK: Provide a comprehensive strategic analysis that will guide our expert consultation. Include:

1. **Clinical Priority Assessment**: Urgency level and key concerns
2. **Consultation Strategy**: Which medical specialties should be involved
3. **Evidence Requirements**: What types of evidence to prioritize
4. **Risk Factors**: Potential complications or red flags to monitor
5. **Initial Hypothesis**: Preliminary diagnostic considerations

Respond with a clear, structured analysis that our expert team can build upon."""

        try:
            response = self.coordinator_client.generate_content(planning_prompt)
            return response.text
        except Exception as e:
            raise ExpertCouncilError(f"Coordinator planning failed: {str(e)}", "coordinator_planning")
    
    async def _gather_medical_evidence(self, query: str, user_context: str, rag_context: str) -> Dict[str, Any]:
        """Step 2: Gather evidence from specialist tools and knowledge base"""
        
        evidence = {
            "sources": [],
            "specialist_insights": {},
            "knowledge_base_results": rag_context
        }
        
        try:
            # Get MedGemma analysis
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                medgemma_response = await client.post(
                    "http://ai_server:9000/ai/wellness",
                    json={"message": f"As a medical specialist, analyze this case: {query}. Provide clinical insights."}
                )
                
                if medgemma_response.status_code == 200:
                    medgemma_data = medgemma_response.json()
                    evidence["specialist_insights"]["medical_expert"] = medgemma_data.get("response", "")
                    evidence["sources"].append("MedGemma Medical Analysis")
            
            # Add RAG context as evidence source
            if rag_context:
                evidence["sources"].append("Medical Knowledge Base")
            
            return evidence
            
        except Exception as e:
            raise ExpertCouncilError(f"Evidence gathering failed: {str(e)}", "evidence_gathering")
    
    async def _complex_reasoning(self, query: str, coordinator_analysis: str, medical_evidence: Dict) -> str:
        """Step 3: Advanced diagnostic reasoning"""
        
        reasoning_prompt = f"""You are an Advanced Medical Reasoning Specialist with expertise in complex diagnostic analysis.

CASE PRESENTATION: {query}

COORDINATOR'S STRATEGIC ANALYSIS:
{coordinator_analysis}

MEDICAL EVIDENCE GATHERED:
{json.dumps(medical_evidence, indent=2)}

TASK: Perform advanced diagnostic reasoning. Analyze:

1. **Differential Diagnosis**: Multiple possible conditions with probability weighting
2. **Pattern Recognition**: Clinical patterns and their significance  
3. **Risk Stratification**: Categorize risks from low to high priority
4. **Diagnostic Confidence**: Level of certainty for each consideration
5. **Recommended Actions**: Next steps for diagnosis/management

Provide sophisticated medical reasoning that integrates all available evidence."""

        try:
            response = self.reasoner_client.generate_content(reasoning_prompt)
            return response.text
        except Exception as e:
            raise ExpertCouncilError(f"Complex reasoning failed: {str(e)}", "complex_reasoning")
    
    async def _safety_critique(self, query: str, complex_analysis: str, medical_evidence: Dict) -> str:
        """Step 4: Safety review and critique"""
        
        safety_prompt = f"""You are the Chief Medical Safety Officer reviewing this Expert Council consultation.

ORIGINAL CASE: {query}

COMPLEX REASONING ANALYSIS:
{complex_analysis}

AVAILABLE EVIDENCE:
{json.dumps(medical_evidence, indent=2)}

CRITICAL SAFETY REVIEW: Evaluate this analysis for:

1. **Safety Concerns**: Any missed red flags or emergency indicators
2. **Risk Assessment**: Adequacy of risk evaluation
3. **Clinical Blind Spots**: Potential oversights in the analysis
4. **Recommendation Safety**: Are suggested actions appropriate and safe?
5. **Patient Safety Priorities**: Most critical safety considerations

Your job is to ensure patient safety above all else. Challenge any assumptions and highlight any concerns."""

        try:
            response = self.critic_client.generate_content(safety_prompt)
            return response.text
        except Exception as e:
            raise ExpertCouncilError(f"Safety critique failed: {str(e)}", "safety_critique")
    
    async def _consensus_synthesis(self, query: str, coordinator_analysis: str, 
                                 complex_analysis: str, safety_critique: str, 
                                 medical_evidence: Dict) -> Dict[str, Any]:
        """Step 5: Synthesize expert opinions into consensus"""
        
        synthesis_prompt = f"""You are the Expert Council Synthesizer. Integrate all expert opinions into a unified assessment.

CASE: {query}

EXPERT OPINIONS:
1. COORDINATOR: {coordinator_analysis}
2. COMPLEX REASONER: {complex_analysis}  
3. SAFETY CRITIC: {safety_critique}

EVIDENCE: {json.dumps(medical_evidence, indent=2)}

SYNTHESIS TASK: Create a unified expert consensus addressing:

1. **Primary Assessment**: Most likely explanation/diagnosis
2. **Confidence Level**: Team confidence (0.0-1.0) with justification
3. **Key Recommendations**: Prioritized action items
4. **Safety Priorities**: Critical safety considerations
5. **Follow-up Needs**: What additional evaluation may be needed

Respond ONLY with a JSON object in this exact format:
{{
    "primary_assessment": "string",
    "confidence_level": 0.85,
    "key_recommendations": ["item1", "item2", "item3"],
    "safety_priorities": ["priority1", "priority2"],
    "follow_up_needs": ["need1", "need2"],
    "consensus_reasoning": "explanation of how we reached this consensus"
}}"""

        try:
            response = self.coordinator_client.generate_content(synthesis_prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            raise ExpertCouncilError(f"Consensus synthesis failed: {str(e)}", "consensus_synthesis")
    
    async def _generate_structured_output(self, consensus_result: Dict, medical_evidence: Dict) -> Dict[str, Any]:
        """Step 6: Generate structured analysis and interactive components"""
        
        return {
            "structured_analysis": {
                "clinical_summary": {
                    "primary_assessment": consensus_result.get("primary_assessment", "Assessment pending"),
                    "key_findings": consensus_result.get("key_recommendations", []),
                    "safety_considerations": consensus_result.get("safety_priorities", [])
                },
                "differential_diagnoses": [
                    {
                        "condition": consensus_result.get("primary_assessment", "Primary consideration"),
                        "probability": "High",
                        "reasoning": consensus_result.get("consensus_reasoning", "Based on expert analysis")
                    }
                ],
                "recommendations": {
                    "immediate_actions": consensus_result.get("key_recommendations", [])[:3],
                    "follow_up_care": consensus_result.get("follow_up_needs", []),
                    "monitoring": ["Monitor symptoms", "Follow up with healthcare provider"]
                },
                "safety_considerations": {
                    "red_flags": consensus_result.get("safety_priorities", []),
                    "when_to_seek_help": ["If symptoms worsen", "If new symptoms develop"]
                }
            },
            "interactive_components": {
                "summary_card": {
                    "title": "Expert Council Assessment",
                    "confidence": consensus_result.get("confidence_level", 0.7),
                    "primary_finding": consensus_result.get("primary_assessment", "Assessment complete")
                },
                "diagnoses_panel": {
                    "primary_consideration": consensus_result.get("primary_assessment", ""),
                    "confidence_score": consensus_result.get("confidence_level", 0.7)
                },
                "action_checklist": {
                    "items": consensus_result.get("key_recommendations", [])
                },
                "safety_alerts": {
                    "priority_items": consensus_result.get("safety_priorities", [])
                }
            },
            "confidence_assessment": {
                "overall_confidence": consensus_result.get("confidence_level", 0.7),
                "confidence_factors": [
                    "Multiple expert review",
                    "Evidence-based analysis", 
                    "Safety validation"
                ]
            }
        }
    
    async def _format_user_response(self, consensus_result: Dict, structured_output: Dict) -> str:
        """Step 7: Format final user response"""
        
        confidence = consensus_result.get("confidence_level", 0.7)
        assessment = consensus_result.get("primary_assessment", "Assessment completed")
        recommendations = consensus_result.get("key_recommendations", [])
        
        response = f"""**Expert Council Assessment**

Our medical expert team has carefully analyzed your situation. Here's our assessment:

**Primary Finding:** {assessment}

**Key Recommendations:**"""
        
        for i, rec in enumerate(recommendations[:3], 1):
            response += f"\n{i}. {rec}"
        
        response += f"""

**Confidence Level:** {confidence:.0%} - This assessment is based on collaborative analysis by multiple medical AI specialists.

**Important:** This analysis is for informational purposes and should complement, not replace, professional medical consultation."""

        return response
    
    def _get_error_suggestion(self, error_type: str) -> str:
        """Get helpful suggestion based on error type"""
        suggestions = {
            "system_unavailable": "Please try again in a few moments. If the issue persists, start with basic consultation.",
            "coordinator_planning": "Please rephrase your question and try again.",
            "evidence_gathering": "Try simplifying your question or check your internet connection.",
            "complex_reasoning": "The system is experiencing high demand. Please try again shortly.",
            "safety_critique": "Please try again or contact support if the issue persists.",
            "consensus_synthesis": "Please try again with a more specific question.",
            "structured_output": "Please try again or use the basic consultation option.",
            "user_response": "Please try again or contact support."
        }
        return suggestions.get(error_type, "Please try again or contact support if the issue persists.")
    
    def _generate_error_response(self, error: ExpertCouncilError) -> str:
        """Generate user-friendly error response"""
        if error.error_type == "system_unavailable":
            return "I'm experiencing some technical difficulties with our expert consultation system. Please try starting with a basic consultation, or try again in a few moments."
        else:
            return "I apologize, but I encountered an issue during the expert consultation. Please try rephrasing your question or starting with our basic consultation option."

# Global instance
expert_council = ExpertCouncil()