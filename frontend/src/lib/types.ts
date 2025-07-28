// src/lib/types/index.ts

export interface ChatRequest {
  query: string;
  user_id: string;
  session_id?: string;
  force_expert_council?: boolean;
}

export interface ChatResponse {
  response: string;
  structured_analysis?: StructuredAnalysis;
  interactive_components?: InteractiveComponents;
  service_used: string;
  confidence: number;
  user_id: string;
  session_id?: string;
  triage_analysis: TriageAnalysis;
  routing_decision?: RoutingDecision;
  expert_council_session?: ExpertCouncilSession;
  reasoning_trace?: ReasoningTrace;
  timestamp: string;
}

export interface StructuredAnalysis {
  clinical_summary: {
    primary_assessment: string;
    key_findings: string[];
    patient_context: string;
    urgency_level: "low" | "medium" | "high" | "critical";
  };
  differential_diagnoses: DiagnosisOption[];
  recommendations: {
    immediate_actions: string[];
    follow_up_care: string[];
    lifestyle_modifications: string[];
    monitoring_parameters: string[];
  };
  safety_considerations: {
    red_flags: string[];
    when_to_seek_immediate_care: string[];
    safety_score: number;
  };
  confidence_assessment: {
    overall_confidence: number;
    evidence_strength: "weak" | "moderate" | "strong";
    limitations: string[];
  };
}

export interface DiagnosisOption {
  condition: string;
  probability: number;
  reasoning: string;
  supporting_evidence: string[];
  next_steps: string[];
}

export interface InteractiveComponents {
  summary_card: {
    title: string;
    urgency_indicator: "green" | "yellow" | "orange" | "red";
    key_points: string[];
    confidence_display: number;
  };
  diagnoses_panel: {
    primary_diagnosis: DiagnosisOption;
    alternative_diagnoses: DiagnosisOption[];
    differential_reasoning: string;
  };
  action_checklist: {
    immediate_actions: ActionItem[];
    follow_up_actions: ActionItem[];
    monitoring_items: ActionItem[];
  };
  safety_alerts: {
    high_priority_alerts: SafetyAlert[];
    general_precautions: string[];
    emergency_indicators: string[];
  };
}

export interface ActionItem {
  action: string;
  priority: "high" | "medium" | "low";
  timeframe: string;
  completed?: boolean;
}

export interface SafetyAlert {
  message: string;
  severity: "warning" | "danger" | "critical";
  action_required: string;
}

export interface TriageAnalysis {
  category: "simple_chitchat" | "medical_query_low_priority" | "medical_query_high_priority" | "medical_emergency";
  confidence: number;
  reasoning: string;
  urgency_score: number;
  medical_indicators: string[];
  semantic_analysis: string;
  llm_driven: boolean;
}

export interface RoutingDecision {
  strategy: "simple_response" | "progressive_consultation" | "direct_expert_council" | "emergency_guidance";
  reason: string;
  bypass_conversation: boolean;
  ai_confidence: number;
}

export interface ExpertCouncilSession {
  session_id: string;
  experts_consulted: string[];
  evidence_sources: string[];
  workflow: string;
}

export interface ReasoningTrace {
  steps: ReasoningStep[];
  total_duration: number;
  workflow_version: string;
}

export interface ReasoningStep {
  step_number: number;
  step_name: string;
  description: string;
  input: string;
  output: string;
  confidence: number;
  duration_ms: number;
  expert_involved?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  metadata?: {
    service_used: string;
    confidence: number;
    triage_category?: string;
    structured_analysis?: StructuredAnalysis;
    interactive_components?: InteractiveComponents;
    reasoning_trace?: ReasoningTrace;
  };
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy";
  service: string;
  systems: {
    rag_system: { status: string; knowledge_entries: number };
    personalization_system: { status: string; total_users: number };
    conversation_system: { status: string };
    intelligent_triage: { status: string; model: string };
    expert_council: { status: string; models: string[]; workflow: string };
  };
  routing: {
    type: string;
    model: string;
  };
}

export interface UserProfile {
  user_id: string;
  name?: string;
  communication_preference: string;
  health_summary: {
    conditions: string[];
    allergies: string[];
    medications: string[];
  };
  preferences: {
    voice_enabled: boolean;
    consent: {
      use_interaction_history: boolean;
      use_health_summary: boolean;
    };
  };
}