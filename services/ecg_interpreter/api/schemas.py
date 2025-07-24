# api/schemas.py
from pydantic import BaseModel
from typing import Dict, List

class AnalysisScope(BaseModel):
    type: str = "Arrhythmia Classification per Beat"
    source_dataset: str = "MIT-BIH Arrhythmia Database (AAMI 4-Class)"
    disclaimer: str = "This analysis identifies arrhythmia types for a single heartbeat and is not a comprehensive cardiac diagnosis."

class ArrhythmiaClassification(BaseModel):
    aami_class: str
    class_description: str
    confidence: float

class MechanisticExplanation(BaseModel):
    diagnostic_strategy: str = "Pathological Hypersynchronization"
    global_order_parameter: float
    energy: float
    group_coherence: Dict[str, float]

class ClinicalAssessmentSignature(BaseModel):
    sync_strength: float
    sync_stability: float
    coupling_integrity: float
    network_heterogeneity: float
    frequency_stability: float

class ArrhythmiaAnalysisResponse(BaseModel):
    analysis_scope: AnalysisScope
    arrhythmia_classification: ArrhythmiaClassification
    mechanistic_explanation: MechanisticExplanation
    clinical_assessment_signature: ClinicalAssessmentSignature