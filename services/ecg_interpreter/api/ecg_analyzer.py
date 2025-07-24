# api/ecg_analyzer.py
import torch
import numpy as np
from typing import Dict
from .schemas import ArrhythmiaAnalysisResponse, AnalysisScope, ArrhythmiaClassification, MechanisticExplanation, ClinicalAssessmentSignature

class ECGArrhythmiaAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device
        
        # AAMI class mapping
        self.class_mapping = {
            0: ("N", "Normal Beat"),
            1: ("S", "Supraventricular Ectopic Beat"),
            2: ("V", "Ventricular Ectopic Beat"),
            3: ("F", "Fusion Beat"),
        }
    
    def run_single_ecg_analysis(self, ecg_tensor: torch.Tensor) -> Dict:
        """Main analysis function - returns structured JSON for API"""
        
        with torch.no_grad():
            # Ensure correct input shape [1, 1, 300]
            if ecg_tensor.dim() == 1:
                ecg_tensor = ecg_tensor.unsqueeze(0).unsqueeze(0)
            elif ecg_tensor.dim() == 2:
                ecg_tensor = ecg_tensor.unsqueeze(0)
            
            ecg_tensor = ecg_tensor.to(self.device)
            
            # Forward pass
            outputs = self.model(ecg_tensor)
            interpret_data = self.model.get_interpretability_data()
            
            # Classification
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction_idx = torch.max(probs, dim=1)
            
            pred_idx = prediction_idx.item()
            aami_class, class_desc = self.class_mapping.get(pred_idx, ("Unknown", "Unknown classification"))
            
            # Mechanistic analysis
            mechanistic_data = self._extract_mechanistic_data(interpret_data)
            clinical_signature = self._compute_clinical_signature(interpret_data, mechanistic_data)
            
            # Build response
            analysis_result = {
                "analysis_scope": AnalysisScope(),
                "arrhythmia_classification": ArrhythmiaClassification(
                    aami_class=aami_class,
                    class_description=class_desc,
                    confidence=confidence.item()
                ),
                "mechanistic_explanation": MechanisticExplanation(
                    global_order_parameter=mechanistic_data['global_order_parameter'],
                    energy=mechanistic_data['energy'],
                    group_coherence=mechanistic_data['group_coherence']
                ),
                "clinical_assessment_signature": ClinicalAssessmentSignature(**clinical_signature)
            }
            
            return analysis_result
    
    def _extract_mechanistic_data(self, interpret_data: Dict) -> Dict:
        """Extract mechanistic insights from model interpretability data"""
        
        # Get final oscillator states
        final_states = interpret_data['oscillator_states'][0]  # Shape: [Time, M, N]
        final_phases = torch.atan2(final_states[:, :, 1], final_states[:, :, 0]).mean(dim=0)  # [M]
        
        # Global order parameter
        global_order_param = torch.abs(torch.mean(torch.exp(1j * final_phases))).item()
        
        # Energy calculation (Kuramoto energy)
        J_tensor = interpret_data['learned_parameters']['coupling_matrix']
        phase_diff = final_phases.unsqueeze(1) - final_phases.unsqueeze(0)
        energy = -torch.sum(J_tensor * torch.cos(phase_diff)) / J_tensor.shape[0]
        
        # Group coherence (P, QRS, T waves)
        p_phases = final_phases[0:8]    # P-wave oscillators
        qrs_phases = final_phases[8:24] # QRS complex oscillators  
        t_phases = final_phases[24:32]  # T-wave oscillators
        
        group_coherence = {
            "p_wave_group": torch.abs(torch.mean(torch.exp(1j * p_phases))).item(),
            "qrs_group": torch.abs(torch.mean(torch.exp(1j * qrs_phases))).item(),
            "t_wave_group": torch.abs(torch.mean(torch.exp(1j * t_phases))).item()
        }
        
        return {
            'global_order_parameter': global_order_param,
            'energy': energy.item(),
            'group_coherence': group_coherence
        }
    
    def _compute_clinical_signature(self, interpret_data: Dict, mechanistic_data: Dict) -> Dict:
        """Compute clinical assessment signature from mechanistic data"""
        
        # Extract coupling matrix
        J_abs = torch.abs(interpret_data['learned_parameters']['coupling_matrix']).cpu().numpy()
        
        # Sync strength (global order parameter)
        sync_strength = mechanistic_data['global_order_parameter']
        
        # Coupling integrity (intra-group vs inter-group coupling strength)
        p_intra = np.mean(J_abs[0:8, 0:8])
        qrs_intra = np.mean(J_abs[8:24, 8:24])
        t_intra = np.mean(J_abs[24:32, 24:32])
        inter_group = np.mean(J_abs[0:8, 8:24])  # P-QRS interaction
        
        coupling_integrity = np.clip(
            (p_intra + qrs_intra + t_intra) / (p_intra + qrs_intra + t_intra + inter_group + 1e-6),
            0, 1
        )
        
        # Network heterogeneity
        network_heterogeneity = np.clip(
            np.std(J_abs) / (np.mean(J_abs) + 1e-6) / 2.0, 0, 1
        )
        
        # Sync stability (consistency of group coherence)
        group_coherences = list(mechanistic_data['group_coherence'].values())
        sync_stability = np.clip(1 - np.std(group_coherences) * 2.5, 0, 1)
        
        # Frequency stability (average group coherence)
        frequency_stability = np.mean(group_coherences)
        
        return {
            'sync_strength': float(sync_strength),
            'sync_stability': float(sync_stability),
            'coupling_integrity': float(coupling_integrity),
            'network_heterogeneity': float(network_heterogeneity),
            'frequency_stability': float(frequency_stability)
        }