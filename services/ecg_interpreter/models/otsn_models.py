# models/otsn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# OTSN Configuration - matches your trained model
OTSN_CONFIG = {
    'num_classes': 4,
    'num_oscillators': 32,
    'dim': 2,
    'simulation_steps': 6,
    'freq_range': [0.5, 40],  # Updated to match your config
    'dt': 0.014,
    'cnn_dropout': 0.11,
    'fc_dropout': 0.15  # Updated to match your config
}

class InterpretableSTCNNBackbone(nn.Module):
    def __init__(self, dropout_rate=0.11):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout1d(dropout_rate)
        )
        self.skip1_proj = nn.AdaptiveAvgPool1d(18)
        self.skip2_proj = nn.AdaptiveAvgPool1d(9)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        skip1 = self.skip1_proj(out1)
        layer4_input = torch.cat([out3, skip1], dim=1)
        out4 = self.layer4(layer4_input)
        skip2 = self.skip2_proj(out2)
        layer5_input = torch.cat([out4, skip2], dim=1)
        out5 = self.layer5(layer5_input)
        
        layer_outputs = {
            'layer1': out1, 'layer2': out2, 'layer3': out3,
            'layer4': out4, 'layer5': out5
        }
        return out5, layer_outputs

class InterpretableOTSNAKOrNLayer(nn.Module):
    def __init__(self, num_oscillators, dim, freq_range, dt=0.014, input_dim=128, simulation_steps=6):
        super().__init__()
        self.M = num_oscillators
        self.N = dim
        self.dt = dt
        self.simulation_steps = simulation_steps
        assert dim == 2, f"Only supports 2D oscillators, got dim={dim}"
        
        freq = torch.logspace(torch.log10(torch.tensor(freq_range[0])), 
                             torch.log10(torch.tensor(freq_range[1])), self.M)
        self.register_parameter('omega', nn.Parameter(2 * torch.pi * freq))
        
        J_init = torch.eye(self.M) * 0.15 + torch.randn(self.M, self.M) * 0.02
        J_init = (J_init + J_init.T) / 2
        for i in range(self.M-1):
            J_init[i, i+1] = 0.1
            J_init[i+1, i] = 0.1
        self.register_parameter('J', nn.Parameter(J_init))
        
        self.input_map = nn.Sequential(
            nn.Linear(input_dim, self.M * self.N * 2),
            nn.LayerNorm(self.M * self.N * 2),
            nn.GELU(),
            nn.Linear(self.M * self.N * 2, self.M * self.N),
            nn.Tanh()
        )
        self.coupling_strength = nn.Parameter(torch.tensor(0.2))
        
        # Interpretability storage
        self.oscillator_trajectories = []
        self.phase_evolution = []
        self.coupling_activations = []
        self.frequency_deviations = []

    def forward(self, features):
        batch_size, seq_len, feature_dim = features.shape
        is_interpreting = not self.training
        
        if is_interpreting:
            self.oscillator_trajectories = []
            self.phase_evolution = []
            self.coupling_activations = []
            self.frequency_deviations = []
        
        if seq_len > 8:
            indices = torch.arange(0, seq_len, 2, device=features.device)
            features = features[:, indices]
            seq_len = features.shape[1]
        
        c_all = self.input_map(features).view(batch_size, seq_len, self.M, self.N)
        
        x = torch.randn(batch_size, self.M, self.N, device=features.device) * 0.1
        angles = torch.linspace(0, 2*torch.pi, self.M+1)[:-1].to(features.device)
        x[:, :, 0] = torch.cos(angles).unsqueeze(0)
        x[:, :, 1] = torch.sin(angles).unsqueeze(0)
        x = F.normalize(x, p=2, dim=-1)
        
        states = []
        for t_idx in range(seq_len):
            c_t = c_all[:, t_idx]
            x_init = x.clone()
            
            if is_interpreting:
                self.oscillator_trajectories.append(x.clone().detach())
            
            for step in range(self.simulation_steps):
                omega_expanded = self.omega.unsqueeze(0).unsqueeze(-1)
                omega_x = torch.stack([
                    -omega_expanded.squeeze(-1) * x[..., 1],
                    omega_expanded.squeeze(-1) * x[..., 0]
                ], dim=-1)
                
                coupling = torch.bmm(
                    self.J.unsqueeze(0).expand(batch_size, -1, -1), x
                ) * self.coupling_strength
                
                if is_interpreting:
                    self.coupling_activations.append(torch.norm(coupling, dim=-1).clone().detach())
                
                dx = omega_x + c_t + coupling
                x = x + self.dt * dx
                x = F.normalize(x, p=2, dim=-1)
            
            if is_interpreting:
                phases = torch.atan2(x[..., 1], x[..., 0])
                self.phase_evolution.append(phases.clone().detach())
                
                if len(self.phase_evolution) > 1:
                    phase_diff = phases - self.phase_evolution[-2]
                    instantaneous_freq = phase_diff / (2 * torch.pi * self.dt)
                    natural_freq = self.omega / (2 * torch.pi)
                    freq_deviation = instantaneous_freq - natural_freq.unsqueeze(0)
                    self.frequency_deviations.append(freq_deviation.clone().detach())
            
            x = 0.9 * x + 0.1 * x_init
            x = F.normalize(x, p=2, dim=-1)
            states.append(x.clone())
        
        return torch.stack(states, dim=1)

class InterpretableEnhancedOTSNModel(nn.Module):
    def __init__(self, num_classes=4, num_oscillators=32, dim=2, simulation_steps=6):
        super().__init__()
        self.cnn_backbone = InterpretableSTCNNBackbone(
            dropout_rate=OTSN_CONFIG.get('cnn_dropout', 0.11)
        )
        self.akorn_layer = InterpretableOTSNAKOrNLayer(
            num_oscillators, dim, OTSN_CONFIG.get('freq_range', [0.5, 8.0]),
            dt=OTSN_CONFIG.get('dt', 0.014), input_dim=128, simulation_steps=simulation_steps
        )
        
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(OTSN_CONFIG.get('fc_dropout', 0.2)),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Interpretability storage
        self.last_oscillator_states = None
        self.last_attention_weights = None
        self.last_cnn_features = None
        self.last_cnn_layer_outputs = None

    def forward(self, x):
        features, cnn_layer_outputs = self.cnn_backbone(x)
        
        if not self.training:
            self.last_cnn_layer_outputs = cnn_layer_outputs
            self.last_attention_weights = None
            self.last_oscillator_states = None
        
        self.last_cnn_features = features.detach()
        
        features_permuted = features.permute(0, 2, 1)
        akorn_out = self.akorn_layer(features_permuted)
        
        pooled_time = akorn_out.mean(dim=2)
        attn_weights = self.attention_pool(pooled_time)
        final_state = (pooled_time * attn_weights).sum(dim=1)
        
        if not self.training:
            self.last_oscillator_states = akorn_out.detach()
            self.last_attention_weights = attn_weights.detach()
        
        output = self.classifier(final_state)
        return output

    def get_interpretability_data(self):
        return {
            'final_cnn_features': self.last_cnn_features,
            'all_cnn_layer_outputs': self.last_cnn_layer_outputs,
            'oscillator_states': self.last_oscillator_states,
            'attention_weights': self.last_attention_weights,
            'akorn_internals': {
                'trajectories': self.akorn_layer.oscillator_trajectories,
                'phase_evolution': self.akorn_layer.phase_evolution,
                'coupling_activations': self.akorn_layer.coupling_activations,
                'frequency_deviations': self.akorn_layer.frequency_deviations
            },
            'learned_parameters': {
                'coupling_matrix': self.akorn_layer.J.detach(),
                'natural_frequencies': self.akorn_layer.omega.detach()
            }
        }

def create_interpretable_otsn_model(**kwargs):
    """Create interpretable OTSN model for API use"""
    model_params = {
        'num_classes': OTSN_CONFIG.get('num_classes', 4),
        'num_oscillators': OTSN_CONFIG.get('num_oscillators', 32),
        'dim': OTSN_CONFIG.get('dim', 2),
        'simulation_steps': OTSN_CONFIG.get('simulation_steps', 6),
    }
    model_params.update(kwargs)
    
    model = InterpretableEnhancedOTSNModel(**model_params)
    print(f"🔬 Creating INTERPRETABLE OTSN Model for Arrhythmia Analysis")
    print(f"📊 Model Statistics: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model