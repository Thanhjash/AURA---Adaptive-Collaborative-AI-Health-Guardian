# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
import io
import os
from models.otsn_models import create_interpretable_otsn_model
from api.ecg_analyzer import ECGArrhythmiaAnalyzer
from api.schemas import ArrhythmiaAnalysisResponse

app = FastAPI(title="AURA ECG Arrhythmia Interpreter", version="1.0.0")

# Global variables for model
MODEL = None
ANALYZER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def load_model():
    """Load OTSN model on startup"""
    global MODEL, ANALYZER
    
    try:
        print("🔬 Loading OTSN Arrhythmia Model...")
        MODEL = create_interpretable_otsn_model()
        
        # Load trained weights
        weights_path = "models/weights/best_otsn_model.pth"
        if os.path.exists(weights_path):
            print(f"📥 Loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove _orig_mod. prefix from compiled model
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                print("📝 Removing _orig_mod. prefix from compiled model weights")
                state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
            
            MODEL.load_state_dict(state_dict)
            
            print("✅ Trained weights loaded successfully")
        else:
            print("⚠️  No weights file found - using untrained model")
        
        MODEL.eval()
        ANALYZER = ECGArrhythmiaAnalyzer(MODEL, DEVICE)
        print(f"✅ Model loaded successfully on {DEVICE}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

@app.get("/")
async def root():
    return {"service": "AURA ECG Arrhythmia Interpreter", "status": "operational", "scope": "AAMI 4-class arrhythmia detection"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}

def preprocess_ecg_data(file_content: bytes) -> torch.Tensor:
    """Preprocess uploaded ECG data to required format"""
    try:
        # Handle different file formats
        if file_content.startswith(b'RIFF'):  # WAV file
            raise HTTPException(400, "WAV files not supported yet")
        
        # Assume text/CSV format for now
        data_str = file_content.decode('utf-8')
        
        # Parse ECG values
        if ',' in data_str:
            ecg_values = [float(x.strip()) for x in data_str.strip().split(',')]
        else:
            ecg_values = [float(x.strip()) for x in data_str.strip().split()]
        
        # Convert to numpy and normalize
        ecg_array = np.array(ecg_values, dtype=np.float32)
        
        # Ensure length is 300 (MIT-BIH standard beat length)
        if len(ecg_array) != 300:
            if len(ecg_array) > 300:
                # Truncate to 300
                ecg_array = ecg_array[:300]
            else:
                # Pad with zeros
                ecg_array = np.pad(ecg_array, (0, 300 - len(ecg_array)), 'constant')
        
        # Normalize (Z-score normalization)
        ecg_array = (ecg_array - np.mean(ecg_array)) / (np.std(ecg_array) + 1e-8)
        
        # Convert to tensor [1, 1, 300]
        ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return ecg_tensor
        
    except Exception as e:
        raise HTTPException(400, f"Failed to preprocess ECG data: {str(e)}")

@app.post("/analyze", response_model=ArrhythmiaAnalysisResponse)
async def analyze_ecg(file: UploadFile = File(...)):
    """Analyze ECG for arrhythmia classification"""
    
    if MODEL is None or ANALYZER is None:
        raise HTTPException(503, "Model not loaded. Service unavailable.")
    
    try:
        # Read and preprocess file
        file_content = await file.read()
        ecg_tensor = preprocess_ecg_data(file_content)
        
        # Run analysis
        analysis_result = ANALYZER.run_single_ecg_analysis(ecg_tensor)
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if MODEL is None:
        return {"status": "Model not loaded"}
    
    return {
        "model_type": "InterpretableEnhancedOTSNModel",
        "scope": "Arrhythmia Classification (AAMI 4-class)",
        "classes": ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusion)"],
        "input_format": "Single ECG beat (300 samples)",
        "dataset": "MIT-BIH Arrhythmia Database",
        "device": str(DEVICE),
        "parameters": sum(p.numel() for p in MODEL.parameters()) if MODEL else 0
    }