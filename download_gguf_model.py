# download_gguf_model.py
import os
from huggingface_hub import hf_hub_download

def download_medgemma_gguf():
    """Download MedGemma GGUF model to cache directory"""
    
    cache_dir = "/mnt/d/3.Project/AURA-Health-Guardian/.hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("📥 Downloading MedGemma GGUF model...")
    print("⏰ This will take 5-10 minutes (~2.5GB)")
    
    try:
        model_path = hf_hub_download(
            repo_id="unsloth/medgemma-4b-it-GGUF",
            filename="medgemma-4b-it-Q4_K_M.gguf",
            cache_dir=cache_dir,
            resume_download=True  # Resume if interrupted
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"📍 Location: {model_path}")
        print(f"📊 Size: {os.path.getsize(model_path) / (1024**3):.2f} GB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    download_medgemma_gguf()