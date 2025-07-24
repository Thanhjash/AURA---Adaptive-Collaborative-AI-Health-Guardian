# test_medgemma.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Set custom cache directory
os.environ['HF_HOME'] = '/mnt/d/3.Project/AURA-Health-Guardian/.hf_cache'

def test_medgemma():
    """Test MedGemma 4-bit model loading and inference"""
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("🔬 Loading MedGemma 4-bit model...")
    
    model_name = "unsloth/medgemma-4b-it-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print(f"✅ Model loaded on: {model.device}")
    print(f"📊 Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Test inference
    prompt = "What are the symptoms of ventricular arrhythmia?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🩺 Response: {response}")
    
    return model, tokenizer

if __name__ == "__main__":
    test_medgemma()