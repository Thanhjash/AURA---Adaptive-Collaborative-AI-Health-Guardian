# scripts/data_ingestion/ingest_medical_data.py
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np

# Use same credentials
SUPABASE_URL = "https://ppyzqyffglucswrqgtja.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBweXpxeWZmZ2x1Y3N3cnFndGphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMyNDMwNTMsImV4cCI6MjA2ODgxOTA1M30.Bq9E6LplCrfAZsx-vHmNyUbwx0uWE5eNJlhWiS3niQU"

# Sample medical knowledge for arrhythmia
SAMPLE_KNOWLEDGE = [
    {
        "title": "Normal Sinus Rhythm",
        "content": "Normal sinus rhythm (NSR) is characterized by a heart rate of 60-100 beats per minute, regular R-R intervals, and normal P-wave morphology. The PR interval is typically 120-200ms, and QRS duration is less than 120ms.",
        "source": "Cardiology Textbook",
        "metadata": {"type": "rhythm_classification", "aami_class": "N"}
    },
    {
        "title": "Ventricular Ectopic Beats", 
        "content": "Ventricular ectopic beats (VEBs) originate from the ventricles and are characterized by wide QRS complexes (>120ms), bizarre morphology, and absence of preceding P-waves. They can be unifocal or multifocal.",
        "source": "ECG Interpretation Guide",
        "metadata": {"type": "rhythm_classification", "aami_class": "V"}
    },
    {
        "title": "Supraventricular Ectopic Beats",
        "content": "Supraventricular ectopic beats arise above the ventricles, typically showing narrow QRS complexes and may have abnormal P-wave morphology. They include atrial premature complexes and junctional beats.",
        "source": "Arrhythmia Manual",
        "metadata": {"type": "rhythm_classification", "aami_class": "S"}
    }
]

def ingest_knowledge():
    """Ingest sample medical knowledge into Supabase"""
    
    # Initialize clients
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
    
    print("🔬 Loading embedding model...")
    
    for i, knowledge in enumerate(SAMPLE_KNOWLEDGE):
        print(f"📝 Processing: {knowledge['title']}")
        
        # Generate embedding
        embedding = model.encode(knowledge['content'])
        
        # Insert into database
        data = {
            "content": knowledge['content'],
            "title": knowledge['title'], 
            "source": knowledge['source'],
            "chunk_index": i,
            "embedding": embedding.tolist(),
            "metadata": knowledge['metadata']
        }
        
        response = supabase.table('medical_knowledge').insert(data).execute()
        print(f"✅ Inserted: {knowledge['title']}")
    
    print("🎉 Knowledge ingestion complete!")

if __name__ == "__main__":
    ingest_knowledge()