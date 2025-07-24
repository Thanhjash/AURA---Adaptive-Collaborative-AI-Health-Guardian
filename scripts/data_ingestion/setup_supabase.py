# scripts/data_ingestion/setup_supabase.py
import os
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np

# Supabase credentials
SUPABASE_URL = "https://ppyzqyffglucswrqgtja.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBweXpxeWZmZ2x1Y3N3cnFndGphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMyNDMwNTMsImV4cCI6MjA2ODgxOTA1M30.Bq9E6LplCrfAZsx-vHmNyUbwx0uWE5eNJlhWiS3niQU"

def create_knowledge_table():
    """Create medical knowledge table with vector embeddings"""
    
    # SQL to create table with pgvector
    create_table_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE IF NOT EXISTS medical_knowledge (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        title VARCHAR(255),
        source VARCHAR(255),
        chunk_index INTEGER,
        embedding vector(384),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS medical_knowledge_embedding_idx 
    ON medical_knowledge USING ivfflat (embedding vector_cosine_ops);
    """
    
    print("🏗️ Creating medical_knowledge table with vector support...")
    print("Run this SQL in your Supabase SQL Editor:")
    print(create_table_sql)

def test_connection():
    """Test Supabase connection"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.table('medical_knowledge').select("*").limit(1).execute()
        print("✅ Supabase connection successful")
        return supabase
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

if __name__ == "__main__":
    create_knowledge_table()