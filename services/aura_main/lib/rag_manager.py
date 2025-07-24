# services/aura_main/lib/rag_manager.py
"""
RAG Manager for AURA - Handles knowledge retrieval from Supabase
"""
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages retrieval-augmented generation using Supabase vector database"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def semantic_search(
        self, 
        query: str, 
        similarity_threshold: float = 0.3,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search against medical knowledge base
        
        Args:
            query: User's health question
            similarity_threshold: Minimum cosine similarity (0-1)
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge chunks with metadata
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embedding_model.encode(query)
            
            # Perform vector similarity search
            response = self.supabase.rpc(
                'match_medical_knowledge',
                {
                    'query_embedding': query_embedding.tolist(),
                    'similarity_threshold': similarity_threshold,
                    'match_count': max_results
                }
            ).execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} relevant knowledge chunks")
                return response.data
            else:
                logger.warning("No relevant knowledge found")
                return []
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    async def get_context_for_query(self, query: str) -> str:
        """
        Get formatted context string for LLM prompt
        
        Args:
            query: User's health question
            
        Returns:
            Formatted context string for prompt injection
        """
        knowledge_chunks = await self.semantic_search(query)
        
        if not knowledge_chunks:
            return "No specific medical knowledge found for this query."
        
        context_parts = []
        for i, chunk in enumerate(knowledge_chunks, 1):
            context_parts.append(
                f"**Source {i}: {chunk.get('title', 'Medical Knowledge')}**\n"
                f"{chunk.get('content', '')}\n"
                f"(Source: {chunk.get('source', 'Unknown')}, "
                f"Relevance: {chunk.get('similarity', 0):.2f})\n"
            )
        
        return "\n".join(context_parts)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check RAG system health"""
        try:
            # Test database connection
            response = self.supabase.table('medical_knowledge').select('id').limit(1).execute()
            
            # Test embedding model
            test_embedding = self.embedding_model.encode("test query")
            
            return {
                "status": "healthy",
                "database_connection": "ok",
                "embedding_model": "ok",
                "knowledge_entries": len(response.data) if response.data else 0,
                "embedding_dimension": len(test_embedding)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }