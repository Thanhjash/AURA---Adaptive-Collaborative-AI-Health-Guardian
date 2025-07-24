# services/aura_main/lib/personalization_manager.py
"""
Personalization Manager for AURA - Handles user profiles and interaction history
"""
import firebase_admin
from firebase_admin import credentials, db
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging
import uuid

logger = logging.getLogger(__name__)

class PersonalizationManager:
    """Manages user personalization using Firebase Realtime Database"""
    
    def __init__(self):
        """Initialize Firebase connection with AURA credentials"""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        self.database_url = os.getenv("FIREBASE_DATABASE_URL")
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.service_account_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': self.database_url
                })
            
            self.db_ref = db.reference()
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
    
    async def create_user_profile(
        self, 
        user_id: str, 
        name: Optional[str] = None,
        communication_preference: str = "friendly"
    ) -> Dict[str, Any]:
        """Create new user profile"""
        
        profile_data = {
            "profile": {
                "created_at": datetime.utcnow().isoformat(),
                "name": name or f"User_{user_id[:8]}",
                "communication_preference": communication_preference,
                "onboarding_complete": False
            },
            "health_summary": {
                "conditions": [],
                "allergies": [],
                "medications": [],
                "last_updated": datetime.utcnow().isoformat()
            },
            "interaction_history": {},
            "preferences": {
                "language": "en",
                "voice_enabled": False,
                "privacy_level": "standard"
            }
        }
        
        try:
            self.db_ref.child('users').child(user_id).set(profile_data)
            logger.info(f"Created profile for user: {user_id}")
            return profile_data
            
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile"""
        try:
            profile = self.db_ref.child('users').child(user_id).get()
            if profile:
                logger.info(f"Retrieved profile for user: {user_id}")
                return profile
            else:
                logger.info(f"No profile found for user: {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    async def log_interaction(
        self,
        user_id: str,
        query: str,
        response_summary: str,
        service_used: str,
        confidence: float,
        session_id: Optional[str] = None
    ) -> str:
        """Log user interaction"""
        
        interaction_id = str(uuid.uuid4())
        interaction_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response_summary": response_summary,
            "service_used": service_used,
            "confidence": confidence,
            "session_id": session_id or "default"
        }
        
        try:
            self.db_ref.child('users').child(user_id).child('interaction_history').child(interaction_id).set(interaction_data)
            logger.info(f"Logged interaction {interaction_id} for user: {user_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            return ""
    
    async def get_user_context(self, user_id: str) -> str:
        """Get formatted user context for LLM prompt"""
        
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "No user profile available."
        
        # Extract key information
        user_info = profile.get('profile', {})
        health_summary = profile.get('health_summary', {})
        
        context_parts = []
        
        # User preferences
        comm_pref = user_info.get('communication_preference', 'friendly')
        context_parts.append(f"Communication style: {comm_pref}")
        
        # Health context
        conditions = health_summary.get('conditions', [])
        if conditions:
            context_parts.append(f"Known conditions: {', '.join(conditions)}")
        
        allergies = health_summary.get('allergies', [])
        if allergies:
            context_parts.append(f"Allergies: {', '.join(allergies)}")
        
        # Recent interactions (last 3)
        interactions = profile.get('interaction_history', {})
        if interactions:
            recent = sorted(
                interactions.values(), 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )[:3]
            
            if recent:
                context_parts.append("Recent topics discussed:")
                for interaction in recent:
                    context_parts.append(f"- {interaction.get('query', 'N/A')}")
        
        if context_parts:
            return "User Context:\n" + "\n".join(context_parts)
        else:
            return "New user - no previous context available."
    
    async def health_check(self) -> Dict[str, Any]:
        """Check personalization system health"""
        try:
            # Test database connection
            test_ref = self.db_ref.child('health_check')
            test_ref.set({"status": "ok", "timestamp": datetime.utcnow().isoformat()})
            
            # Count users
            users = self.db_ref.child('users').get()
            user_count = len(users) if users else 0
            
            return {
                "status": "healthy",
                "database_connection": "ok",
                "total_users": user_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }