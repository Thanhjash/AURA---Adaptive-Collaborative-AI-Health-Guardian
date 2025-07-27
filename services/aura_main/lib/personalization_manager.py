# services/aura_main/lib/personalization_manager.py
"""
Enhanced Personalization Manager for AURA - Firestore with advanced features
Improvements: Async operations, efficient queries, granular privacy controls, council session observability
FIXED: Timezone-aware datetime handling
"""
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone  # ADDED timezone import
import uuid
import logging
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from google.oauth2 import service_account
import json

logger = logging.getLogger(__name__)

class PersonalizationManager:
    """Enhanced personalization with Firestore and advanced privacy controls"""
    
    def __init__(self):
        """Initialize Firestore with service account"""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        self._initialize_firestore()
    
    def _initialize_firestore(self):
        """Initialize async Firestore client"""
        try:
            if self.service_account_path and os.path.exists(self.service_account_path):
                # Load service account and extract project_id
                with open(self.service_account_path, 'r') as f:
                    service_account_info = json.load(f)
                    
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info
                )
                
                self.project_id = service_account_info['project_id']
                self.db = firestore.AsyncClient(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                raise FileNotFoundError(f"Service account file not found: {self.service_account_path}")
                
            logger.info("Firestore initialized successfully")
            
        except Exception as e:
            logger.error(f"Firestore initialization failed: {e}")
            raise
    
    async def create_user_profile(
        self, 
        user_id: str, 
        name: Optional[str] = None,
        communication_preference: str = "friendly"
    ) -> Dict[str, Any]:
        """Create new user profile with enhanced privacy controls"""
        
        profile_data = {
            "profile": {
                "created_at": datetime.now(timezone.utc),  # FIXED
                "name": name or f"User_{user_id[:8]}",
                "communication_preference": communication_preference,
                "onboarding_complete": False,
                "last_active": datetime.now(timezone.utc)  # FIXED
            },
            "health_summary": {
                "conditions": [],
                "allergies": [],
                "medications": [],
                "last_updated": datetime.now(timezone.utc)  # FIXED
            },
            "preferences": {
                "language": "en",
                "voice_enabled": False,
                "consent": {
                    "use_interaction_history": True,
                    "use_health_summary": True,
                    "allow_proactive_outreach": False,
                    "data_retention_days": 365
                },
                "privacy_level": "standard"
            },
            "metadata": {
                "version": "2.0",
                "total_interactions": 0,
                "last_context_update": datetime.now(timezone.utc)  # FIXED
            }
        }
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            await user_ref.set(profile_data)
            
            logger.info(f"Created enhanced profile for user: {user_id}")
            return profile_data
            
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile efficiently"""
        try:
            user_ref = self.db.collection('users').document(user_id)
            doc = await user_ref.get()
            
            if doc.exists:
                profile = doc.to_dict()
                
                # Update last active timestamp
                await user_ref.update({
                    "profile.last_active": datetime.now(timezone.utc)  # FIXED
                })
                
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
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log interaction to subcollection for efficient querying"""
        
        # Check user consent first
        profile = await self.get_user_profile(user_id)
        if not profile:
            return ""
            
        consent = profile.get('preferences', {}).get('consent', {})
        if not consent.get('use_interaction_history', True):
            logger.info(f"Interaction logging disabled for user: {user_id}")
            return ""
        
        interaction_id = str(uuid.uuid4())
        interaction_data = {
            "timestamp": datetime.now(timezone.utc),  # FIXED
            "query": query,
            "response_summary": response_summary,
            "service_used": service_used,
            "confidence": confidence,
            "session_id": session_id or "default",
            "metadata": metadata or {}
        }
        
        try:
            # Store in subcollection for efficient querying
            interactions_ref = self.db.collection('users').document(user_id).collection('interactions')
            await interactions_ref.document(interaction_id).set(interaction_data)
            
            # Update user metadata
            user_ref = self.db.collection('users').document(user_id)
            await user_ref.update({
                "metadata.total_interactions": firestore.Increment(1),
                "metadata.last_interaction": datetime.now(timezone.utc)  # FIXED
            })
            
            logger.info(f"Logged interaction {interaction_id} for user: {user_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            return ""
    
    async def get_user_context(self, user_id: str) -> str:
        """Get efficient user context with consent-aware filtering"""
        
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "No user profile available."
        
        # Check consent for different data types
        consent = profile.get('preferences', {}).get('consent', {})
        
        context_parts = []
        
        # User preferences (always available)
        user_info = profile.get('profile', {})
        comm_pref = user_info.get('communication_preference', 'friendly')
        context_parts.append(f"Communication style: {comm_pref}")
        
        # Health context (consent-based)
        if consent.get('use_health_summary', True):
            health_summary = profile.get('health_summary', {})
            
            conditions = health_summary.get('conditions', [])
            if conditions:
                context_parts.append(f"Known conditions: {', '.join(conditions)}")
            
            allergies = health_summary.get('allergies', [])
            if allergies:
                context_parts.append(f"Allergies: {', '.join(allergies)}")
        
        # Recent interactions (consent-based, efficient query)
        if consent.get('use_interaction_history', True):
            recent_interactions = await self._get_recent_interactions(user_id, limit=3)
            
            if recent_interactions:
                context_parts.append("Recent topics discussed:")
                for interaction in recent_interactions:
                    context_parts.append(f"- {interaction.get('query', 'N/A')}")
        
        if context_parts:
            return "User Context:\n" + "\n".join(context_parts)
        else:
            return "New user - no previous context available."
    
    async def _get_recent_interactions(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Efficiently get recent interactions using Firestore subcollection query"""
        try:
            interactions_ref = (
                self.db.collection('users')
                .document(user_id)
                .collection('interactions')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            
            docs = await interactions_ref.get()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to get recent interactions: {e}")
            return []
    
    async def update_health_summary(
        self, 
        user_id: str, 
        conditions: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        medications: Optional[List[str]] = None
    ) -> bool:
        """Update user health summary with consent checking"""
        
        profile = await self.get_user_profile(user_id)
        if not profile:
            return False
            
        consent = profile.get('preferences', {}).get('consent', {})
        if not consent.get('use_health_summary', True):
            logger.warning(f"Health summary updates disabled for user: {user_id}")
            return False
        
        update_data = {
            "health_summary.last_updated": datetime.now(timezone.utc)  # FIXED
        }
        
        if conditions is not None:
            update_data["health_summary.conditions"] = conditions
        if allergies is not None:
            update_data["health_summary.allergies"] = allergies
        if medications is not None:
            update_data["health_summary.medications"] = medications
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            await user_ref.update(update_data)
            logger.info(f"Updated health summary for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update health summary: {e}")
            return False
    
    async def update_user_consent(
        self, 
        user_id: str, 
        consent_updates: Dict[str, Any]
    ) -> bool:
        """Update user consent preferences with validation"""
        
        valid_consent_keys = {
            'use_interaction_history', 
            'use_health_summary', 
            'allow_proactive_outreach',
            'data_retention_days'
        }
        
        # Validate consent keys
        invalid_keys = set(consent_updates.keys()) - valid_consent_keys
        if invalid_keys:
            logger.warning(f"Invalid consent keys: {invalid_keys}")
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            
            # Build update dictionary with proper Firestore paths
            update_data = {}
            for key, value in consent_updates.items():
                update_data[f"preferences.consent.{key}"] = value
            
            update_data["metadata.last_consent_update"] = datetime.now(timezone.utc)  # FIXED
            
            await user_ref.update(update_data)
            logger.info(f"Updated consent for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update consent: {e}")
            return False
    
    async def get_user_data_export(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for transparency/privacy compliance"""
        
        try:
            # Get main profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {}
            
            # Get all interactions
            interactions_ref = (
                self.db.collection('users')
                .document(user_id)
                .collection('interactions')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
            )
            
            interaction_docs = await interactions_ref.get()
            interactions = [doc.to_dict() for doc in interaction_docs]
            
            export_data = {
                "profile": profile,
                "interactions": interactions,
                "export_timestamp": datetime.now(timezone.utc),  # FIXED
                "total_interactions": len(interactions)
            }
            
            logger.info(f"Generated data export for user: {user_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export user data: {e}")
            return {}
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Completely delete user data (privacy compliance)"""
        
        try:
            # Delete all interactions first
            interactions_ref = (
                self.db.collection('users')
                .document(user_id)
                .collection('interactions')
            )
            
            # Batch delete interactions
            interaction_docs = await interactions_ref.get()
            batch = self.db.batch()
            
            for doc in interaction_docs:
                batch.delete(doc.reference)
            
            await batch.commit()
            
            # Delete main profile
            user_ref = self.db.collection('users').document(user_id)
            await user_ref.delete()
            
            logger.info(f"Deleted all data for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False
    
    async def cleanup_expired_data(self) -> int:
        """Clean up data based on retention policies"""
        
        cleaned_count = 0
        
        try:
            # Get all users
            users_ref = self.db.collection('users')
            user_docs = await users_ref.get()
            
            for user_doc in user_docs:
                user_data = user_doc.to_dict()
                consent = user_data.get('preferences', {}).get('consent', {})
                retention_days = consent.get('data_retention_days', 365)
                
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)  # FIXED
                
                # Clean old interactions
                interactions_ref = (
                    user_doc.reference
                    .collection('interactions')
                    .where('timestamp', '<', cutoff_date)
                )
                
                old_interactions = await interactions_ref.get()
                
                if old_interactions:
                    batch = self.db.batch()
                    for interaction_doc in old_interactions:
                        batch.delete(interaction_doc.reference)
                    
                    await batch.commit()
                    cleaned_count += len(old_interactions)
            
            logger.info(f"Cleaned up {cleaned_count} expired interactions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            return 0
    
    async def log_interaction_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Log user feedback for specific interaction - enables Human-in-the-Loop learning
        Creates Golden Dataset for future model improvements
        """
        
        valid_feedback_types = {"rating", "correction", "helpful", "accuracy", "suggestion"}
        
        try:
            # Validate feedback structure
            if not any(key in feedback for key in valid_feedback_types):
                logger.warning(f"Invalid feedback structure: {feedback}")
                return False
            
            # Prepare feedback data with timestamp
            feedback_data = {
                "feedback": feedback,
                "feedback_timestamp": datetime.now(timezone.utc),  # FIXED
                "feedback_version": "1.0"
            }
            
            # Update specific interaction with feedback
            interaction_ref = (
                self.db.collection('users')
                .document(user_id)
                .collection('interactions')
                .document(interaction_id)
            )
            
            await interaction_ref.update(feedback_data)
            
            # Log to Golden Dataset collection for model training
            golden_dataset_ref = self.db.collection('golden_dataset').document()
            
            # Get original interaction for context
            interaction_doc = await interaction_ref.get()
            if interaction_doc.exists:
                interaction_data = interaction_doc.to_dict()
                
                golden_record = {
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                    "original_query": interaction_data.get("query"),
                    "original_response": interaction_data.get("response_summary"),
                    "service_used": interaction_data.get("service_used"),
                    "confidence": interaction_data.get("confidence"),
                    "feedback": feedback,
                    "timestamp": interaction_data.get("timestamp"),
                    "feedback_timestamp": datetime.now(timezone.utc),  # FIXED
                    "golden_dataset_version": "1.0"
                }
                
                await golden_dataset_ref.set(golden_record)
            
            logger.info(f"Logged feedback for interaction {interaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return False

    async def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get feedback analytics for system improvement insights"""
        
        try:
            golden_ref = self.db.collection('golden_dataset')
            feedback_docs = await golden_ref.get()
            
            if not feedback_docs:
                return {"total_feedback": 0}
            
            # Analyze feedback patterns
            total_feedback = len(feedback_docs)
            ratings = []
            service_feedback = {}
            
            for doc in feedback_docs:
                data = doc.to_dict()
                feedback = data.get("feedback", {})
                service = data.get("service_used", "unknown")
                
                # Collect ratings
                if "rating" in feedback:
                    ratings.append(feedback["rating"])
                
                # Service-specific feedback
                if service not in service_feedback:
                    service_feedback[service] = {"count": 0, "positive": 0}
                
                service_feedback[service]["count"] += 1
                
                if feedback.get("helpful", False) or feedback.get("rating", 0) >= 4:
                    service_feedback[service]["positive"] += 1
            
            # Calculate metrics
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            return {
                "total_feedback": total_feedback,
                "average_rating": round(avg_rating, 2),
                "service_performance": service_feedback,
                "feedback_collection_rate": f"{total_feedback} interactions with feedback"
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback analytics: {e}")
            return {"error": str(e)}

    # NEW: Expert Council Session Observability
    async def log_council_session(
        self, 
        session_id: str, 
        user_id: str,
        original_query: str,
        council_result: Dict[str, Any]
    ) -> bool:
        """
        Log complete Expert Council session for observability and debugging
        This is the "black box" recorder for all Expert Council decisions
        """
        
        try:
            # Prepare comprehensive session data
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "original_query": original_query,
                "timestamp": datetime.now(timezone.utc),  # FIXED
                "success": council_result.get("success", False),
                
                # Core council output
                "user_response": council_result.get("user_response", ""),
                "confidence": council_result.get("confidence", 0.0),
                
                # Structured analysis (if available)
                "structured_analysis": council_result.get("structured_analysis", {}),
                "interactive_components": council_result.get("interactive_components", {}),
                
                # Complete reasoning trace
                "reasoning_trace": council_result.get("reasoning_trace", {}),
                
                # Metadata and performance
                "metadata": council_result.get("metadata", {}),
                "duration_seconds": council_result.get("duration_seconds", 0),
                "step_breakdown": council_result.get("step_breakdown", {}),
                
                # Error handling
                "error_info": {
                    "error_type": council_result.get("error_type"),
                    "failed_step": council_result.get("failed_step"),
                    "error_message": council_result.get("error_message"),
                    "suggestion": council_result.get("suggestion")
                } if not council_result.get("success", False) else None,
                
                # System context
                "system_info": {
                    "experts_consulted": council_result.get("metadata", {}).get("experts_consulted", []),
                    "evidence_sources": council_result.get("metadata", {}).get("evidence_sources", []),
                    "workflow_version": council_result.get("metadata", {}).get("workflow", "medagent_pro_v3"),
                    "models_used": council_result.get("metadata", {}).get("models_used", [])
                }
            }
            
            # Store in council_sessions collection
            session_ref = self.db.collection('council_sessions').document(session_id)
            await session_ref.set(session_data)
            
            logger.info(f"Successfully logged council session: {session_id} (success: {session_data['success']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log council session {session_id}: {e}")
            return False
    
    async def get_council_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific council session for review/debugging"""
        
        try:
            session_ref = self.db.collection('council_sessions').document(session_id)
            doc = await session_ref.get()
            
            if doc.exists:
                session_data = doc.to_dict()
                logger.info(f"Retrieved council session: {session_id}")
                return session_data
            else:
                logger.info(f"Council session not found: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get council session {session_id}: {e}")
            return None
    
    async def get_council_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get Expert Council performance analytics"""
        
        try:
            # Query recent council sessions
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)  # FIXED
            
            sessions_ref = (
                self.db.collection('council_sessions')
                .where('timestamp', '>=', cutoff_date)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
            )
            
            session_docs = await sessions_ref.get()
            
            if not session_docs:
                return {
                    "period_days": days,
                    "total_sessions": 0,
                    "success_rate": 0.0,
                    "average_duration": 0.0,
                    "error_breakdown": {}
                }
            
            # Analyze session data
            total_sessions = len(session_docs)
            successful_sessions = 0
            total_duration = 0
            confidence_scores = []
            error_types = {}
            step_failures = {}
            
            for doc in session_docs:
                session = doc.to_dict()
                
                if session.get("success", False):
                    successful_sessions += 1
                    confidence_scores.append(session.get("confidence", 0.0))
                else:
                    # Track error patterns
                    error_type = session.get("error_info", {}).get("error_type", "unknown_error")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    failed_step = session.get("error_info", {}).get("failed_step", "unknown_step")
                    step_failures[failed_step] = step_failures.get(failed_step, 0) + 1
                
                total_duration += session.get("duration_seconds", 0)
            
            # Calculate metrics
            success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
            avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                "period_days": days,
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "success_rate": round(success_rate, 3),
                "average_duration_seconds": round(avg_duration, 2),
                "average_confidence": round(avg_confidence, 3),
                "error_breakdown": error_types,
                "step_failure_breakdown": step_failures,
                "performance_summary": {
                    "excellent": len([c for c in confidence_scores if c >= 0.8]),
                    "good": len([c for c in confidence_scores if 0.6 <= c < 0.8]),
                    "acceptable": len([c for c in confidence_scores if 0.4 <= c < 0.6]),
                    "poor": len([c for c in confidence_scores if c < 0.4])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get council analytics: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with Firestore statistics and council observability"""
        try:
            # Test connection with a simple query
            users_ref = self.db.collection('users')
            sample_query = await users_ref.limit(1).get()
            
            # Count total users (for small datasets)
            all_users = await users_ref.get()
            user_count = len(all_users)
            
            # Count council sessions
            council_ref = self.db.collection('council_sessions')
            council_docs = await council_ref.limit(100).get()  # Limit for performance
            council_count = len(council_docs)
            
            # Calculate some basic stats
            total_interactions = 0
            active_users_7d = 0
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)  # FIXED
            
            for user_doc in all_users:
                user_data = user_doc.to_dict()
                metadata = user_data.get('metadata', {})
                total_interactions += metadata.get('total_interactions', 0)
                
                last_active = user_data.get('profile', {}).get('last_active')
                if last_active and last_active > week_ago:
                    active_users_7d += 1
            
            return {
                "status": "healthy",
                "database_connection": "ok",
                "total_users": user_count,
                "total_interactions": total_interactions,
                "active_users_7d": active_users_7d,
                "council_sessions_tracked": council_count,
                "database_type": "firestore",
                "features": [
                    "efficient_querying",
                    "subcollections",
                    "granular_consent",
                    "data_export",
                    "automatic_cleanup",
                    "council_session_observability"
                ]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": "firestore"
            }

    async def get_recent_council_sessions(self, limit: int = 10, successful_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get recent council sessions with efficient server-side filtering.
        OPTIMIZED: Builds a dynamic query to let Firestore do the heavy lifting.
        """
        try:
            # Start with the base query
            query = self.db.collection('council_sessions')

            # Dynamically add filters based on parameters
            if successful_only:
                # This requires a composite index on (success, timestamp)
                query = query.where('success', '==', True)
            
            # Always order by timestamp and apply the final limit
            # This is now the exact number of documents we need
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            docs = await query.get()
            
            # The processing logic remains the same, but it now works on a smaller, pre-filtered dataset
            sessions = [
                {
                    "session_id": doc.to_dict().get("session_id"),
                    "timestamp": doc.to_dict().get("timestamp"),
                    "success": doc.to_dict().get("success", False),
                    "confidence": doc.to_dict().get("confidence", 0.0),
                    "duration_seconds": doc.to_dict().get("duration_seconds", 0),
                    "user_id": doc.to_dict().get("user_id", ""),
                    "original_query": doc.to_dict().get("original_query", "")[:100] + "..." if len(doc.to_dict().get("original_query", "")) > 100 else doc.to_dict().get("original_query", "")
                }
                for doc in docs
            ]
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get recent council sessions with server-side filtering: {e}")
            # Return empty list on failure, the endpoint will handle the HTTP response
            return []
