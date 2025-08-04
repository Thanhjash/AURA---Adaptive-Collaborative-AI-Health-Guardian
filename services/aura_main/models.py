# File: services/aura_main/models.py
# Shared Pydantic models to avoid circular dependencies

from typing import Optional
from pydantic import BaseModel

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    force_expert_council: bool = False

class SessionState(BaseModel):
    session_id: str
    user_id: str
    conversation_state: str = "INITIAL_GREETING"
    turn_count: int = 0
    symptom_profile: dict = {}
    message_history: list = []
    created_at: Optional[str] = None
    last_updated: Optional[str] = None