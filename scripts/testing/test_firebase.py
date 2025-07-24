#!/usr/bin/env python3
"""Test Firebase personalization integration"""
import asyncio
import sys
import os
sys.path.insert(0, '/app')

from lib.personalization_manager import PersonalizationManager

async def test_firebase():
    """Test Firebase integration"""
    try:
        pm = PersonalizationManager()
        print("🔥 Testing Firebase Integration")
        
        # Health check
        health = await pm.health_check()
        print(f"✅ Health: {health['status']}")
        
        # Create test user
        user_id = "test_user_123"
        profile = await pm.create_user_profile(user_id, name="Test User")
        print(f"✅ Created profile: {profile['profile']['name']}")
        
        # Log interaction
        await pm.log_interaction(
            user_id, 
            "What is arrhythmia?", 
            "Explained heart rhythm",
            "rag_system", 
            0.85
        )
        print("✅ Logged interaction")
        
        # Get context
        context = await pm.get_user_context(user_id)
        print(f"✅ Context length: {len(context)}")
        
        print("🎉 Firebase integration working!")
        
    except Exception as e:
        print(f"❌ Firebase test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_firebase())