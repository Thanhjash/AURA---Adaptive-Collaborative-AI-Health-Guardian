#!/usr/bin/env python3
# scripts/testing/test_progressive_conversation.py
"""
Test Progressive Conversation Flow with Session ID Extraction
Tests the fixed context passing implementation
"""
import asyncio
import httpx
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_progressive_conversation():
    """Test the complete progressive conversation flow"""
    
    print("🧪 Testing AURA Progressive Conversation Flow")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Test 1: Health Check
        print("\n1️⃣ Health Check...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ System healthy - v{health_data.get('version', 'unknown')}")
                print(f"   Fix status: {health_data.get('fix_status', 'N/A')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return
        
        # Test 2: Start New Conversation
        print("\n2️⃣ Starting new conversation...")
        
        start_payload = {
            "query": "I have chest pain",
            "user_id": "test_user_progressive"
        }
        
        try:
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json=start_payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract session_id from response
                session_id = data.get('session_id')
                
                print(f"✅ Conversation started ({end_time - start_time:.1f}s)")
                print(f"   Session ID: {session_id}")
                print(f"   State: {data.get('conversation_state')}")
                print(f"   Service: {data.get('service_used')}")
                print(f"   Context provided: {data.get('context_provided', {})}")
                print(f"   Response preview: {data.get('response', '')[:100]}...")
                
                if not session_id:
                    print("❌ No session_id returned")
                    return
                
                # Test 3: Continue Conversation
                print(f"\n3️⃣ Continuing conversation with session: {session_id[:20]}...")
                
                continue_payload = {
                    "query": "It started this morning and feels sharp, like a stabbing pain",
                    "user_id": "test_user_progressive",
                    "session_id": session_id  # Use actual session_id
                }
                
                start_time = time.time()
                response2 = await client.post(
                    f"{BASE_URL}/api/chat",
                    json=continue_payload,
                    headers={"Content-Type": "application/json"}
                )
                end_time = time.time()
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    
                    print(f"✅ Conversation continued ({end_time - start_time:.1f}s)")
                    print(f"   State: {data2.get('conversation_state')}")
                    print(f"   Service: {data2.get('service_used')}")
                    print(f"   Context score: {data2.get('context_score', 'N/A')}")
                    print(f"   Escalation signals: {data2.get('escalation_signals', {})}")
                    print(f"   Context provided: {data2.get('context_provided', {})}")
                    print(f"   Response preview: {data2.get('response', '')[:100]}...")
                    
                    # Test 4: Get Session History
                    print(f"\n4️⃣ Getting session history...")
                    
                    history_response = await client.get(f"{BASE_URL}/api/chat/session/{session_id}/history")
                    
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        print(f"✅ Session history retrieved")
                        print(f"   Total messages: {history_data.get('message_count', 0)}")
                        print(f"   Final state: {history_data.get('conversation_state')}")
                        print(f"   Escalation triggers: {history_data.get('escalation_triggers', [])}")
                        
                        # Show conversation flow
                        messages = history_data.get('messages', [])
                        print(f"\n📝 Conversation Flow:")
                        for i, msg in enumerate(messages[-4:], 1):  # Last 4 messages
                            role = "👤 User" if msg.get('role') == 'user' else "🤖 AURA"
                            content = msg.get('content', '')[:60] + "..." if len(msg.get('content', '')) > 60 else msg.get('content', '')
                            service = f" ({msg.get('service', '')})" if msg.get('service') else ""
                            print(f"   {i}. {role}: {content}{service}")
                    else:
                        print(f"❌ History retrieval failed: {history_response.status_code}")
                        
                    # Test 5: Third Turn to Test State Progression
                    print(f"\n5️⃣ Third conversation turn...")
                    
                    third_payload = {
                        "query": "I also feel a bit nauseous and dizzy. Should I be worried?",
                        "user_id": "test_user_progressive", 
                        "session_id": session_id
                    }
                    
                    start_time = time.time()
                    response3 = await client.post(
                        f"{BASE_URL}/api/chat",
                        json=third_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    end_time = time.time()
                    
                    if response3.status_code == 200:
                        data3 = response3.json()
                        
                        print(f"✅ Third turn completed ({end_time - start_time:.1f}s)")
                        print(f"   State: {data3.get('conversation_state')}")
                        print(f"   Service: {data3.get('service_used')}")
                        print(f"   Escalation: {data3.get('escalation_signals', {}).get('should_escalate', False)}")
                        
                        # Check if Expert Council was triggered
                        if data3.get('service_used') == 'expert_council_medagent_pro':
                            print(f"🏥 Expert Council TRIGGERED!")
                            print(f"   Confidence: {data3.get('confidence', 0):.0%}")
                            print(f"   Session ID: {data3.get('expert_council_session', {}).get('session_id', 'N/A')}")
                        else:
                            print(f"📋 Still in progressive consultation")
                            
                    else:
                        print(f"❌ Third turn failed: {response3.status_code}")
                        print(f"   Error: {response3.text}")
                        
                else:
                    print(f"❌ Continue conversation failed: {response2.status_code}")
                    try:
                        error_data = response2.json()
                        print(f"   Error: {error_data.get('error', 'Unknown error')}")
                        print(f"   Suggestion: {error_data.get('suggestion', 'N/A')}")
                    except:
                        print(f"   Raw error: {response2.text}")
                        
            else:
                print(f"❌ Start conversation failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Conversation test failed: {e}")
        
        # Test 6: Direct Expert Council (for comparison)
        print(f"\n6️⃣ Testing direct Expert Council...")
        
        expert_payload = {
            "query": "I have multiple symptoms: chest pain, nausea, and dizziness",
            "user_id": "test_user_expert",
            "force_expert_council": True
        }
        
        try:
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json=expert_payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Direct Expert Council completed ({end_time - start_time:.1f}s)")
                print(f"   Service: {data.get('service_used')}")
                print(f"   Flow type: {data.get('flow_type')}")
                print(f"   Confidence: {data.get('confidence', 0):.0%}")
            else:
                print(f"❌ Expert Council failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Expert Council test failed: {e}")

    print(f"\n🏁 Test completed at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)

def run_simple_test():
    """Simple synchronous test using requests"""
    import requests
    
    print("🧪 Simple Progressive Conversation Test")
    
    # Start conversation
    start_response = requests.post(
        f"{BASE_URL}/api/chat",
        json={"query": "I have chest pain", "user_id": "test_simple"},
        timeout=30
    )
    
    if start_response.status_code == 200:
        start_data = start_response.json()
        session_id = start_data.get('session_id')
        print(f"✅ Started: {session_id}")
        
        # Continue conversation
        continue_response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "query": "It's a sharp pain in my chest",
                "user_id": "test_simple",
                "session_id": session_id
            },
            timeout=30
        )
        
        if continue_response.status_code == 200:
            continue_data = continue_response.json()
            print(f"✅ Continued: {continue_data.get('conversation_state')}")
            print(f"   Context provided: {continue_data.get('context_provided', {})}")
        else:
            print(f"❌ Continue failed: {continue_response.status_code}")
    else:
        print(f"❌ Start failed: {start_response.status_code}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        run_simple_test()
    else:
        asyncio.run(test_progressive_conversation())