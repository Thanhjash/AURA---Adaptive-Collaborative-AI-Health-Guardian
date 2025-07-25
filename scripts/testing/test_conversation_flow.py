# scripts/testing/test_conversation_flow.py
"""
Quick test script for Progressive Consultation Flow
Tests Firebase integration and conversation state management
"""
import asyncio
import httpx
import json

async def test_progressive_conversation():
    """Test the progressive conversation flow"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Progressive Consultation Flow...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Health Check
        print("\n1️⃣ Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"✅ Health check: {response.status_code}")
            health_data = response.json()
            print(f"   Conversation system: {health_data.get('systems', {}).get('conversation_system', {}).get('status', 'unknown')}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Test 2: Start new conversation
        print("\n2️⃣ Starting new conversation...")
        try:
            payload = {
                "query": "I have chest pain",
                "user_id": "test_user_simple"
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ New conversation started")
                print(f"   Session ID: {data.get('session_id', 'N/A')}")
                print(f"   State: {data.get('conversation_state', 'N/A')}")
                print(f"   Service: {data.get('service_used', 'N/A')}")
                print(f"   Response length: {len(data.get('response', ''))}")
                
                session_id = data.get('session_id')
                
                # Test 3: Continue conversation if session_id exists
                if session_id:
                    print(f"\n3️⃣ Continuing conversation with session: {session_id}")
                    
                    continue_payload = {
                        "query": "It started this morning and is getting worse",
                        "user_id": "test_user_simple", 
                        "session_id": session_id
                    }
                    
                    response2 = await client.post(
                        f"{base_url}/api/chat",
                        json=continue_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        print(f"✅ Conversation continued")
                        print(f"   State: {data2.get('conversation_state', 'N/A')}")
                        print(f"   Service: {data2.get('service_used', 'N/A')}")
                        print(f"   Escalation signals: {data2.get('escalation_signals', {})}")
                    else:
                        print(f"❌ Continue conversation failed: {response2.status_code}")
                        print(f"   Error: {response2.text}")
                
            else:
                print(f"❌ New conversation failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Conversation test failed: {e}")
        
        # Test 4: Direct Expert Council (should work)
        print(f"\n4️⃣ Testing direct Expert Council...")
        try:
            expert_payload = {
                "query": "Multiple symptoms analysis needed",
                "user_id": "test_user_expert",
                "force_expert_council": True
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=expert_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Expert Council direct access working")
                print(f"   Service: {data.get('service_used', 'N/A')}")
                print(f"   Flow type: {data.get('flow_type', 'N/A')}")
            else:
                print(f"❌ Expert Council failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Expert Council test failed: {e}")

    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_progressive_conversation())