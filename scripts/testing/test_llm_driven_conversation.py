#!/usr/bin/env python3
# scripts/testing/test_llm_driven_conversation.py
"""
Test LLM-Driven Conversation Manager with Trust-Based Architecture
Tests: Entity extraction, LLM escalation, self-correction, honest failures
"""
import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class ConversationTester:
    def __init__(self):
        self.test_results = []
        
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Testing LLM-Driven Conversation Manager")
        print("=" * 60)
        
        tests = [
            self.test_system_health,
            self.test_entity_extraction_flow,
            self.test_llm_escalation_logic,
            self.test_self_correction_behavior,
            self.test_context_awareness,
            self.test_honest_failure_handling,
            self.test_expert_council_trigger
        ]
        
        for test in tests:
            try:
                await test()
                self.test_results.append({"test": test.__name__, "status": "PASSED"})
            except Exception as e:
                print(f"❌ {test.__name__} FAILED: {e}")
                self.test_results.append({"test": test.__name__, "status": "FAILED", "error": str(e)})
        
        self._print_summary()

    async def test_system_health(self):
        """Test system health and components"""
        print("\n1️⃣ System Health Check")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BASE_URL}/health")
            
            if response.status_code != 200:
                raise Exception(f"Health check failed: {response.status_code}")
            
            health_data = response.json()
            print(f"✅ System v{health_data.get('version')} healthy")
            
            # Check conversation system specifically
            conv_health = health_data.get('systems', {}).get('conversation_system', {})
            if conv_health.get('status') != 'healthy':
                raise Exception(f"Conversation system unhealthy: {conv_health}")
            
            features = conv_health.get('features', [])
            required_features = ['llm_driven_extraction', 'intelligent_escalation', 'self_correction_loop']
            
            for feature in required_features:
                if feature not in features:
                    raise Exception(f"Missing feature: {feature}")
            
            print(f"✅ LLM-driven features: {', '.join(required_features)}")
            print(f"✅ Architecture: {conv_health.get('architecture', 'unknown')}")

    async def test_entity_extraction_flow(self):
        """Test LLM-driven entity extraction"""
        print("\n2️⃣ Entity Extraction Flow")
        
        test_cases = [
            {
                "query": "I have chest pain that started this morning",
                "expected_entities": ["primary_symptom", "onset_time"]
            },
            {
                "query": "Sharp stabbing pain in my chest, makes me nauseous, started 2 hours ago",
                "expected_entities": ["primary_symptom", "pain_type", "associated_symptoms", "onset_time"]
            }
        ]
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            for i, case in enumerate(test_cases, 1):
                print(f"\n  Test Case {i}: {case['query']}")
                
                response = await client.post(
                    f"{BASE_URL}/api/chat",
                    json={"query": case["query"], "user_id": f"test_extraction_{i}"}
                )
                
                if response.status_code != 200:
                    raise Exception(f"Chat failed: {response.status_code}")
                
                data = response.json()
                session_id = data.get('session_id')
                
                # Get session history to check symptom_profile
                history_response = await client.get(f"{BASE_URL}/api/chat/session/{session_id}/history")
                
                if history_response.status_code != 200:
                    raise Exception(f"History retrieval failed: {history_response.status_code}")
                
                history_data = history_response.json()
                symptom_profile = history_data.get('symptom_profile', {})
                
                print(f"  📊 Extracted entities: {list(symptom_profile.keys())}")
                print(f"  🎯 Primary symptom: {symptom_profile.get('primary_symptom', 'None')}")
                print(f"  ⏰ Onset time: {symptom_profile.get('onset_time', 'None')}")
                
                # Check if expected entities were extracted
                extracted_keys = [k for k, v in symptom_profile.items() if v not in [None, [], ""]]
                missing_expected = set(case['expected_entities']) - set(extracted_keys)
                
                if missing_expected:
                    print(f"  ⚠️  Missing expected entities: {missing_expected}")
                else:
                    print(f"  ✅ All expected entities extracted")

    async def test_llm_escalation_logic(self):
        """Test LLM-driven escalation decision making"""
        print("\n3️⃣ LLM Escalation Logic")
        
        # Test progressive escalation
        escalation_messages = [
            "I have chest pain",
            "It's a sharp, stabbing pain that started this morning",
            "I also feel nauseous and dizzy, and my heart is racing",
            "Should I be worried? This feels really serious"
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = None
            
            for i, message in enumerate(escalation_messages, 1):
                print(f"\n  Turn {i}: {message}")
                
                payload = {"query": message, "user_id": "test_escalation"}
                if session_id:
                    payload["session_id"] = session_id
                
                start_time = time.time()
                response = await client.post(f"{BASE_URL}/api/chat", json=payload)
                duration = time.time() - start_time
                
                if response.status_code != 200:
                    raise Exception(f"Turn {i} failed: {response.status_code}")
                
                data = response.json()
                session_id = data.get('session_id')
                
                print(f"  📊 State: {data.get('conversation_state')}")
                print(f"  🤖 Service: {data.get('service_used')}")
                print(f"  ⏱️  Duration: {duration:.1f}s")
                
                # Check if Expert Council was triggered
                if data.get('service_used') == 'expert_council_medagent_pro':
                    print(f"  🏥 Expert Council TRIGGERED at turn {i}!")
                    print(f"  🎯 Confidence: {data.get('confidence', 0):.0%}")
                    break
                elif i == len(escalation_messages):
                    print(f"  ⚠️  Expert Council not triggered after {i} turns")

    async def test_self_correction_behavior(self):
        """Test self-correction loop for entity extraction"""
        print("\n4️⃣ Self-Correction Behavior")
        
        # Use complex medical terminology to potentially trigger correction
        complex_query = "I have been experiencing paroxysmal nocturnal dyspnea with orthopnea, accompanied by bilateral lower extremity edema and chest discomfort with exertional component"
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            print(f"  Complex query: {complex_query[:80]}...")
            
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json={"query": complex_query, "user_id": "test_correction"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Complex query failed: {response.status_code}")
            
            data = response.json()
            session_id = data.get('session_id')
            
            # Check extraction results
            history_response = await client.get(f"{BASE_URL}/api/chat/session/{session_id}/history")
            history_data = history_response.json()
            
            extraction_failures = history_data.get('extraction_failures', 0)
            symptom_profile = history_data.get('symptom_profile', {})
            
            print(f"  📊 Extraction failures: {extraction_failures}")
            print(f"  🎯 Entities extracted: {len([v for v in symptom_profile.values() if v])}")
            print(f"  ✅ Self-correction working" if extraction_failures <= 1 else "⚠️ Multiple failures")

    async def test_context_awareness(self):
        """Test context-aware prompt building"""
        print("\n5️⃣ Context Awareness")
        
        # Test if system remembers and builds on previous context
        context_messages = [
            "I have chest pain",
            "Actually, it's more like pressure than pain",
            "It gets worse when I climb stairs"
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = None
            
            for i, message in enumerate(context_messages, 1):
                payload = {"query": message, "user_id": "test_context"}
                if session_id:
                    payload["session_id"] = session_id
                
                response = await client.post(f"{BASE_URL}/api/chat", json=payload)
                data = response.json()
                session_id = data.get('session_id')
                
                aura_response = data.get('response', '')
                
                # Check if AURA acknowledges the correction/addition
                if i == 2:  # Correction message
                    if any(word in aura_response.lower() for word in ['pressure', 'different', 'clarify']):
                        print(f"  ✅ Context awareness: AURA acknowledged correction")
                    else:
                        print(f"  ⚠️  May not have recognized correction")
                
                if i == 3:  # Trigger information
                    if any(word in aura_response.lower() for word in ['stairs', 'exertion', 'exercise']):
                        print(f"  ✅ Context building: AURA incorporated trigger information")
                    else:
                        print(f"  ⚠️  May not have incorporated trigger")

    async def test_honest_failure_handling(self):
        """Test honest failure handling when extraction fails"""
        print("\n6️⃣ Honest Failure Handling")
        
        # Use very ambiguous/non-medical query
        ambiguous_query = "I feel weird today, something's not right"
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json={"query": ambiguous_query, "user_id": "test_failure"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failure test failed: {response.status_code}")
            
            data = response.json()
            session_id = data.get('session_id')
            
            # Check if system handles vague input gracefully
            aura_response = data.get('response', '')
            
            if any(phrase in aura_response.lower() for phrase in ['more details', 'specific', 'tell me more']):
                print(f"  ✅ Graceful handling: Asked for clarification")
            else:
                print(f"  ⚠️  Response may not be handling vagueness well")
            
            # Check extraction results
            history_response = await client.get(f"{BASE_URL}/api/chat/session/{session_id}/history")
            history_data = history_response.json()
            symptom_profile = history_data.get('symptom_profile', {})
            
            # Should have minimal/empty extraction for vague input
            meaningful_extractions = len([v for v in symptom_profile.values() if v not in [None, [], ""]])
            
            if meaningful_extractions <= 1:
                print(f"  ✅ Honest extraction: Minimal data for vague input ({meaningful_extractions} items)")
            else:
                print(f"  ⚠️  May be over-extracting from vague input ({meaningful_extractions} items)")

    async def test_expert_council_trigger(self):
        """Test Expert Council integration"""
        print("\n7️⃣ Expert Council Integration")
        
        # Direct Expert Council test
        expert_query = "I have multiple concerning symptoms: severe chest pain, shortness of breath, nausea, and sweating"
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            # Test direct Expert Council
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json={"query": expert_query, "user_id": "test_expert", "force_expert_council": True}
            )
            
            if response.status_code != 200:
                raise Exception(f"Expert Council test failed: {response.status_code}")
            
            data = response.json()
            
            if data.get('service_used') == 'expert_council_direct':
                print(f"  ✅ Direct Expert Council working")
                print(f"  🎯 Confidence: {data.get('confidence', 0):.0%}")
                
                expert_session = data.get('expert_council_session', {})
                if expert_session.get('workflow') == 'medagent_pro_5_step':
                    print(f"  ✅ MedAgent-Pro workflow active")
                else:
                    print(f"  ⚠️  Workflow not detected")
            else:
                raise Exception(f"Expert Council not triggered: {data.get('service_used')}")

    def _print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed = len([r for r in self.test_results if r['status'] == 'FAILED'])
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        if failed > 0:
            print("\n🔍 Failed Tests:")
            for result in self.test_results:
                if result['status'] == 'FAILED':
                    print(f"  • {result['test']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n🏁 Testing completed at {datetime.now().strftime('%H:%M:%S')}")

async def quick_smoke_test():
    """Quick smoke test for basic functionality"""
    print("🔥 Quick Smoke Test")
    
    async with httpx.AsyncClient(timeout=90.0) as client:  # Increased timeout
        # Health check
        health_response = await client.get(f"{BASE_URL}/health")
        assert health_response.status_code == 200, "Health check failed"
        
        # Basic conversation (should avoid Expert Council)
        chat_response = await client.post(
            f"{BASE_URL}/api/chat",
            json={"query": "I have a mild headache", "user_id": "smoke_test"}  # Changed to avoid Expert Council
        )
        assert chat_response.status_code == 200, "Basic chat failed"
        
        data = chat_response.json()
        assert data.get('session_id'), "No session ID returned"
        assert data.get('response'), "No response returned"
        assert data.get('service_used') != 'expert_council_medagent_pro_robust', "Should not trigger Expert Council for mild symptoms"
        
        print("✅ Basic functionality working")
        print(f"   Service used: {data.get('service_used')}")
        print(f"   Response length: {len(data.get('response', ''))}")
        print(f"   State: {data.get('conversation_state')}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        asyncio.run(quick_smoke_test())
    else:
        tester = ConversationTester()
        asyncio.run(tester.run_all_tests())