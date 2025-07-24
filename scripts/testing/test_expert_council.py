# scripts/testing/test_expert_council.py
import asyncio
import sys
import os
sys.path.append('/mnt/d/3.Project/AURA-Health-Guardian/services/aura_main')

from core.expert_council import expert_council

async def test_expert_council():
    """Test Expert Council Protocol"""
    
    print("🏥 Testing AURA Expert Council Protocol")
    print("=" * 50)
    
    # Test case: Complex cardiac query
    query = "I have chest pain and irregular heartbeat, what could this be?"
    user_context = "35-year-old male, history of hypertension, prefers detailed explanations"
    rag_context = "Chest pain can indicate various cardiac conditions including arrhythmias, angina, or myocardial infarction."
    
    print(f"Query: {query}")
    print(f"User Context: {user_context}")
    print("\n🔄 Convening Expert Council...")
    
    try:
        result = await expert_council.run_debate(
            query=query,
            user_context=user_context,
            rag_context=rag_context
        )
        
        print("\n✅ Expert Council Session Complete!")
        print(f"Experts Consulted: {result['council_session']['experts_consulted']}")
        print(f"Consensus Reached: {result['consensus']['consensus_reached']}")
        print(f"Overall Confidence: {result['consensus']['confidence']:.2%}")
        
        print("\n📋 Expert Opinions:")
        for expert, opinion in result['expert_opinions'].items():
            status = opinion['status']
            print(f"  {expert}: {status}")
            if status == 'success':
                print(f"    Confidence: {opinion.get('confidence', 'N/A')}")
                print(f"    Opinion: {opinion.get('opinion', 'N/A')[:100]}...")
        
        print(f"\n🎯 Final Recommendation:")
        print(result['consensus']['recommendation'][:300] + "...")
        
        print(f"\n🔍 Reasoning Trace ({len(result['reasoning_trace'])} steps):")
        for step in result['reasoning_trace'][:3]:  # Show first 3 steps
            print(f"  Step {step['step']}: {step['action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Expert Council test failed: {e}")
        return False

async def test_orchestrator_integration():
    """Test full orchestrator integration"""
    import httpx
    
    print("\n🔗 Testing Orchestrator Integration")
    print("=" * 40)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test Expert Council debug endpoint
            response = await client.post(
                "http://localhost:8000/api/expert-council/debug",
                params={
                    "query": "chest pain and palpitations",
                    "user_id": "test_user"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Orchestrator Expert Council integration working!")
                print(f"Experts: {result['council_session']['experts_consulted']}")
                return True
            else:
                print(f"❌ Orchestrator test failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False

async def main():
    print("🚀 AURA Expert Council Test Suite")
    print("==================================")
    
    # Test 1: Expert Council Core
    council_success = await test_expert_council()
    
    # Test 2: Orchestrator Integration (requires running services)
    orchestrator_success = await test_orchestrator_integration()
    
    print("\n📊 Test Results:")
    print(f"Expert Council Core: {'✅ PASS' if council_success else '❌ FAIL'}")
    print(f"Orchestrator Integration: {'✅ PASS' if orchestrator_success else '❌ FAIL'}")
    
    if council_success and orchestrator_success:
        print("\n🎉 Expert Council Protocol implementation SUCCESSFUL!")
        print("AURA is now ready for multi-agent medical consultations!")
    else:
        print("\n⚠️ Some tests failed. Check service connectivity and debug.")

if __name__ == "__main__":
    asyncio.run(main())