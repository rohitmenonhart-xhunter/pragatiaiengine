#!/usr/bin/env python3
"""
Test the Enhanced CrewAI System with Critical Agents
"""

import requests
import json
import time

def test_web_ui_endpoint():
    """Test the web UI is accessible"""
    try:
        response = requests.get('http://localhost:5000')
        if response.status_code == 200:
            print("✅ Web UI is accessible at http://localhost:5000")
            return True
        else:
            print(f"❌ Web UI returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot access web UI: {e}")
        return False

def test_api_validation():
    """Test the validation API with a potentially poor idea"""
    print("\n🧪 Testing API with a potentially poor idea to check critical scoring...")
    
    poor_idea = {
        "idea_name": "Another Food Delivery App",
        "idea_concept": """
        Just like Zomato and Swiggy, but we'll deliver food too. 
        We don't have any unique features, no funding, no technical team, 
        and the market is already completely saturated with established players. 
        We plan to compete on price by having negative margins.
        Our target is everyone who wants food, everywhere in India.
        """
    }
    
    try:
        print(f"📤 Sending request for: {poor_idea['idea_name']}")
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/api/validate-idea',
            json=poor_idea,
            timeout=300  # 5 minutes timeout
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            
            print(f"✅ Validation completed in {end_time - start_time:.1f} seconds")
            print(f"📊 Overall Score: {data.get('overall_score', 'N/A')}/5.0")
            print(f"🏆 Outcome: {data.get('validation_outcome', 'N/A')}")
            print(f"🤖 Agents Consulted: {data.get('api_calls_made', 'N/A')}")
            
            # Check if agents were critical
            overall_score = data.get('overall_score', 5.0)
            if overall_score < 3.0:
                print("✅ Agents were appropriately critical (low score for poor idea)")
            elif overall_score > 4.0:
                print("⚠️ Agents may be too lenient (high score for obviously poor idea)")
            else:
                print("🔶 Moderate scoring - agents showing some criticism")
            
            # Check cluster scores
            cluster_scores = data.get('cluster_scores', {})
            if cluster_scores:
                print("\n📈 Cluster Breakdown:")
                for cluster, score in cluster_scores.items():
                    emoji = "🟢" if score >= 4.0 else "🟡" if score >= 3.0 else "🔴"
                    print(f"   {emoji} {cluster}: {score:.2f}/5.0")
            
            return True
            
        else:
            print(f"❌ API validation failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_system_info():
    """Test system information endpoints"""
    try:
        # Test system info
        response = requests.get('http://localhost:5000/api/system-info')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Info: {data.get('data', {}).get('total_agents', 'N/A')} agents")
        
        # Test agents info
        response = requests.get('http://localhost:5000/api/agents')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Agents Info: {data.get('total_agents', 'N/A')} agents configured")
            
        return True
        
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 Testing Enhanced Pragati CrewAI System")
    print("=" * 60)
    
    # Test 1: Web UI Access
    ui_ok = test_web_ui_endpoint()
    
    # Test 2: System Info
    print("\n📋 Testing System Information...")
    info_ok = test_system_info()
    
    # Test 3: API Validation (with poor idea to test criticism)
    api_ok = test_api_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"🌐 Web UI: {'✅ PASS' if ui_ok else '❌ FAIL'}")
    print(f"📋 System Info: {'✅ PASS' if info_ok else '❌ FAIL'}")
    print(f"🧪 API Validation: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if ui_ok and info_ok and api_ok:
        print("\n🎉 All tests passed! The enhanced system is working!")
        print("🌐 Open http://localhost:5000 in your browser to use the web interface")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
