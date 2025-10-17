#!/usr/bin/env python
"""
Performance Testing Script for Pragati AI Engine
Tests the validation system and measures performance metrics
"""

import time
import requests
import json
from datetime import datetime

# Test idea
test_idea = {
    "idea_name": "Smart Agriculture IoT Platform for Indian Farmers",
    "idea_concept": """An IoT-based agricultural platform that uses sensor networks and AI to help 
    Indian farmers optimize crop yields. The system monitors soil conditions (moisture, pH, nutrients), 
    weather patterns, and crop health in real-time. It provides automated recommendations for irrigation, 
    fertilization, and pest control, specifically designed for Indian farming conditions. The platform 
    includes a mobile app in regional languages, integration with government schemes like PM-KISAN, 
    and a marketplace for farmers to sell directly to buyers, cutting out middlemen."""
}

def test_system_health():
    """Test system health endpoint"""
    print("\n" + "="*80)
    print("🏥 TESTING SYSTEM HEALTH")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:5001/health")
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"📊 System Info:")
        for key, value in data['system_info'].items():
            print(f"   • {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_framework_info():
    """Get framework information"""
    print("\n" + "="*80)
    print("📋 FRAMEWORK INFORMATION")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:5001/api/framework-info")
        data = response.json()
        
        if data.get('success'):
            info = data['data']
            print(f"✅ System: {info.get('system_name', 'N/A')}")
            print(f"📊 Total Agents: {info.get('total_agents', 0)}")
            print(f"📈 Cluster Distribution:")
            for cluster, count in info.get('cluster_distribution', {}).items():
                print(f"   • {cluster}: {count} agents")
        
        return True
    except Exception as e:
        print(f"❌ Framework info failed: {e}")
        return False

def test_validation_performance():
    """Test validation with performance metrics"""
    print("\n" + "="*80)
    print("🚀 TESTING VALIDATION PERFORMANCE")
    print("="*80)
    print(f"💡 Idea: {test_idea['idea_name']}")
    print(f"📝 Concept: {test_idea['idea_concept'][:100]}...")
    print("\n⏱️  Starting validation (this may take 30-60 seconds)...\n")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:5001/api/validate-idea",
            json=test_idea,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print("=" * 80)
            print("✅ VALIDATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            # Overall metrics
            print(f"\n📊 OVERALL PERFORMANCE METRICS:")
            print(f"   • Overall Score: {data.get('overall_score', 0):.2f}/5.0")
            print(f"   • Validation Outcome: {data.get('validation_outcome', 'N/A')}")
            print(f"   • Processing Time: {data.get('processing_time', total_time):.2f} seconds")
            print(f"   • Total Wall Time: {total_time:.2f} seconds")
            print(f"   • API Calls Made: {data.get('api_calls_made', 0)}")
            print(f"   • Consensus Level: {data.get('consensus_level', 0)*100:.1f}%")
            
            # Cluster scores
            if 'cluster_scores' in data:
                print(f"\n📈 CLUSTER SCORES:")
                for cluster, score in data['cluster_scores'].items():
                    emoji = "🟢" if score >= 4.0 else "🟡" if score >= 3.0 else "🔴"
                    print(f"   {emoji} {cluster}: {score:.2f}/5.0")
            
            # Performance analysis
            print(f"\n⚡ PERFORMANCE ANALYSIS:")
            agents_consulted = data.get('api_calls_made', 0)
            if agents_consulted > 0:
                avg_time_per_agent = total_time / agents_consulted
                print(f"   • Average Time per Agent: {avg_time_per_agent:.2f} seconds")
                print(f"   • Agents per Second: {agents_consulted/total_time:.2f}")
            
            # Collaboration insights
            if 'collaboration_insights' in data and data['collaboration_insights']:
                print(f"\n💡 COLLABORATION INSIGHTS:")
                for insight in data['collaboration_insights'][:3]:
                    print(f"   • {insight}")
            
            # Save report
            if 'html_report' in data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"validation_report_{timestamp}.html"
                with open(report_filename, 'w') as f:
                    f.write(data['html_report'])
                print(f"\n📄 HTML Report saved: {report_filename}")
            
            return {
                "success": True,
                "overall_score": data.get('overall_score', 0),
                "processing_time": total_time,
                "agents_consulted": agents_consulted,
                "consensus_level": data.get('consensus_level', 0)
            }
        else:
            print(f"❌ Validation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return {"success": False}
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 5 minutes")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return {"success": False, "error": str(e)}

def generate_performance_summary(results):
    """Generate performance summary"""
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    if results.get('success'):
        print(f"✅ System Status: OPERATIONAL")
        print(f"📈 Overall Score: {results.get('overall_score', 0):.2f}/5.0")
        print(f"⏱️  Processing Time: {results.get('processing_time', 0):.2f} seconds")
        print(f"🤖 Agents Consulted: {results.get('agents_consulted', 0)}")
        print(f"🤝 Consensus Level: {results.get('consensus_level', 0)*100:.1f}%")
        
        # Performance rating
        processing_time = results.get('processing_time', 0)
        if processing_time < 30:
            performance_rating = "⚡ EXCELLENT"
        elif processing_time < 60:
            performance_rating = "✅ GOOD"
        elif processing_time < 120:
            performance_rating = "🟡 ACCEPTABLE"
        else:
            performance_rating = "🔴 SLOW"
        
        print(f"\n🏆 Performance Rating: {performance_rating}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if processing_time < 60:
            print(f"   ✅ System is performing well with efficient agent processing")
        else:
            print(f"   ⚠️  Consider optimizing agent execution or increasing parallel processing")
        
        if results.get('consensus_level', 0) > 0.8:
            print(f"   ✅ High consensus indicates reliable evaluation")
        elif results.get('consensus_level', 0) > 0.6:
            print(f"   🟡 Moderate consensus - review conflicting agent assessments")
        else:
            print(f"   ⚠️  Low consensus - idea shows mixed signals")
    else:
        print(f"❌ System Status: FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 PRAGATI AI ENGINE - PERFORMANCE TEST")
    print("="*80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Endpoint: http://localhost:5001")
    print("="*80)
    
    # Run tests
    if test_system_health():
        test_framework_info()
        results = test_validation_performance()
        generate_performance_summary(results)
    else:
        print("\n❌ System health check failed. Please ensure the server is running.")
        print("Start server with: python app/app_v3.py")

