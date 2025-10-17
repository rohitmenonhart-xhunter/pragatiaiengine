#!/usr/bin/env python3
"""
Simple test for CrewAI Multi-Agent Validation System
Tests a few key agents to validate the system is working.
"""

import os
import sys
import time
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Disable CrewAI tracing prompts for testing
os.environ['CREWAI_TRACING_ENABLED'] = 'false'

def test_agent_initialization():
    """Test that agents can be initialized properly"""
    print("🚀 Testing Agent Initialization")
    print("=" * 50)
    
    try:
        from crew_ai_integration import PragatiCrewAIValidator
        
        print("✅ Successfully imported PragatiCrewAIValidator")
        
        # Initialize the validator
        print("🔧 Initializing validation system...")
        start_time = time.time()
        
        validator = PragatiCrewAIValidator()
        
        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        print(f"🤖 Total agents created: {len(validator.agents)}")
        
        # Test system info
        system_info = validator.get_system_info()
        print(f"📊 System Info:")
        print(f"  • Version: {system_info.get('version', 'N/A')}")
        print(f"  • Total Agents: {system_info.get('total_agents', 0)}")
        print(f"  • LLM Model: {system_info.get('llm_model', 'N/A')}")
        
        return validator
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_validation(validator):
    """Test a simple validation with limited agent calls"""
    print("\n🧪 Testing Simple Validation")
    print("=" * 50)
    
    try:
        test_idea = {
            "name": "AI Study Assistant",
            "concept": """
            A mobile app that uses AI to help students create personalized study plans, 
            generate practice questions, and track learning progress. The app would 
            integrate with popular learning platforms and provide smart recommendations 
            based on the student's learning style and performance data.
            """
        }
        
        print(f"💡 Test Idea: {test_idea['name']}")
        
        # Run validation
        validation_start = time.time()
        
        result = validator.validate_idea(
            test_idea['name'], 
            test_idea['concept']
        )
        
        validation_time = time.time() - validation_start
        
        # Display results
        print(f"\n🎯 Validation completed in {validation_time:.2f} seconds")
        print("=" * 50)
        print("📊 VALIDATION RESULTS")
        print("=" * 50)
        
        if result.get('error'):
            print(f"❌ Validation failed: {result['error']}")
            return False
        
        print(f"📈 Overall Score: {result.get('overall_score', 'N/A')}/5.0")
        print(f"🏆 Validation Outcome: {result.get('validation_outcome', 'N/A')}")
        print(f"🤖 Agents Consulted: {result.get('api_calls_made', 'N/A')}")
        print(f"⏱️ Processing Time: {validation_time:.2f} seconds")
        
        # Check if we got some basic results
        if 'evaluated_data' in result and result['evaluated_data']:
            print(f"✅ Evaluation data generated successfully")
            cluster_count = len(result['evaluated_data'])
            print(f"📊 Clusters evaluated: {cluster_count}")
        
        if 'html_report' in result and result['html_report']:
            print(f"✅ HTML report generated successfully")
            report_path = "simple_test_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(result['html_report'])
            print(f"📄 Report saved: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎯 Pragati CrewAI Simple Validation Test")
    print("=" * 60)
    
    # Test 1: Agent Initialization
    validator = test_agent_initialization()
    if not validator:
        print("\n❌ Agent initialization failed. Exiting.")
        return False
    
    # Test 2: Simple Validation
    validation_success = test_simple_validation(validator)
    if not validation_success:
        print("\n❌ Validation test failed.")
        return False
    
    print("\n🎉 Simple validation test completed successfully!")
    print("🚀 The CrewAI multi-agent system is working!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
