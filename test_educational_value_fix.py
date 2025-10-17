#!/usr/bin/env python3
"""
Test script to verify the Educational Value agent fix
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))
os.environ['PYTHONPATH'] = str(app_dir)

# Set environment variables to disable tracing
os.environ['CREWAI_TRACING_ENABLED'] = 'false'
os.environ['CREWAI_LOG_LEVEL'] = 'INFO'
os.environ['CREWAI_VERBOSE'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_VERBOSE'] = 'false'

from dotenv import load_dotenv
load_dotenv()

from crew_ai_validation.core import CrewAIValidationOrchestrator

def test_educational_value_agent():
    """Test the Educational Value agent with a food delivery app idea"""
    
    print("🧪 Testing Educational Value Agent Fix")
    print("=" * 50)
    
    # Initialize the orchestrator
    orchestrator = CrewAIValidationOrchestrator()
    
    # Test idea
    idea_name = "Scheduled Food Delivery App"
    idea_concept = "Food delivery app that works on scheduled basis, like scheduled lunch, scheduled breakfast, scheduled dinner, scheduled snacks delivery. Consumer pays per month basis in India starting from Chennai."
    
    print(f"📝 Testing Idea: {idea_name}")
    print(f"📋 Concept: {idea_concept}")
    print()
    
    # Test just the Educational Value agent
    try:
        # Get the educational value agent
        educational_agent = orchestrator.agent_registry.get('agent_109_educational_value')
        if not educational_agent:
            print("❌ Educational Value agent not found!")
            print("Available agents:")
            for key in list(orchestrator.agent_registry.keys())[:10]:
                print(f"  - {key}")
            print(f"... and {len(orchestrator.agent_registry) - 10} more")
            
            # Search for educational value agent
            print("\n🔍 Searching for Educational Value agent...")
            for key, agent_info in orchestrator.agent_registry.items():
                if 'educational' in key.lower() or 'educational' in agent_info.get('sub_parameter', '').lower():
                    print(f"Found: {key} -> {agent_info.get('sub_parameter', 'N/A')}")
            return
        
        print(f"🤖 Found Educational Value Agent: {educational_agent['sub_parameter']}")
        print(f"📊 Weight: {educational_agent['config']['weight']}%")
        print(f"🔗 Dependencies: {educational_agent['config'].get('dependencies', [])}")
        print()
        
        # Create evaluation prompt using the base agent method
        base_agent = educational_agent['base_agent']
        prompt = base_agent.create_evaluation_prompt(idea_name, idea_concept)
        
        print("📝 Generated Evaluation Prompt:")
        print("-" * 30)
        print(prompt)
        print()
        
        # Check if the prompt contains industry context
        if "Food & Delivery" in prompt:
            print("✅ Industry context correctly identified as 'Food & Delivery'")
        else:
            print("❌ Industry context not properly identified")
            
        if "food industry metrics" in prompt.lower():
            print("✅ Food industry specific guidance included")
        else:
            print("❌ Food industry specific guidance missing")
            
        if "NOT general academic education" in prompt:
            print("✅ Clear guidance to avoid general education focus")
        else:
            print("❌ Missing guidance to avoid general education focus")
            
    except Exception as e:
        print(f"❌ Error testing Educational Value agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_educational_value_agent()
