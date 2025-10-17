#!/usr/bin/env python3
"""
Test script to validate the food delivery app idea with the improved Educational Value agent
"""

import os
import sys
import asyncio
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

async def test_food_delivery_validation():
    """Test the full validation system with the food delivery app idea"""
    
    print("🍕 Testing Food Delivery App Validation")
    print("=" * 50)
    
    # Initialize the orchestrator
    orchestrator = CrewAIValidationOrchestrator()
    
    # Test idea
    idea_name = "Scheduled Food Delivery App"
    idea_concept = "Food delivery app that works on scheduled basis, like scheduled lunch, scheduled breakfast, scheduled dinner, scheduled snacks delivery. Consumer pays per month basis in India starting from Chennai."
    
    print(f"📝 Testing Idea: {idea_name}")
    print(f"📋 Concept: {idea_concept}")
    print()
    
    try:
        # Run validation
        print("🚀 Starting validation...")
        result = await orchestrator.validate_idea(idea_name, idea_concept)
        
        print("\n📊 VALIDATION RESULTS:")
        print("=" * 50)
        print(f"Overall Score: {result.overall_score:.2f}/5.0")
        print(f"Validation Outcome: {result.validation_outcome}")
        print(f"Processing Time: {result.total_processing_time:.2f} seconds")
        
        # Find and display Educational Value agent result
        educational_result = None
        for evaluation in result.agent_evaluations:
            if 'educational' in evaluation.agent_id.lower():
                educational_result = evaluation
                break
        
        if educational_result:
            print(f"\n🎓 EDUCATIONAL VALUE AGENT RESULT:")
            print(f"Score: {educational_result.assigned_score:.2f}/5.0")
            print(f"Confidence: {educational_result.confidence_level:.2f}")
            print(f"Explanation: {educational_result.explanation}")
            
            # Check if the explanation is food-industry focused
            if any(word in educational_result.explanation.lower() for word in ['nutrition', 'meal', 'cooking', 'food', 'eating', 'diet']):
                print("✅ Educational Value agent correctly focused on food industry context!")
            else:
                print("❌ Educational Value agent may not be food-industry focused")
        else:
            print("❌ Educational Value agent result not found")
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_food_delivery_validation())
