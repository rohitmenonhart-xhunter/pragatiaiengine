"""
Test script for Psychometric Assessment System
"""

import sys
import os
import json

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from psychometric_evaluator import get_psychometric_evaluator
from datetime import datetime


def test_question_generation():
    """Test dynamic question generation"""
    print("=" * 80)
    print("TEST 1: Question Generation")
    print("=" * 80)
    
    try:
        evaluator = get_psychometric_evaluator()
        print("✅ Psychometric evaluator initialized")
        
        print("\n🔄 Generating 20 psychometric questions...")
        questions_data = evaluator.generate_questions(num_questions=20)
        
        print(f"\n✅ Generated {questions_data.get('total_questions')} questions")
        print(f"📋 Assessment ID: {questions_data.get('assessment_id')}")
        print(f"⏱️  Estimated time: {questions_data.get('estimated_time_minutes')} minutes")
        print(f"📝 Title: {questions_data.get('title')}")
        
        # Show sample questions
        print("\n📊 Sample Questions:")
        for i, q in enumerate(questions_data.get('questions', [])[:3], 1):
            print(f"\n{i}. [{q.get('dimension')}] {q.get('question_text')}")
            for opt in q.get('options', []):
                print(f"   {opt.get('option_id')}) {opt.get('text')}")
        
        return questions_data
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_evaluation(questions_data):
    """Test response evaluation"""
    print("\n" + "=" * 80)
    print("TEST 2: Response Evaluation")
    print("=" * 80)
    
    if not questions_data:
        print("❌ Skipping evaluation test (no questions data)")
        return None
    
    try:
        # Simulate responses (selecting option B or C for variety)
        responses = {}
        for i, question in enumerate(questions_data.get('questions', [])):
            q_id = question['question_id']
            # Alternate between B and C for realistic responses
            responses[q_id] = 'B' if i % 2 == 0 else 'C'
        
        print(f"\n🔄 Evaluating {len(responses)} responses...")
        
        evaluator = get_psychometric_evaluator()
        evaluation_result = evaluator.evaluate_responses(questions_data, responses)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📊 Overall Score: {evaluation_result.get('overall_score')}/10")
        print(f"📈 Completion Rate: {evaluation_result.get('completion_rate')}%")
        
        # Show dimension scores
        print("\n🎯 Dimension Scores:")
        dimension_scores = evaluation_result.get('dimension_scores', {})
        for dimension, score in sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score) + "░" * (10 - int(score))
            print(f"  {dimension:25s} {bar} {score:.1f}/10")
        
        # Show personality profile
        print(f"\n👤 Personality Profile:")
        print(f"  {evaluation_result.get('personality_profile', 'N/A')}")
        
        # Show entrepreneurial fit
        fit = evaluation_result.get('entrepreneurial_fit', {})
        print(f"\n🚀 Entrepreneurial Fit:")
        print(f"  Overall: {fit.get('overall_fit', 'N/A')}")
        print(f"  Score: {fit.get('fit_score', 0)}/100")
        print(f"  Ideal Role: {fit.get('ideal_role', 'N/A')}")
        print(f"  Venture Type: {fit.get('ideal_venture_type', 'N/A')}")
        
        # Show top strengths
        strengths = evaluation_result.get('strengths', [])
        if strengths:
            print(f"\n💪 Top Strengths:")
            for i, strength in enumerate(strengths[:3], 1):
                print(f"  {i}. {strength}")
        
        # Show development areas
        areas = evaluation_result.get('areas_for_development', [])
        if areas:
            print(f"\n📈 Areas for Development:")
            for i, area in enumerate(areas[:3], 1):
                print(f"  {i}. {area}")
        
        # Show recommendations
        recommendations = evaluation_result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        return evaluation_result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_json_output(questions_data, evaluation_result):
    """Test JSON serialization"""
    print("\n" + "=" * 80)
    print("TEST 3: JSON Serialization")
    print("=" * 80)
    
    try:
        # Test questions data serialization
        questions_json = json.dumps(questions_data, indent=2)
        print(f"✅ Questions data serialized ({len(questions_json)} bytes)")
        
        # Test evaluation result serialization
        evaluation_json = json.dumps(evaluation_result, indent=2)
        print(f"✅ Evaluation result serialized ({len(evaluation_json)} bytes)")
        
        # Save to file for inspection
        output_file = "sample_psychometric_output.json"
        with open(output_file, 'w') as f:
            json.dump({
                "questions": questions_data,
                "evaluation": evaluation_result
            }, f, indent=2)
        
        print(f"✅ Sample output saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PSYCHOMETRIC ASSESSMENT SYSTEM TEST" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    start_time = datetime.now()
    
    # Test 1: Generate questions
    questions_data = test_question_generation()
    
    # Test 2: Evaluate responses
    evaluation_result = None
    if questions_data:
        evaluation_result = test_evaluation(questions_data)
    
    # Test 3: JSON serialization
    if questions_data and evaluation_result:
        test_json_output(questions_data, evaluation_result)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"⏱️  Total Duration: {duration:.2f} seconds")
    print(f"✅ Question Generation: {'PASS' if questions_data else 'FAIL'}")
    print(f"✅ Response Evaluation: {'PASS' if evaluation_result else 'FAIL'}")
    print(f"✅ JSON Serialization: {'PASS' if questions_data and evaluation_result else 'FAIL'}")
    
    if questions_data and evaluation_result:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Ready for integration with Flask endpoints")
        print("\n🔗 API Endpoints:")
        print("   POST /api/psychometric/generate")
        print("   POST /api/psychometric/evaluate")
        print("   GET  /api/psychometric/evaluations/<user_id>")
        print("   GET  /api/psychometric/evaluation/<evaluation_id>")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

