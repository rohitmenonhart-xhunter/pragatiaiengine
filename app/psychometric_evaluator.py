"""
Psychometric Evaluation Module
Generates dynamic psychometric questions and analyzes responses
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()
logger = logging.getLogger(__name__)


class PsychometricEvaluator:
    """
    Generates psychometric assessments and evaluates responses
    Assesses personality traits, cognitive abilities, and entrepreneurial fit
    """
    
    # Psychometric dimensions to evaluate
    DIMENSIONS = {
        "leadership": {
            "name": "Leadership & Vision",
            "description": "Ability to lead, inspire, and set strategic direction"
        },
        "risk_tolerance": {
            "name": "Risk Tolerance",
            "description": "Comfort with uncertainty and calculated risk-taking"
        },
        "resilience": {
            "name": "Resilience & Adaptability",
            "description": "Ability to recover from setbacks and adapt to change"
        },
        "innovation": {
            "name": "Innovation & Creativity",
            "description": "Capacity for creative thinking and novel solutions"
        },
        "decision_making": {
            "name": "Decision Making",
            "description": "Quality and speed of judgment under pressure"
        },
        "emotional_intelligence": {
            "name": "Emotional Intelligence",
            "description": "Self-awareness and interpersonal effectiveness"
        },
        "persistence": {
            "name": "Persistence & Grit",
            "description": "Determination to pursue long-term goals"
        },
        "strategic_thinking": {
            "name": "Strategic Thinking",
            "description": "Ability to analyze complex situations and plan ahead"
        },
        "communication": {
            "name": "Communication Skills",
            "description": "Clarity and effectiveness in conveying ideas"
        },
        "problem_solving": {
            "name": "Problem Solving",
            "description": "Analytical and creative approach to challenges"
        }
    }
    
    def __init__(self):
        """Initialize the psychometric evaluator"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Use GPT-4 Turbo (latest GPT-4.1)
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.7,
            api_key=api_key,
            max_tokens=4000,  # Limit response size
            timeout=60  # 60 second timeout
        )
        logger.info("✅ Psychometric Evaluator initialized with GPT-4.1")
    
    def generate_questions(self, num_questions: int = 20) -> Dict:
        """
        Generate dynamic psychometric questions
        
        Args:
            num_questions: Number of questions to generate (default: 20)
            
        Returns:
            Dictionary containing questions and metadata
        """
        try:
            logger.info(f"🔄 Generating {num_questions} psychometric questions...")
            
            prompt = f"""
Generate exactly {num_questions} psychometric questions for entrepreneurs.

CRITICAL: Return ONLY valid JSON. No explanations, no markdown, just JSON.

Dimensions to assess: leadership, risk_tolerance, resilience, innovation, decision_making, emotional_intelligence, persistence, strategic_thinking, communication, problem_solving

JSON structure:
{{
    "assessment_id": "assess_{num_questions}q",
    "title": "Entrepreneur Assessment",
    "description": "Evaluate entrepreneurial traits and skills",
    "estimated_time_minutes": {int(num_questions * 0.75)},
    "questions": [
        {{
            "question_id": "q1",
            "dimension": "leadership",
            "question_text": "Question text here",
            "question_type": "situational",
            "options": [
                {{"option_id": "A", "text": "Option A", "score_profile": {{"leadership": 7, "decision_making": 8}}}},
                {{"option_id": "B", "text": "Option B", "score_profile": {{"leadership": 8, "emotional_intelligence": 9}}}},
                {{"option_id": "C", "text": "Option C", "score_profile": {{"strategic_thinking": 6, "decision_making": 4}}}},
                {{"option_id": "D", "text": "Option D", "score_profile": {{"innovation": 9, "problem_solving": 8}}}}
            ]
        }}
    ]
}}

Generate {num_questions} questions covering all 10 dimensions. Keep questions concise. Return ONLY the JSON."""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse JSON response
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            questions_data = json.loads(content)
            
            # Add metadata
            questions_data["generated_at"] = datetime.now().isoformat()
            questions_data["total_questions"] = len(questions_data.get("questions", []))
            
            logger.info(f"✅ Generated {questions_data['total_questions']} questions successfully")
            
            return questions_data
            
        except Exception as e:
            logger.error(f"❌ Failed to generate questions: {e}")
            raise
    
    def evaluate_responses(self, questions_data: Dict, responses: Dict[str, str]) -> Dict:
        """
        Evaluate user responses and generate psychometric summary
        
        Args:
            questions_data: Original questions with scoring profiles
            responses: Dict mapping question_id to selected option_id
            
        Returns:
            Comprehensive psychometric analysis
        """
        try:
            logger.info(f"🔄 Evaluating {len(responses)} responses...")
            
            # Calculate raw scores
            dimension_scores = {dim: [] for dim in self.DIMENSIONS.keys()}
            answered_questions = []
            
            for question in questions_data.get("questions", []):
                q_id = question["question_id"]
                if q_id not in responses:
                    continue
                
                selected_option_id = responses[q_id]
                
                # Find the selected option
                selected_option = None
                for option in question["options"]:
                    if option["option_id"] == selected_option_id:
                        selected_option = option
                        break
                
                if not selected_option:
                    continue
                
                # Record scores
                score_profile = selected_option.get("score_profile", {})
                for dimension, score in score_profile.items():
                    if dimension in dimension_scores:
                        dimension_scores[dimension].append(score)
                
                answered_questions.append({
                    "question_id": q_id,
                    "question_text": question["question_text"],
                    "dimension": question["dimension"],
                    "selected_option": selected_option_id,
                    "selected_text": selected_option["text"]
                })
            
            # Calculate average scores per dimension
            dimension_averages = {}
            for dimension, scores in dimension_scores.items():
                if scores:
                    dimension_averages[dimension] = round(sum(scores) / len(scores), 2)
                else:
                    dimension_averages[dimension] = 0.0
            
            # Generate AI-powered analysis
            analysis = self._generate_ai_analysis(
                dimension_averages,
                answered_questions,
                questions_data
            )
            
            # Compile results
            result = {
                "assessment_id": questions_data.get("assessment_id", "unknown"),
                "evaluated_at": datetime.now().isoformat(),
                "total_questions": questions_data.get("total_questions", 0),
                "questions_answered": len(answered_questions),
                "completion_rate": round(len(answered_questions) / questions_data.get("total_questions", 1) * 100, 1),
                
                "dimension_scores": dimension_averages,
                
                "overall_score": round(sum(dimension_averages.values()) / len(dimension_averages), 2) if dimension_averages else 0.0,
                
                "strengths": analysis.get("strengths", []),
                "areas_for_development": analysis.get("areas_for_development", []),
                "personality_profile": analysis.get("personality_profile", ""),
                "entrepreneurial_fit": analysis.get("entrepreneurial_fit", {}),
                "recommendations": analysis.get("recommendations", []),
                "detailed_insights": analysis.get("detailed_insights", {}),
                
                "response_details": answered_questions
            }
            
            logger.info(f"✅ Evaluation complete. Overall score: {result['overall_score']}/10")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate responses: {e}")
            raise
    
    def _generate_ai_analysis(
        self,
        dimension_scores: Dict[str, float],
        answered_questions: List[Dict],
        questions_data: Dict
    ) -> Dict:
        """
        Generate deep AI-powered analysis of psychometric results
        
        Args:
            dimension_scores: Average scores per dimension
            answered_questions: List of questions and selected answers
            questions_data: Original questions data
            
        Returns:
            Detailed analysis dictionary
        """
        try:
            # Prepare dimension details
            dimension_details = []
            for dim, score in dimension_scores.items():
                info = self.DIMENSIONS.get(dim, {})
                dimension_details.append(f"- {info.get('name', dim)}: {score}/10 - {info.get('description', '')}")
            
            # Sample responses for context
            sample_responses = answered_questions[:5]  # First 5 for context
            
            prompt = f"""
Analyze entrepreneur assessment results. Return ONLY valid JSON.

SCORES (out of 10):
{chr(10).join(dimension_details)}

OVERALL: {round(sum(dimension_scores.values()) / len(dimension_scores), 2)}/10

JSON format:

{{
    "personality_profile": "Brief 2-3 sentence profile",
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "areas_for_development": ["Area 1", "Area 2", "Area 3"],
    "entrepreneurial_fit": {{
        "overall_fit": "High",
        "fit_score": 85,
        "reasoning": "Brief explanation",
        "ideal_role": "Founder",
        "ideal_venture_type": "Tech startup"
    }},
    "recommendations": ["Rec 1", "Rec 2", "Rec 3"],
    "detailed_insights": {{
        "leadership_style": "Brief description",
        "decision_making_pattern": "Brief description",
        "stress_response": "Brief description",
        "growth_potential": "Brief description",
        "team_dynamics": "Brief description",
        "unique_qualities": "Brief description"
    }}
}}

Be specific and reference scores. Return ONLY JSON."""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse JSON
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to generate AI analysis: {e}")
            # Return basic analysis as fallback
            return {
                "personality_profile": "Analysis completed based on responses provided.",
                "strengths": ["Completed assessment"],
                "areas_for_development": ["Continue personal development"],
                "entrepreneurial_fit": {
                    "overall_fit": "Medium",
                    "fit_score": 70,
                    "reasoning": "Assessment completed successfully",
                    "ideal_role": "To be determined",
                    "ideal_venture_type": "Various"
                },
                "recommendations": ["Review detailed scores", "Continue learning"],
                "detailed_insights": {
                    "leadership_style": "To be determined from detailed analysis",
                    "decision_making_pattern": "Review individual responses",
                    "stress_response": "Continue self-reflection",
                    "growth_potential": "Positive trajectory",
                    "team_dynamics": "Further assessment recommended",
                    "unique_qualities": "Individual strengths identified"
                }
            }


def get_psychometric_evaluator() -> PsychometricEvaluator:
    """Get singleton instance of psychometric evaluator"""
    if not hasattr(get_psychometric_evaluator, "_instance"):
        get_psychometric_evaluator._instance = PsychometricEvaluator()
    return get_psychometric_evaluator._instance

