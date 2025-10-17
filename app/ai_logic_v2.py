

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE DATA STRUCTURES AND ENUMS
# ============================================================================

class ValidationOutcome(Enum):
    """Validation outcome categories based on overall score"""
    GOOD = "GOOD"
    MODERATE = "MODERATE" 
    NOT_RECOMMENDED = "NOT_RECOMMENDED"

@dataclass
class EvaluationResult:
    """Result of AI evaluation for a sub-parameter"""
    assigned_score: int  # 1-5
    explanation: str
    assumptions: str
    weight_contribution: float = 0.0

@dataclass
class ValidationResponse:
    """Complete validation response structure"""
    overall_score: float
    validation_outcome: ValidationOutcome
    evaluated_data: Dict[str, Any]
    html_report: str
    error: Optional[str] = None
    processing_time: float = 0.0
    api_calls_made: int = 0


class EvaluationFramework:
    """
    Professional evaluation framework implementing the exact structure
    from ai_logic_guide.txt with precise weights and parameters.
    """
    
    # Cluster weights (must sum to 100%)
    CLUSTER_WEIGHTS = {
        "Core Idea": 15,
        "Market Opportunity": 20, 
        "Execution": 20,
        "Business Model": 15,
        "Team": 10,
        "Compliance": 10,
        "Risk & Strategy": 10
    }
    
    # Parameter weights within each cluster (must sum to 100% per cluster)
    PARAMETER_WEIGHTS = {
        "Core Idea": {
            "Novelty & Uniqueness": 50,
            "Problem-Solution Fit": 30,
            "UX/Usability Potential": 20
        },
        "Market Opportunity": {
            "Market Validation": 40,
            "Geographic Specificity (India)": 30,
            "Product-Market Fit": 30
        },
        "Execution": {
            "Technical Feasibility": 40,
            "Operational Viability": 30,
            "Scalability Potential": 30
        },
        "Business Model": {
            "Financial Viability": 60,
            "Defensibility": 40
        },
        "Team": {
            "Founder-Fit": 60,
            "Culture/Values": 40
        },
        "Compliance": {
            "Regulatory (India)": 40,
            "Sustainability (ESG)": 30,
            "Ecosystem Support (India)": 30
        },
        "Risk & Strategy": {
            "Risk Assessment": 40,
            "Investor Attractiveness": 30,
            "Academic/National Alignment": 30
        }
    }
    
    # EXPANDED Sub-parameter weights - 64+ comprehensive evaluation criteria
    SUB_PARAMETER_WEIGHTS = {
        "Core Idea": {
            "Novelty & Uniqueness": {
                "Originality": 30,
                "Differentiation": 25,
                "Innovation Index": 25,
                "Disruptive Potential": 20
            },
            "Problem-Solution Fit": {
                "Problem Severity": 25,
                "Solution Effectiveness": 25,
                "Market Gap Analysis": 20,
                "Customer Pain Validation": 15,
                "Solution Uniqueness": 15
            },
            "UX/Usability Potential": {
                "Intuitive Design": 30,
                "Accessibility Compliance": 25,
                "User Interface Quality": 20,
                "Mobile Responsiveness": 15,
                "Cross-Platform Compatibility": 10
            }
        },
        "Market Opportunity": {
            "Market Validation": {
                "Market Size (TAM)": 25,
                "Competitive Intensity": 20,
                "Market Growth Rate": 20,
                "Customer Acquisition Potential": 15,
                "Market Penetration Strategy": 10,
                "Timing & Market Readiness": 10
            },
            "Geographic Specificity (India)": {
                "Regulatory Landscape": 25,
                "Infrastructure Readiness": 25,
                "Local Market Understanding": 20,
                "Cultural Adaptation": 15,
                "Regional Expansion Potential": 15
            },
            "Product-Market Fit": {
                "User Engagement": 20,
                "Retention Potential": 20,
                "Customer Satisfaction Metrics": 20,
                "Product Stickiness": 15,
                "Market Feedback Integration": 15,
                "Viral Coefficient": 10
            }
        },
        "Execution": {
            "Technical Feasibility": {
                "Technology Maturity": 20,
                "Scalability & Performance": 20,
                "Technical Architecture": 15,
                "Development Complexity": 15,
                "Security Framework": 15,
                "API Integration Capability": 15
            },
            "Operational Viability": {
                "Resource Availability": 20,
                "Process Efficiency": 20,
                "Supply Chain Management": 15,
                "Quality Assurance": 15,
                "Operational Scalability": 15,
                "Cost Structure Optimization": 15
            },
            "Scalability Potential": {
                "Business Model Scalability": 20,
                "Market Expansion Potential": 20,
                "Technology Scalability": 15,
                "Operational Scalability": 15,
                "Financial Scalability": 15,
                "International Expansion": 15
            }
        },
        "Business Model": {
            "Financial Viability": {
                "Revenue Stream Diversity": 20,
                "Profitability & Margins": 20,
                "Cash Flow Sustainability": 15,
                "Customer Lifetime Value": 15,
                "Unit Economics": 15,
                "Financial Projections Accuracy": 15
            },
            "Defensibility": {
                "Intellectual Property (IP)": 20,
                "Network Effects": 20,
                "Brand Moat": 15,
                "Data Moat": 15,
                "Switching Costs": 15,
                "Regulatory Barriers": 15
            }
        },
        "Team": {
            "Founder-Fit": {
                "Relevant Experience": 20,
                "Complementary Skills": 20,
                "Industry Expertise": 15,
                "Leadership Capability": 15,
                "Execution Track Record": 15,
                "Domain Knowledge": 15
            },
            "Culture/Values": {
                "Mission Alignment": 20,
                "Diversity & Inclusion": 20,
                "Team Dynamics": 15,
                "Communication Effectiveness": 15,
                "Adaptability": 15,
                "Work Ethics & Values": 15
            }
        },
        "Compliance": {
            "Regulatory (India)": {
                "Data Privacy Compliance": 20,
                "Sector-Specific Compliance": 20,
                "Tax Compliance": 15,
                "Labor Law Compliance": 15,
                "Import/Export Regulations": 15,
                "Digital India Compliance": 15
            },
            "Sustainability (ESG)": {
                "Environmental Impact": 20,
                "Social Impact (SDGs)": 20,
                "Governance Standards": 15,
                "Ethical Business Practices": 15,
                "Community Engagement": 15,
                "Carbon Footprint": 15
            },
            "Ecosystem Support (India)": {
                "Government & Institutional Support": 20,
                "Investor & Partner Landscape": 20,
                "Startup Ecosystem Integration": 15,
                "Mentorship Availability": 15,
                "Industry Associations": 15,
                "Academic Partnerships": 15
            }
        },
        "Risk & Strategy": {
            "Risk Assessment": {
                "Technical Risks": 20,
                "Market Risks": 20,
                "Financial Risks": 15,
                "Competitive Risks": 15,
                "Regulatory Risks": 15,
                "Operational Risks": 15
            },
            "Investor Attractiveness": {
                "Valuation Potential": 20,
                "Exit Strategy Viability": 20,
                "ROI Potential": 15,
                "Investment Stage Readiness": 15,
                "Due Diligence Preparedness": 15,
                "Investor Fit": 15
            },
            "Academic/National Alignment": {
                "National Policy Alignment (India)": 20,
                "Academic/Research Contribution": 20,
                "Innovation Ecosystem Impact": 15,
                "Knowledge Transfer Potential": 15,
                "Research Commercialization": 15,
                "Educational Value": 15
            }
        }
    }
    
    @classmethod
    def validate_weights(cls) -> bool:
        """Validate that all weights sum to 100% at each level"""
        # Check cluster weights
        cluster_sum = sum(cls.CLUSTER_WEIGHTS.values())
        if abs(cluster_sum - 100) > 0.01:
            logger.error(f"Cluster weights sum to {cluster_sum}, not 100")
            return False
            
        # Check parameter weights within each cluster
        for cluster, params in cls.PARAMETER_WEIGHTS.items():
            param_sum = sum(params.values())
            if abs(param_sum - 100) > 0.01:
                logger.error(f"Parameter weights in {cluster} sum to {param_sum}, not 100")
                return False
                
        # Check sub-parameter weights within each parameter
        for cluster, params in cls.SUB_PARAMETER_WEIGHTS.items():
            for param, sub_params in params.items():
                sub_param_sum = sum(sub_params.values())
                if abs(sub_param_sum - 100) > 0.01:
                    logger.error(f"Sub-parameter weights in {cluster}->{param} sum to {sub_param_sum}, not 100")
                    return False
                    
        return True
    
    @classmethod
    def get_evaluation_structure(cls) -> Dict[str, Any]:
        """Get the complete evaluation structure"""
        return {
            "clusters": cls.CLUSTER_WEIGHTS,
            "parameters": cls.PARAMETER_WEIGHTS,
            "sub_parameters": cls.SUB_PARAMETER_WEIGHTS
        }
    
    @classmethod
    def calculate_overall_weight(cls, cluster: str, parameter: str, sub_parameter: str) -> float:
        """Calculate the overall weight contribution of a sub-parameter"""
        cluster_weight = cls.CLUSTER_WEIGHTS.get(cluster, 0) / 100
        parameter_weight = cls.PARAMETER_WEIGHTS.get(cluster, {}).get(parameter, 0) / 100
        sub_param_weight = cls.SUB_PARAMETER_WEIGHTS.get(cluster, {}).get(parameter, {}).get(sub_parameter, 0) / 100
        
        return cluster_weight * parameter_weight * sub_param_weight * 100


class AIEngine:
    """
    Professional AI evaluation engine using Google Gemini 2.0 Flash or OpenAI GPT-4
    with sophisticated prompting and error handling.
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = None
        self.openai_client = None
        self.api_calls_count = 0
        self.current_provider = None
        
        # Try to initialize AI providers in order of preference
        self._initialize_ai_providers()
    
    def _initialize_ai_providers(self):
        """Initialize AI providers in order of preference"""
        # Try Gemini first
        if self.gemini_api_key.strip() and genai is not None:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.current_provider = "gemini"
                logger.info("Gemini AI model initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model: {e}")
        
        # Try OpenAI as fallback
        if self.openai_api_key.strip() and openai is not None:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                # Test the connection
                test_response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.current_provider = "openai"
                logger.info("OpenAI GPT-4 initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # No AI provider available
        logger.error("No AI provider available. Check API keys and installations.")
        self.current_provider = None
    
    def evaluate_sub_parameter(
        self, 
        idea_name: str, 
        idea_concept: str,
        cluster: str,
        parameter: str, 
        sub_parameter: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single sub-parameter using AI
        
        Args:
            idea_name: Name of the idea
            idea_concept: Detailed description of the idea
            cluster: Evaluation cluster name
            parameter: Parameter name within cluster
            sub_parameter: Sub-parameter name within parameter
            context: Additional context for evaluation
            
        Returns:
            EvaluationResult with score, explanation, and assumptions
        """
        if self.current_provider is None:
            return self._fallback_evaluation(sub_parameter)
        
        try:
            if self.current_provider == "gemini":
                return self._evaluate_with_gemini(idea_name, idea_concept, cluster, parameter, sub_parameter, context)
            elif self.current_provider == "openai":
                return self._evaluate_with_openai(idea_name, idea_concept, cluster, parameter, sub_parameter, context)
            else:
                return self._fallback_evaluation(sub_parameter)
            
        except Exception as e:
            logger.error(f"AI evaluation failed for {sub_parameter}: {e}")
            # Try switching to the other provider if available
            if self.current_provider == "gemini" and self.openai_client:
                logger.info("Switching to OpenAI due to Gemini error")
                self.current_provider = "openai"
                return self._evaluate_with_openai(idea_name, idea_concept, cluster, parameter, sub_parameter, context)
            return self._fallback_evaluation(sub_parameter)
    
    def _evaluate_with_gemini(self, idea_name: str, idea_concept: str, cluster: str, parameter: str, sub_parameter: str, context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate using Google Gemini"""
        prompt = self._create_evaluation_prompt(
            idea_name, idea_concept, cluster, parameter, sub_parameter, context
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "assignedScore": {"type": "INTEGER"},
                        "explanation": {"type": "STRING"},
                        "assumptions": {"type": "STRING"}
                    },
                    "required": ["assignedScore", "explanation", "assumptions"]
                }
            }
        )
        
        self.api_calls_count += 1
        parsed_response = json.loads(response.text)
        
        return EvaluationResult(
            assigned_score=parsed_response["assignedScore"],
            explanation=parsed_response["explanation"],
            assumptions=parsed_response["assumptions"]
        )
    
    def _evaluate_with_openai(self, idea_name: str, idea_concept: str, cluster: str, parameter: str, sub_parameter: str, context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate using OpenAI GPT-4"""
        prompt = self._create_evaluation_prompt(
            idea_name, idea_concept, cluster, parameter, sub_parameter, context
        )
        
        # Add JSON format instruction for OpenAI
        prompt += """

Please respond with a valid JSON object in exactly this format:
{
    "assignedScore": <integer between 1-5>,
    "explanation": "<your explanation>",
    "assumptions": "<your assumptions>"
}"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a world-class startup idea evaluator with expertise in venture capital, technology, market analysis, and business strategy. You provide precise, analytical evaluations based on proven startup frameworks. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3  # Lower temperature for more consistent, analytical responses
        )
        
        self.api_calls_count += 1
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON if it's embedded in other text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from OpenAI response")
        
        return EvaluationResult(
            assigned_score=int(parsed_response["assignedScore"]),
            explanation=parsed_response["explanation"],
            assumptions=parsed_response["assumptions"]
        )
    
    def _create_evaluation_prompt(
        self, 
        idea_name: str, 
        idea_concept: str,
        cluster: str,
        parameter: str,
        sub_parameter: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a sophisticated evaluation prompt"""
        
        # Sub-parameter specific guidance
        guidance = self._get_sub_parameter_guidance(cluster, parameter, sub_parameter)
        
        prompt = f"""
You are an expert AI idea validator for "Pragati - Idea to Impact Platform", specializing in Indian startup ecosystem evaluation.

**IDEA DETAILS:**
- **Name:** {idea_name}
- **Concept:** {idea_concept}

**EVALUATION CONTEXT:**
- **Cluster:** {cluster} 
- **Parameter:** {parameter}
- **Sub-Parameter:** {sub_parameter}

**EVALUATION FOCUS:**
{guidance}

**SCORING RUBRIC (1-5):**
- **5 (Excellent):** Strong evidence of success factors, minimal risks, clear competitive advantage
- **4 (Good):** Positive indicators, generally strong with minor improvement areas
- **3 (Moderate):** Mixed evidence, some challenges but addressable with effort
- **2 (Weak):** Significant gaps or risks requiring substantial work
- **1 (Poor):** Fundamental flaws, major red flags, or no evidence of viability

**INDIAN CONTEXT CONSIDERATIONS:**
- Regulatory environment and compliance requirements
- Infrastructure readiness and digital adoption
- Cultural preferences and user behavior patterns  
- Government policies and startup ecosystem support
- Market dynamics specific to Indian demographics

**RESPONSE FORMAT:**
Provide a JSON response with:
{{
    "assignedScore": <integer 1-5>,
    "explanation": "<2-3 sentences explaining the score based on the idea and sub-parameter criteria>",
    "assumptions": "<1-2 sentences stating key assumptions made in this evaluation>"
}}

Evaluate specifically for the {sub_parameter} aspect, considering both the idea's inherent qualities and its fit within the Indian market context.
"""
        return prompt
    
    def _get_sub_parameter_guidance(self, cluster: str, parameter: str, sub_parameter: str) -> str:
        """Get specific guidance for each sub-parameter"""
        
        guidance_map = {
            ("Core Idea", "Novelty & Uniqueness", "Originality"): 
                "Assess how genuinely new or innovative this idea is compared to existing solutions globally. Consider technological novelty, approach uniqueness, and disruptive potential.",
            
            ("Core Idea", "Novelty & Uniqueness", "Differentiation"):
                "Evaluate how this idea stands out from direct and indirect competitors. Focus on unique value propositions and sustainable competitive advantages.",
            
            ("Core Idea", "Problem-Solution Fit", "Problem Severity"):
                "Analyze the intensity and prevalence of the problem being addressed. Consider frequency, impact, and urgency of the pain point for target users.",
            
            ("Core Idea", "Problem-Solution Fit", "Solution Effectiveness"):
                "Evaluate how well the proposed solution addresses the identified problem. Consider completeness, elegance, and practical effectiveness.",
            
            ("Market Opportunity", "Market Validation", "Market Size (TAM)"):
                "Assess the total addressable market size in India. Consider population demographics, economic indicators, and growth potential.",
            
            ("Market Opportunity", "Geographic Specificity (India)", "Regulatory Landscape"):
                "Evaluate regulatory compliance requirements and challenges specific to India for this type of solution.",
            
            ("Execution", "Technical Feasibility", "Technology Maturity"):
                "Assess the maturity and reliability of required technologies. Consider availability, stability, and implementation complexity.",
            
            ("Business Model", "Financial Viability", "Revenue Stream Diversity"):
                "Evaluate the diversity and sustainability of potential revenue streams. Consider scalability and market acceptance.",
            
            ("Team", "Founder-Fit", "Relevant Experience"):
                "Assess the founder's relevant experience and domain expertise for executing this specific idea.",
            
            ("Compliance", "Regulatory (India)", "Data Privacy Compliance"):
                "Evaluate compliance with Indian data protection laws and privacy regulations.",
            
            ("Risk & Strategy", "Risk Assessment", "Technical Risks"):
                "Identify and assess potential technical risks and challenges in implementation and scaling."
        }
        
        return guidance_map.get((cluster, parameter, sub_parameter), 
                              f"Evaluate the {sub_parameter} aspect thoroughly, considering all relevant factors for startup success in India.")
    
    def _fallback_evaluation(self, sub_parameter: str) -> EvaluationResult:
        """Provide fallback evaluation when AI is unavailable"""
        return EvaluationResult(
            assigned_score=3,
            explanation=f"AI evaluation unavailable for {sub_parameter}. Default moderate score applied.",
            assumptions="Evaluation performed without AI assistance due to technical limitations."
        )
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "total_api_calls": self.api_calls_count,
            "current_provider": self.current_provider,
            "gemini_available": self.model is not None,
            "openai_available": self.openai_client is not None,
            "model_available": self.current_provider is not None
        }


class ReportGenerator:
    """
    Professional HTML report generator with comprehensive analysis
    and actionable insights.
    """
    
    @staticmethod
    def generate_comprehensive_report(
        idea_name: str,
        idea_concept: str,
        overall_score: float,
        validation_outcome: ValidationOutcome,
        evaluated_data: Dict[str, Any],
        processing_stats: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive HTML report"""
        
        current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pragati AI Validation Report - {idea_name}</title>
    <style>
        {ReportGenerator._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <div class="logo-section">
                <h1>Pragati - Idea to Impact Platform</h1>
                <p class="tagline">AI-Powered Innovation Validation</p>
            </div>
            <div class="report-meta">
                <p><strong>Report Date:</strong> {current_date}</p>
                <p><strong>Report ID:</strong> {str(uuid.uuid4())[:8].upper()}</p>
            </div>
        </header>

        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="score-card {validation_outcome.value.lower()}">
                    <h3>Overall Score</h3>
                    <div class="score-display">{overall_score:.1f}/5.0</div>
                    <div class="outcome-badge">{validation_outcome.value}</div>
                </div>
                <div class="idea-overview">
                    <h3>{idea_name}</h3>
                    <p>{idea_concept[:300]}{'...' if len(idea_concept) > 300 else ''}</p>
                </div>
            </div>
        </section>

        <section class="validation-outcome">
            <h2>Validation Outcome</h2>
            {ReportGenerator._get_outcome_analysis(validation_outcome, overall_score)}
        </section>

        <section class="detailed-analysis">
            <h2>Detailed Analysis</h2>
            {ReportGenerator._generate_cluster_analysis(evaluated_data)}
        </section>

        <section class="recommendations">
            <h2>Strategic Recommendations</h2>
            {ReportGenerator._generate_recommendations(validation_outcome, overall_score, evaluated_data)}
        </section>

        <section class="methodology">
            <h2>Methodology</h2>
            {ReportGenerator._get_methodology_section()}
        </section>

        <footer class="report-footer">
            <p>Generated by Pragati AI Validation System v2.0 | {processing_stats.get('total_api_calls', 0)} AI evaluations performed</p>
            <p><em>This report is generated using advanced AI analysis and should be used as guidance for decision-making.</em></p>
        </footer>
    </div>
</body>
</html>
"""
        return html_template
    
    @staticmethod
    def _get_css_styles() -> str:
        """Get comprehensive CSS styles for the report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        .logo-section h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 5px;
        }

        .tagline {
            color: #7f8c8d;
            font-style: italic;
        }

        .report-meta {
            text-align: right;
            color: #666;
        }

        .executive-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            margin-top: 20px;
        }

        .score-card {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }

        .score-display {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }

        .outcome-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .good .outcome-badge {
            background: #27ae60;
        }

        .moderate .outcome-badge {
            background: #f39c12;
        }

        .not_recommended .outcome-badge {
            background: #e74c3c;
        }

        .idea-overview h3 {
            font-size: 1.8em;
            margin-bottom: 15px;
        }

        .validation-outcome, .detailed-analysis, .recommendations, .methodology {
            margin-bottom: 30px;
        }

        h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        .cluster-analysis {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }

        .cluster-header {
            background: #3498db;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
        }

        .cluster-content {
            padding: 20px;
        }

        .parameter-row {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 15px;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }

        .parameter-row:last-child {
            border-bottom: none;
        }

        .score-indicator {
            display: inline-block;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            color: white;
            text-align: center;
            line-height: 30px;
            font-weight: bold;
        }

        .score-5 { background: #27ae60; }
        .score-4 { background: #2ecc71; }
        .score-3 { background: #f39c12; }
        .score-2 { background: #e67e22; }
        .score-1 { background: #e74c3c; }

        .recommendation-card {
            background: #fff;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .recommendation-card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .report-footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }

        @media (max-width: 768px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .parameter-row {
                grid-template-columns: 1fr;
                text-align: center;
            }
        }
        """
    
    @staticmethod
    def _get_outcome_analysis(outcome: ValidationOutcome, score: float) -> str:
        """Generate outcome-specific analysis"""
        if outcome == ValidationOutcome.GOOD:
            return f"""
            <div class="outcome-analysis good">
                <h3>✅ Recommended for Development</h3>
                <p>Your idea shows strong potential with an overall score of {score:.1f}/5.0. The evaluation reveals solid fundamentals across key success factors. This idea is recommended for further development and potential incubation.</p>
                <div class="next-steps">
                    <h4>Immediate Next Steps:</h4>
                    <ul>
                        <li>Develop a detailed business plan and financial projections</li>
                        <li>Begin market validation with target users</li>
                        <li>Consider intellectual property protection strategies</li>
                        <li>Explore funding opportunities and accelerator programs</li>
                    </ul>
                </div>
            </div>
            """
        elif outcome == ValidationOutcome.MODERATE:
            return f"""
            <div class="outcome-analysis moderate">
                <h3>⚠️ Requires Strategic Improvements</h3>
                <p>Your idea has potential but shows notable areas for improvement with a score of {score:.1f}/5.0. While the core concept may be sound, specific aspects need strengthening before proceeding to development.</p>
                <div class="improvement-focus">
                    <h4>Focus Areas for Improvement:</h4>
                    <ul>
                        <li>Strengthen market validation and user research</li>
                        <li>Refine the business model and revenue strategy</li>
                        <li>Address technical feasibility concerns</li>
                        <li>Enhance competitive differentiation</li>
                    </ul>
                </div>
            </div>
            """
        else:  # NOT_RECOMMENDED
            return f"""
            <div class="outcome-analysis not-recommended">
                <h3>❌ Not Recommended at Current Stage</h3>
                <p>The evaluation reveals significant challenges with an overall score of {score:.1f}/5.0. While this doesn't mean the idea lacks merit entirely, substantial rework is needed before it can be considered viable.</p>
                <div class="fundamental-issues">
                    <h4>Key Issues to Address:</h4>
                    <ul>
                        <li>Fundamental problem-solution fit needs validation</li>
                        <li>Market opportunity requires deeper analysis</li>
                        <li>Technical or operational feasibility concerns</li>
                        <li>Business model viability questions</li>
                    </ul>
                </div>
            </div>
            """
    
    @staticmethod
    def _generate_cluster_analysis(evaluated_data: Dict[str, Any]) -> str:
        """Generate detailed cluster-by-cluster analysis"""
        html_sections = []
        
        # This would iterate through the evaluation structure
        # For now, providing a framework that can be expanded
        cluster_order = [
            "Core Idea", "Market Opportunity", "Execution", 
            "Business Model", "Team", "Compliance", "Risk & Strategy"
        ]
        
        for cluster in cluster_order:
            html_sections.append(f"""
            <div class="cluster-analysis">
                <div class="cluster-header">
                    {cluster} Analysis
                </div>
                <div class="cluster-content">
                    <p>Detailed analysis for {cluster} cluster would be generated here based on evaluated_data structure.</p>
                    <!-- This section would be populated with actual evaluation results -->
                </div>
            </div>
            """)
        
        return "\n".join(html_sections)
    
    @staticmethod
    def _generate_recommendations(outcome: ValidationOutcome, score: float, evaluated_data: Dict[str, Any]) -> str:
        """Generate strategic recommendations based on evaluation"""
        recommendations = []
        
        if outcome == ValidationOutcome.GOOD:
            recommendations = [
                {
                    "title": "Market Entry Strategy",
                    "content": "Develop a phased market entry approach starting with early adopters in your target segment."
                },
                {
                    "title": "Product Development",
                    "content": "Focus on MVP development with core features that address the primary pain points identified."
                },
                {
                    "title": "Funding Preparation", 
                    "content": "Prepare comprehensive pitch materials and financial projections for potential investors."
                }
            ]
        elif outcome == ValidationOutcome.MODERATE:
            recommendations = [
                {
                    "title": "Market Validation",
                    "content": "Conduct deeper market research and user interviews to validate assumptions."
                },
                {
                    "title": "Business Model Refinement",
                    "content": "Explore alternative revenue models and pricing strategies."
                },
                {
                    "title": "Technical Validation",
                    "content": "Build prototypes or conduct technical feasibility studies."
                }
            ]
        else:
            recommendations = [
                {
                    "title": "Fundamental Reassessment",
                    "content": "Revisit the core problem definition and solution approach."
                },
                {
                    "title": "Market Research",
                    "content": "Conduct comprehensive market analysis and customer discovery."
                },
                {
                    "title": "Pivot Consideration",
                    "content": "Consider pivoting the idea based on identified market needs."
                }
            ]
        
        html_content = ""
        for rec in recommendations:
            html_content += f"""
            <div class="recommendation-card">
                <h4>{rec['title']}</h4>
                <p>{rec['content']}</p>
            </div>
            """
        
        return html_content
    
    @staticmethod
    def _get_methodology_section() -> str:
        """Get the methodology explanation section"""
        return """
        <div class="methodology-content">
            <p>This evaluation uses a comprehensive 7-cluster framework specifically designed for the Indian startup ecosystem:</p>
            <ol>
                <li><strong>Core Idea (15%):</strong> Novelty, problem-solution fit, and user experience potential</li>
                <li><strong>Market Opportunity (20%):</strong> Market validation, Indian market specifics, and product-market fit</li>
                <li><strong>Execution (20%):</strong> Technical feasibility, operational viability, and scalability</li>
                <li><strong>Business Model (15%):</strong> Financial viability and competitive defensibility</li>
                <li><strong>Team (10%):</strong> Founder-market fit and organizational culture</li>
                <li><strong>Compliance (10%):</strong> Regulatory compliance, sustainability, and ecosystem support</li>
                <li><strong>Risk & Strategy (10%):</strong> Risk assessment, investor appeal, and strategic alignment</li>
            </ol>
            <p>Each cluster contains multiple parameters and sub-parameters, evaluated using advanced AI analysis with Google Gemini 2.0 Flash, specifically trained on Indian market dynamics and startup success factors.</p>
        </div>
        """

# ============================================================================
# VALIDATION ORCHESTRATOR - MAIN COORDINATION CLASS
# ============================================================================

class ValidationOrchestrator:
    """
    Main orchestration class that coordinates the entire validation process.
    This is the primary interface for the Flask application.
    """
    
    def __init__(self):
        self.framework = EvaluationFramework()
        self.ai_engine = AIEngine()
        self.report_generator = ReportGenerator()
        self.enable_parallel_processing = True  # V2 Enhanced Performance Mode
        self.max_workers = 8  # Parallel evaluation threads
        
        # Validate framework weights on initialization
        if not self.framework.validate_weights():
            raise ValueError("Evaluation framework weights are invalid")
    
    def validate_idea(
        self, 
        idea_name: str, 
        idea_concept: str, 
        custom_weights: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> ValidationResponse:
        """
        Main validation function - interface for Flask application
        
        Args:
            idea_name: Name of the idea
            idea_concept: Detailed description of the idea concept
            custom_weights: Optional custom cluster weights (percentages)
            **kwargs: Additional parameters (for compatibility)
            
        Returns:
            ValidationResponse with complete evaluation results
        """
        start_time = datetime.now()
        
        try:
            # Input validation
            if not idea_name or not idea_concept:
                return ValidationResponse(
                    overall_score=0.0,
                    validation_outcome=ValidationOutcome.NOT_RECOMMENDED,
                    evaluated_data={},
                    html_report="",
                    error="Idea name and concept cannot be empty"
                )
            
            # Apply custom weights if provided
            if custom_weights:
                self._apply_custom_weights(custom_weights)
            
            # Perform comprehensive evaluation
            evaluated_data = self._perform_evaluation(idea_name, idea_concept)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(evaluated_data)
            
            # Determine validation outcome
            validation_outcome = self._determine_outcome(overall_score)
            
            # Generate comprehensive report
            processing_stats = self.ai_engine.get_api_usage_stats()
            html_report = self.report_generator.generate_comprehensive_report(
                idea_name, idea_concept, overall_score, validation_outcome,
                evaluated_data, processing_stats
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResponse(
                overall_score=overall_score,
                validation_outcome=validation_outcome,
                evaluated_data=evaluated_data,
                html_report=html_report,
                processing_time=processing_time,
                api_calls_made=processing_stats["total_api_calls"]
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResponse(
                overall_score=0.0,
                validation_outcome=ValidationOutcome.NOT_RECOMMENDED,
                evaluated_data={},
                html_report="",
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _apply_custom_weights(self, custom_weights: Dict[str, int]):
        """Apply custom cluster weights"""
        # Normalize custom weights to ensure they sum to 100%
        total_weight = sum(custom_weights.values())
        if total_weight > 0:
            normalized_weights = {k: (v / total_weight) * 100 for k, v in custom_weights.items()}
            self.framework.CLUSTER_WEIGHTS.update(normalized_weights)
    
    def _perform_evaluation(self, idea_name: str, idea_concept: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation across all sub-parameters with high-performance parallel processing"""
        evaluated_data = {}
        
        if self.enable_parallel_processing:
            return self._perform_parallel_evaluation(idea_name, idea_concept)
        else:
            # Fallback to sequential evaluation
            return self._perform_sequential_evaluation(idea_name, idea_concept)
    
    def _perform_parallel_evaluation(self, idea_name: str, idea_concept: str) -> Dict[str, Any]:
        """High-performance parallel evaluation using ThreadPoolExecutor"""
        evaluated_data = {}
        
        # Prepare all evaluation tasks
        evaluation_tasks = []
        for cluster, parameters in self.framework.SUB_PARAMETER_WEIGHTS.items():
            for parameter, sub_parameters in parameters.items():
                for sub_parameter in sub_parameters.keys():
                    evaluation_tasks.append((cluster, parameter, sub_parameter))
        
        # Execute evaluations in parallel with progress tracking
        logger.info(f"Starting parallel evaluation of {len(evaluation_tasks)} sub-parameters with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._evaluate_single_sub_parameter, 
                    idea_name, idea_concept, cluster, parameter, sub_parameter
                ): (cluster, parameter, sub_parameter)
                for cluster, parameter, sub_parameter in evaluation_tasks
            }
            
            # Initialize data structure
            for cluster, parameters in self.framework.SUB_PARAMETER_WEIGHTS.items():
                evaluated_data[cluster] = {}
                for parameter, sub_parameters in parameters.items():
                    evaluated_data[cluster][parameter] = {}
            
            # Collect results
            completed_count = 0
            from concurrent.futures import as_completed
            for future in as_completed(future_to_task):
                cluster, parameter, sub_parameter = future_to_task[future]
                try:
                    evaluation_result = future.result()
                    evaluated_data[cluster][parameter][sub_parameter] = asdict(evaluation_result)
                    completed_count += 1
                    
                    if completed_count % 5 == 0:  # Progress update every 5 evaluations
                        logger.info(f"Parallel evaluation progress: {completed_count}/{len(evaluation_tasks)} completed")
                        
                    logger.info(f"Evaluated {cluster} > {parameter} > {sub_parameter}: {evaluation_result.assigned_score}/5")
                except Exception as e:
                    logger.error(f"Evaluation failed for {cluster} > {parameter} > {sub_parameter}: {e}")
                    # Use fallback evaluation
                    fallback_result = self.ai_engine._fallback_evaluation(sub_parameter)
                    evaluated_data[cluster][parameter][sub_parameter] = asdict(fallback_result)
        
        logger.info(f"Parallel evaluation completed: {len(evaluation_tasks)} sub-parameters processed")
        return evaluated_data
    
    def _perform_sequential_evaluation(self, idea_name: str, idea_concept: str) -> Dict[str, Any]:
        """Sequential evaluation (fallback mode)"""
        evaluated_data = {}
        
        for cluster, parameters in self.framework.SUB_PARAMETER_WEIGHTS.items():
            evaluated_data[cluster] = {}
            
            for parameter, sub_parameters in parameters.items():
                evaluated_data[cluster][parameter] = {}
                
                for sub_parameter in sub_parameters.keys():
                    evaluation_result = self._evaluate_single_sub_parameter(
                        idea_name, idea_concept, cluster, parameter, sub_parameter
                    )
                    evaluated_data[cluster][parameter][sub_parameter] = asdict(evaluation_result)
                    logger.info(f"Evaluated {cluster} > {parameter} > {sub_parameter}: {evaluation_result.assigned_score}/5")
        
        return evaluated_data
    
    def _evaluate_single_sub_parameter(self, idea_name: str, idea_concept: str, 
                                     cluster: str, parameter: str, sub_parameter: str) -> 'EvaluationResult':
        """Evaluate a single sub-parameter (thread-safe)"""
        # Evaluate using AI
        evaluation_result = self.ai_engine.evaluate_sub_parameter(
            idea_name, idea_concept, cluster, parameter, sub_parameter
        )
        
        # Calculate weight contribution
        weight_contribution = self.framework.calculate_overall_weight(
            cluster, parameter, sub_parameter
        )
        evaluation_result.weight_contribution = weight_contribution
        
        return evaluation_result
    
    def _calculate_overall_score(self, evaluated_data: Dict[str, Any]) -> float:
        """Calculate the weighted overall score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for cluster, parameters in evaluated_data.items():
            for parameter, sub_parameters in parameters.items():
                for sub_parameter, result in sub_parameters.items():
                    score = result["assigned_score"]
                    weight = result["weight_contribution"] / 100
                    
                    total_weighted_score += score * weight
                    total_weight += weight
        
        # Normalize to 5-point scale
        if total_weight > 0:
            return min(5.0, max(1.0, total_weighted_score))
        else:
            return 1.0
    
    def _determine_outcome(self, overall_score: float) -> ValidationOutcome:
        """Determine validation outcome based on overall score"""
        if overall_score >= 4.0:
            return ValidationOutcome.GOOD
        elif overall_score >= 2.5:
            return ValidationOutcome.MODERATE
        else:
            return ValidationOutcome.NOT_RECOMMENDED

# ============================================================================
# MAIN INTERFACE FUNCTION - FOR FLASK INTEGRATION
# ============================================================================

# Global orchestrator instance
_orchestrator = None

def validate_idea(idea_name: str, idea_concept: str, weights: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Main validation function for Flask application integration.
    
    This function provides the same interface as the original ai_logic.py
    to ensure seamless integration with the existing Flask application.
    
    Args:
        idea_name: Name of the idea
        idea_concept: Detailed description of the idea
        weights: Optional custom cluster weights
        
    Returns:
        Dictionary with validation results in original format
    """
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = ValidationOrchestrator()
    
    # Perform validation
    response = _orchestrator.validate_idea(idea_name, idea_concept, weights)
    
    # Convert to original format for Flask compatibility
    return {
        "overall_score": response.overall_score,
        "validation_outcome": response.validation_outcome.value,
        "evaluated_data": response.evaluated_data,
        "html_report": response.html_report,
        "error": response.error
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_evaluation_framework_info() -> Dict[str, Any]:
    """Get comprehensive information about the evaluation framework"""
    return {
        "version": "2.0",
        "framework": EvaluationFramework.get_evaluation_structure(),
        "total_sub_parameters": sum(
            len(sub_params) 
            for cluster_params in EvaluationFramework.SUB_PARAMETER_WEIGHTS.values()
            for sub_params in cluster_params.values()
        ),
        "weights_validated": EvaluationFramework.validate_weights()
    }

def get_system_health() -> Dict[str, Any]:
    """Get system health information"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = ValidationOrchestrator()
    
    return {
        "ai_engine_available": _orchestrator.ai_engine.model is not None,
        "framework_weights_valid": EvaluationFramework.validate_weights(),
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "system_status": "operational" if _orchestrator.ai_engine.model else "degraded"
    }

if __name__ == "__main__":
    # Example usage and testing
    print("Pragati AI Logic V2 - System Check")
    print("=" * 50)
    
    # Check framework weights
    print(f"Framework weights valid: {EvaluationFramework.validate_weights()}")
    
    # Check system health
    health = get_system_health()
    print(f"System health: {health}")
    
    # Example validation (if API key is available)
    if health["ai_engine_available"]:
        result = validate_idea(
            "Smart Campus Management System",
            "An AI-powered platform to optimize campus resources, track student activities, and improve administrative efficiency in Indian colleges."
        )
        print(f"Example validation score: {result['overall_score']:.2f}")
        print(f"Outcome: {result['validation_outcome']}")
    else:
        print("AI engine not available - check GEMINI_API_KEY configuration")
