"""
Integration module for CrewAI Multi-Agent Validation System with Pragati Backend
Replaces the existing AI logic with 109+ specialized agents.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from crew_ai_validation import CrewAIValidationOrchestrator, ValidationResult, ValidationOutcome
from crew_ai_validation.agent_factory import ComprehensiveAgentFactory
from crew_ai_validation.base_agent import AgentCollaborationManager

# Load environment variables
load_dotenv()


class PragatiCrewAIValidator:
    """
    Main integration class that replaces ai_logic_v2.py with CrewAI multi-agent system.
    Maintains compatibility with existing Flask app while using 109+ specialized agents.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.3,
            model="gpt-4o",
            max_tokens=1500
        )
        
        # Initialize agent factory and create all agents
        print("Initializing 109+ specialized validation agents...")
        self.agent_factory = ComprehensiveAgentFactory(self.llm)
        self.agents = self.agent_factory.create_all_agents()
        
        # Initialize collaboration manager
        self.collaboration_manager = AgentCollaborationManager(self.agents)
        
        # Initialize orchestrator with our agents
        self.orchestrator = CrewAIValidationOrchestrator()
        self.orchestrator.agent_registry = self._convert_agents_to_registry_format()
        
        print(f"✅ Successfully initialized {len(self.agents)} validation agents")
        self._print_agent_summary()
    
    def _convert_agents_to_registry_format(self) -> Dict[str, Dict[str, Any]]:
        """Convert our agents to the format expected by the orchestrator"""
        registry = {}
        for agent_id, agent in self.agents.items():
            registry[agent_id] = {
                'agent': agent.create_agent(),
                'cluster': agent.cluster,
                'parameter': agent.parameter,
                'sub_parameter': agent.sub_parameter,
                'config': {
                    'weight': agent.weight,
                    'dependencies': agent.dependencies
                }
            }
        return registry
    
    def _print_agent_summary(self):
        """Print summary of created agents"""
        cluster_counts = self.agent_factory.get_agent_count_by_cluster()
        print("\n📊 Agent Distribution by Cluster:")
        for cluster, count in cluster_counts.items():
            print(f"  • {cluster}: {count} agents")
        print(f"\n🎯 Total Agents: {self.agent_factory.get_total_agent_count()}")
    
    async def validate_idea_async(self, idea_name: str, idea_concept: str, 
                                custom_weights: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Async validation using all 109+ agents (main method)
        """
        return await self.orchestrator.validate_idea(idea_name, idea_concept, custom_weights)
    
    def validate_idea(self, idea_name: str, idea_concept: str, 
                     custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Synchronous validation method (compatible with existing Flask app)
        This is the main interface that replaces the original validate_idea function
        """
        # Run async validation in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            validation_result = loop.run_until_complete(
                self.validate_idea_async(idea_name, idea_concept, custom_weights)
            )
            
            # Convert to format expected by existing Flask app
            return self._convert_to_legacy_format(validation_result)
            
        except Exception as e:
            print(f"Error in CrewAI validation: {e}")
            return self._create_fallback_result(idea_name, idea_concept, str(e))
    
    def _convert_to_legacy_format(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Convert CrewAI result to format expected by existing Flask app"""
        
        # Generate HTML report
        html_report = self._generate_html_report(validation_result)
        
        # Convert agent evaluations to legacy format
        evaluated_data = self._convert_evaluations_to_legacy_format(validation_result.agent_evaluations)
        
        return {
            "overall_score": validation_result.overall_score,
            "validation_outcome": validation_result.validation_outcome.value,
            "evaluated_data": evaluated_data,
            "html_report": html_report,
            "error": None,
            "processing_time": validation_result.total_processing_time,
            "api_calls_made": validation_result.total_agents_consulted,
            "consensus_level": validation_result.consensus_level,
            "collaboration_insights": validation_result.collaboration_insights,
            "cluster_scores": validation_result.cluster_scores,
            "validation_id": validation_result.validation_id,
            "timestamp": validation_result.timestamp
        }
    
    def _convert_evaluations_to_legacy_format(self, evaluations: List) -> Dict[str, Any]:
        """Convert agent evaluations to nested dictionary format with ALL details"""
        evaluated_data = {}
        
        for evaluation in evaluations:
            cluster = evaluation.cluster
            parameter = evaluation.sub_cluster  # This maps to the parameter level
            sub_parameter = evaluation.parameter_name  # This maps to the sub-parameter level
            
            # Initialize nested structure
            if cluster not in evaluated_data:
                evaluated_data[cluster] = {}
            if parameter not in evaluated_data[cluster]:
                evaluated_data[cluster][parameter] = {}
            
            # Add COMPLETE evaluation data including all bullet points
            evaluated_data[cluster][parameter][sub_parameter] = {
                "assignedScore": evaluation.assigned_score,
                "explanation": evaluation.explanation,
                "assumptions": evaluation.assumptions,
                "weight_contribution": evaluation.weight_contribution,
                "confidence_level": evaluation.confidence_level,
                "processing_time": evaluation.processing_time,
                "agent_id": evaluation.agent_id,
                "dependencies": evaluation.dependencies,
                # NEW: Include all rich agent insights
                "key_insights": evaluation.key_insights if hasattr(evaluation, 'key_insights') else [],
                "recommendations": evaluation.recommendations if hasattr(evaluation, 'recommendations') else [],
                "risk_factors": evaluation.risk_factors if hasattr(evaluation, 'risk_factors') else [],
                "strengths": getattr(evaluation, 'strengths', []),
                "weaknesses": getattr(evaluation, 'weaknesses', []),
                "peer_challenges": evaluation.peer_challenges if hasattr(evaluation, 'peer_challenges') else [],
                "evidence_gaps": evaluation.evidence_gaps if hasattr(evaluation, 'evidence_gaps') else [],
                "indian_market_considerations": evaluation.indian_market_considerations if hasattr(evaluation, 'indian_market_considerations') else ""
            }
        
        return evaluated_data
    
    def _generate_html_report(self, validation_result: ValidationResult) -> str:
        """Generate comprehensive HTML report"""
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Get top performing and underperforming areas
        cluster_scores = validation_result.cluster_scores
        best_cluster = max(cluster_scores.items(), key=lambda x: x[1]) if cluster_scores else ("N/A", 0)
        worst_cluster = min(cluster_scores.items(), key=lambda x: x[1]) if cluster_scores else ("N/A", 0)
        
        # Generate recommendations based on outcome
        recommendations = self._generate_recommendations(validation_result.validation_outcome, 
                                                       validation_result.overall_score)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pragati AI Validation Report - Multi-Agent Analysis</title>
            <style>
                {self._get_report_css()}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <div class="logo-section">
                        <h1>Pragati AI - Multi-Agent Validation</h1>
                        <p class="tagline">109+ Specialized AI Agents for Comprehensive Idea Assessment</p>
                    </div>
                    <div class="report-meta">
                        <p><strong>Report Date:</strong> {current_date}</p>
                        <p><strong>Validation ID:</strong> {validation_result.validation_id}</p>
                        <p><strong>Agents Consulted:</strong> {validation_result.total_agents_consulted}</p>
                    </div>
                </header>

                <section class="executive-summary">
                    <h2>Executive Summary</h2>
                    <div class="summary-grid">
                        <div class="score-card {validation_result.validation_outcome.value.lower()}">
                            <h3>Overall Score</h3>
                            <div class="score-display">{validation_result.overall_score:.2f}/5.0</div>
                            <div class="outcome-badge">{validation_result.validation_outcome.value}</div>
                            <div class="consensus-indicator">
                                <small>Agent Consensus: {validation_result.consensus_level:.1%}</small>
                            </div>
                        </div>
                        <div class="performance-overview">
                            <h3>Performance Overview</h3>
                            <p><strong>Best Performing Area:</strong> {best_cluster[0]} ({best_cluster[1]:.2f}/5.0)</p>
                            <p><strong>Area for Improvement:</strong> {worst_cluster[0]} ({worst_cluster[1]:.2f}/5.0)</p>
                            <p><strong>Processing Time:</strong> {validation_result.total_processing_time:.1f} seconds</p>
                        </div>
                    </div>
                </section>

                <section class="cluster-breakdown">
                    <h2>Detailed Cluster Analysis</h2>
                    {self._generate_cluster_breakdown_html(validation_result.cluster_scores)}
                </section>

                <section class="collaboration-insights">
                    <h2>Multi-Agent Collaboration Insights</h2>
                    <div class="insights-grid">
                        {self._generate_insights_html(validation_result.collaboration_insights)}
                    </div>
                </section>

                <section class="recommendations">
                    <h2>Strategic Recommendations</h2>
                    {recommendations}
                </section>

                <section class="methodology">
                    <h2>Multi-Agent Methodology</h2>
                    <div class="methodology-content">
                        <p>This evaluation utilized {validation_result.total_agents_consulted} specialized AI agents working in collaboration:</p>
                        <ul>
                            <li><strong>Specialized Expertise:</strong> Each agent focuses on a specific validation parameter</li>
                            <li><strong>Collaborative Analysis:</strong> Agents share insights and build upon each other's assessments</li>
                            <li><strong>Dependency Resolution:</strong> Complex interdependencies between parameters are properly handled</li>
                            <li><strong>Consensus Building:</strong> Multiple viewpoints are synthesized into coherent recommendations</li>
                        </ul>
                        <p><strong>Consensus Level:</strong> {validation_result.consensus_level:.1%} - 
                        {'High agreement among agents' if validation_result.consensus_level > 0.8 else 'Moderate agreement with some divergent views' if validation_result.consensus_level > 0.6 else 'Significant disagreement indicates mixed evaluation'}</p>
                    </div>
                </section>

                <footer class="report-footer">
                    <p>Generated by Pragati AI Multi-Agent Validation System | {validation_result.total_agents_consulted} agents consulted</p>
                    <p><em>This report represents the collaborative assessment of specialized AI agents and should guide strategic decision-making.</em></p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_cluster_breakdown_html(self, cluster_scores: Dict[str, float]) -> str:
        """Generate HTML for cluster score breakdown"""
        html = '<div class="cluster-grid">'
        
        for cluster, score in cluster_scores.items():
            status_class = "excellent" if score >= 4.0 else "good" if score >= 3.0 else "moderate" if score >= 2.0 else "poor"
            html += f"""
            <div class="cluster-card {status_class}">
                <h4>{cluster}</h4>
                <div class="cluster-score">{score:.2f}</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {(score/5.0)*100}%"></div>
                </div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_insights_html(self, insights: List[str]) -> str:
        """Generate HTML for collaboration insights"""
        html = ""
        for insight in insights:
            html += f'<div class="insight-card"><p>{insight}</p></div>'
        return html
    
    def _generate_recommendations(self, outcome: ValidationOutcome, score: float) -> str:
        """Generate recommendations based on validation outcome"""
        if outcome == ValidationOutcome.EXCELLENT:
            return """
            <div class="recommendations-excellent">
                <h3>🚀 Excellent Potential - Ready for Launch</h3>
                <ul>
                    <li>Proceed with MVP development and market entry strategy</li>
                    <li>Prepare for Series A funding with strong validation metrics</li>
                    <li>Focus on rapid scaling and market capture</li>
                    <li>Build strategic partnerships to accelerate growth</li>
                </ul>
            </div>
            """
        elif outcome == ValidationOutcome.GOOD:
            return """
            <div class="recommendations-good">
                <h3>✅ Strong Potential - Recommended for Development</h3>
                <ul>
                    <li>Develop detailed business plan and go-to-market strategy</li>
                    <li>Address identified weaknesses before full-scale launch</li>
                    <li>Consider seed funding to accelerate development</li>
                    <li>Build pilot programs to validate market assumptions</li>
                </ul>
            </div>
            """
        elif outcome == ValidationOutcome.MODERATE:
            return """
            <div class="recommendations-moderate">
                <h3>⚠️ Moderate Potential - Requires Strategic Improvements</h3>
                <ul>
                    <li>Focus on strengthening low-scoring areas before proceeding</li>
                    <li>Conduct additional market validation and customer research</li>
                    <li>Consider pivoting aspects of the business model</li>
                    <li>Seek mentorship and advisory support</li>
                </ul>
            </div>
            """
        else:
            return """
            <div class="recommendations-weak">
                <h3>❌ Significant Challenges - Major Rework Needed</h3>
                <ul>
                    <li>Fundamental reassessment of problem-solution fit required</li>
                    <li>Extensive market research and validation needed</li>
                    <li>Consider substantial pivoting or alternative approaches</li>
                    <li>Focus on addressing core viability concerns</li>
                </ul>
            </div>
            """
    
    def _get_report_css(self) -> str:
        """Get CSS styles for the HTML report"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
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
            background: rgba(255,255,255,0.2);
        }
        
        .consensus-indicator {
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .cluster-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .cluster-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }
        
        .cluster-card.excellent { border-left-color: #27ae60; }
        .cluster-card.good { border-left-color: #2ecc71; }
        .cluster-card.moderate { border-left-color: #f39c12; }
        .cluster-card.poor { border-left-color: #e74c3c; }
        
        .cluster-score {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .score-bar {
            width: 100%;
            height: 10px;
            background: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .score-fill {
            height: 100%;
            background: #3498db;
            transition: width 0.3s ease;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .insight-card {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .recommendations-excellent { border-left: 5px solid #27ae60; padding: 20px; background: #d5f4e6; }
        .recommendations-good { border-left: 5px solid #2ecc71; padding: 20px; background: #d1f2eb; }
        .recommendations-moderate { border-left: 5px solid #f39c12; padding: 20px; background: #fdeaa7; }
        .recommendations-weak { border-left: 5px solid #e74c3c; padding: 20px; background: #fadbd8; }
        
        .methodology-content {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .report-footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        ul {
            padding-left: 20px;
        }
        
        li {
            margin-bottom: 8px;
        }
        """
    
    def _create_fallback_result(self, idea_name: str, idea_concept: str, error: str) -> Dict[str, Any]:
        """Create fallback result when validation fails"""
        return {
            "overall_score": 3.0,
            "validation_outcome": "MODERATE",
            "evaluated_data": {},
            "html_report": f"<h1>Validation Error</h1><p>Error: {error}</p>",
            "error": error,
            "processing_time": 0.0,
            "api_calls_made": 0
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the validation system"""
        return {
            "system_name": "Pragati CrewAI Multi-Agent Validator",
            "version": "1.0.0",
            "total_agents": len(self.agents),
            "cluster_distribution": self.agent_factory.get_agent_count_by_cluster(),
            "dependency_graph": self.agent_factory.get_dependency_graph(),
            "llm_model": "gpt-4",
            "capabilities": [
                "109+ specialized validation agents",
                "Inter-agent collaboration and dependency resolution", 
                "Comprehensive Indian market analysis",
                "Real-time consensus building",
                "Detailed HTML reporting"
            ]
        }


# Global instance for Flask integration
_pragati_validator = None

def get_pragati_validator() -> PragatiCrewAIValidator:
    """Get global validator instance (singleton pattern)"""
    global _pragati_validator
    if _pragati_validator is None:
        _pragati_validator = PragatiCrewAIValidator()
    return _pragati_validator


def validate_idea(idea_name: str, idea_concept: str, weights: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Main validation function for Flask application integration.
    This replaces the validate_idea function from ai_logic_v2.py
    
    Args:
        idea_name: Name of the idea
        idea_concept: Detailed description of the idea
        weights: Optional custom cluster weights
        
    Returns:
        Dictionary with validation results compatible with existing Flask app
    """
    try:
        validator = get_pragati_validator()
        
        # Convert weights to float if provided
        custom_weights = None
        if weights:
            custom_weights = {k: float(v) for k, v in weights.items()}
        
        return validator.validate_idea(idea_name, idea_concept, custom_weights)
        
    except Exception as e:
        print(f"Error in CrewAI validation: {e}")
        return {
            "overall_score": 3.0,
            "validation_outcome": "ERROR",
            "evaluated_data": {},
            "html_report": f"<h1>System Error</h1><p>Please try again later. Error: {str(e)}</p>",
            "error": str(e)
        }


def get_evaluation_framework_info() -> Dict[str, Any]:
    """Get comprehensive information about the evaluation framework (compatibility function)"""
    try:
        validator = get_pragati_validator()
        return validator.get_system_info()
    except Exception as e:
        return {
            "error": f"Failed to get framework info: {str(e)}",
            "version": "CrewAI_v1.0",
            "total_agents": 109
        }


def get_system_health() -> Dict[str, Any]:
    """Get system health information (compatibility function)"""
    try:
        validator = get_pragati_validator()
        return {
            "system_status": "operational",
            "ai_engine_available": True,
            "total_agents": len(validator.agents),
            "api_key_configured": bool(validator.openai_api_key),
            "framework_valid": True
        }
    except Exception as e:
        return {
            "system_status": "degraded",
            "ai_engine_available": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the system
    print("Testing Pragati CrewAI Multi-Agent Validation System")
    print("=" * 60)
    
    validator = PragatiCrewAIValidator()
    
    # Test validation
    test_result = validator.validate_idea(
        "AI-Powered Smart Farming Solution",
        "An IoT and AI-based platform that helps farmers optimize crop yields through real-time monitoring, predictive analytics, and automated irrigation systems designed specifically for Indian agricultural conditions."
    )
    
    print(f"Test validation completed:")
    print(f"Overall Score: {test_result['overall_score']:.2f}")
    print(f"Outcome: {test_result['validation_outcome']}")
    print(f"Processing Time: {test_result.get('processing_time', 0):.1f} seconds")
    print(f"Agents Consulted: {test_result.get('api_calls_made', 0)}")
