"""
CrewAI Multi-Agent Validation System - Core Module
This module coordinates 109+ specialized agents for comprehensive idea validation.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# TEST MODE - Set to True to use only 2 agents for testing
# ============================================
TEST_MODE = True  # Change to False to use all 109 agents
TEST_AGENT_LIMIT = 2  # Number of agents to use in test mode
# ============================================
# MOCK_MODE has been removed - all agents now run real evaluations
# ============================================

class ValidationOutcome(Enum):
    """Validation outcome categories (100-point scale)"""
    EXCELLENT = "EXCELLENT"      # 90-100
    GOOD = "GOOD"               # 70-89
    MODERATE = "MODERATE"       # 50-69
    WEAK = "WEAK"              # 30-49
    POOR = "POOR"              # 0-29

@dataclass
class AgentEvaluation:
    """Individual agent evaluation result (100-point scale)"""
    agent_id: str
    parameter_name: str
    cluster: str
    sub_cluster: str
    sub_parameter: str
    assigned_score: float  # 0-100 scale
    confidence_level: float  # 0.0-1.0
    explanation: str
    assumptions: List[str]
    dependencies: List[str]  # Other agents this depends on
    weight_contribution: float
    processing_time: float
    timestamp: str
    # Enhanced fields from agent output (bullet points)
    key_insights: List[str] = None
    strengths: List[str] = None  # NEW: Explicit strengths
    weaknesses: List[str] = None  # NEW: Explicit weaknesses
    recommendations: List[str] = None
    risk_factors: List[str] = None
    peer_challenges: List[str] = None
    evidence_gaps: List[str] = None
    indian_market_considerations: str = None

@dataclass
class ValidationResult:
    """Complete validation result from all agents"""
    idea_id: str
    validation_id: str
    overall_score: float
    validation_outcome: ValidationOutcome
    agent_evaluations: List[AgentEvaluation]
    cluster_scores: Dict[str, float]
    collaboration_insights: List[str]
    total_processing_time: float
    total_agents_consulted: int
    consensus_level: float  # How much agents agreed
    timestamp: str
    overall_summary: Optional[str] = None
    cluster_summaries: Optional[Dict[str, str]] = None
    key_recommendations: Optional[List[str]] = None
    critical_risks: Optional[List[str]] = None
    market_insights: Optional[List[str]] = None

class CrewAIValidationOrchestrator:
    """
    Main orchestrator for the 109+ agent validation system
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Fast LLM for quick processing (under 20 mins)
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.2,  # Lower for faster, more focused responses
            model="gpt-4.1-mini",  # Faster model
            request_timeout=45,  # 45 second timeout per agent
            max_retries=1  # Only 1 retry to save time
        )
        
        # Initialize agent registry
        self.agent_registry = {}
        self.task_registry = {}
        self.crew = None
        
        # Load evaluation framework
        self.evaluation_framework = self._load_evaluation_framework()
        
        # Initialize all agents
        self._initialize_all_agents()
    
    def _load_evaluation_framework(self) -> Dict[str, Any]:
        """Load the complete evaluation framework with 109 parameters"""
        return {
            "Core Idea": {
                "Novelty & Uniqueness": {
                    "Originality": {"weight": 30, "dependencies": ["Innovation Index"]},
                    "Differentiation": {"weight": 25, "dependencies": ["Market Gap Analysis"]},
                    "Innovation Index": {"weight": 25, "dependencies": []},
                    "Disruptive Potential": {"weight": 20, "dependencies": ["Technology Maturity"]}
                },
                "Problem-Solution Fit": {
                    "Problem Severity": {"weight": 25, "dependencies": []},
                    "Solution Effectiveness": {"weight": 25, "dependencies": ["Technical Feasibility"]},
                    "Market Gap Analysis": {"weight": 20, "dependencies": ["Market Size (TAM)"]},
                    "Customer Pain Validation": {"weight": 15, "dependencies": ["User Engagement"]},
                    "Solution Uniqueness": {"weight": 15, "dependencies": ["Originality"]}
                },
                "UX/Usability Potential": {
                    "Intuitive Design": {"weight": 30, "dependencies": []},
                    "Accessibility Compliance": {"weight": 25, "dependencies": ["Regulatory Landscape"]},
                    "User Interface Quality": {"weight": 20, "dependencies": ["Intuitive Design"]},
                    "Mobile Responsiveness": {"weight": 15, "dependencies": []},
                    "Cross-Platform Compatibility": {"weight": 10, "dependencies": ["Technical Architecture"]}
                }
            },
            "Market Opportunity": {
                "Market Validation": {
                    "Market Size (TAM)": {"weight": 25, "dependencies": []},
                    "Competitive Intensity": {"weight": 20, "dependencies": ["Market Size (TAM)"]},
                    "Market Growth Rate": {"weight": 20, "dependencies": ["Market Size (TAM)"]},
                    "Customer Acquisition Potential": {"weight": 15, "dependencies": ["User Engagement"]},
                    "Market Penetration Strategy": {"weight": 10, "dependencies": ["Cultural Adaptation"]},
                    "Timing & Market Readiness": {"weight": 10, "dependencies": ["Infrastructure Readiness"]}
                },
                "Geographic Specificity (India)": {
                    "Regulatory Landscape": {"weight": 25, "dependencies": []},
                    "Infrastructure Readiness": {"weight": 25, "dependencies": []},
                    "Local Market Understanding": {"weight": 20, "dependencies": ["Cultural Adaptation"]},
                    "Cultural Adaptation": {"weight": 15, "dependencies": []},
                    "Regional Expansion Potential": {"weight": 15, "dependencies": ["Infrastructure Readiness"]}
                },
                "Product-Market Fit": {
                    "User Engagement": {"weight": 20, "dependencies": ["Intuitive Design"]},
                    "Retention Potential": {"weight": 20, "dependencies": ["Product Stickiness"]},
                    "Customer Satisfaction Metrics": {"weight": 20, "dependencies": ["Solution Effectiveness"]},
                    "Product Stickiness": {"weight": 15, "dependencies": ["Network Effects"]},
                    "Market Feedback Integration": {"weight": 15, "dependencies": ["Process Efficiency"]},
                    "Viral Coefficient": {"weight": 10, "dependencies": ["Network Effects"]}
                }
            },
            "Execution": {
                "Technical Feasibility": {
                    "Technology Maturity": {"weight": 20, "dependencies": []},
                    "Scalability & Performance": {"weight": 20, "dependencies": ["Technology Maturity"]},
                    "Technical Architecture": {"weight": 15, "dependencies": ["Technology Maturity"]},
                    "Development Complexity": {"weight": 15, "dependencies": ["Technical Architecture"]},
                    "Security Framework": {"weight": 15, "dependencies": ["Data Privacy Compliance"]},
                    "API Integration Capability": {"weight": 15, "dependencies": ["Technical Architecture"]}
                },
                "Operational Viability": {
                    "Resource Availability": {"weight": 20, "dependencies": []},
                    "Process Efficiency": {"weight": 20, "dependencies": ["Resource Availability"]},
                    "Supply Chain Management": {"weight": 15, "dependencies": ["Process Efficiency"]},
                    "Quality Assurance": {"weight": 15, "dependencies": ["Process Efficiency"]},
                    "Operational Scalability": {"weight": 15, "dependencies": ["Process Efficiency"]},
                    "Cost Structure Optimization": {"weight": 15, "dependencies": ["Unit Economics"]}
                },
                "Scalability Potential": {
                    "Business Model Scalability": {"weight": 20, "dependencies": ["Financial Viability"]},
                    "Market Expansion Potential": {"weight": 20, "dependencies": ["Market Size (TAM)"]},
                    "Technology Scalability": {"weight": 15, "dependencies": ["Scalability & Performance"]},
                    "Operational Scalability": {"weight": 15, "dependencies": ["Process Efficiency"]},
                    "Financial Scalability": {"weight": 15, "dependencies": ["Revenue Stream Diversity"]},
                    "International Expansion": {"weight": 15, "dependencies": ["Cultural Adaptation"]}
                }
            },
            "Business Model": {
                "Financial Viability": {
                    "Revenue Stream Diversity": {"weight": 20, "dependencies": []},
                    "Profitability & Margins": {"weight": 20, "dependencies": ["Unit Economics"]},
                    "Cash Flow Sustainability": {"weight": 15, "dependencies": ["Revenue Stream Diversity"]},
                    "Customer Lifetime Value": {"weight": 15, "dependencies": ["Retention Potential"]},
                    "Unit Economics": {"weight": 15, "dependencies": ["Revenue Stream Diversity"]},
                    "Financial Projections Accuracy": {"weight": 15, "dependencies": ["Market Size (TAM)"]}
                },
                "Defensibility": {
                    "Intellectual Property (IP)": {"weight": 20, "dependencies": ["Originality"]},
                    "Network Effects": {"weight": 20, "dependencies": ["User Engagement"]},
                    "Brand Moat": {"weight": 15, "dependencies": ["Differentiation"]},
                    "Data Moat": {"weight": 15, "dependencies": ["User Engagement"]},
                    "Switching Costs": {"weight": 15, "dependencies": ["Product Stickiness"]},
                    "Regulatory Barriers": {"weight": 15, "dependencies": ["Regulatory Landscape"]}
                }
            },
            "Team": {
                "Founder-Fit": {
                    "Relevant Experience": {"weight": 20, "dependencies": []},
                    "Complementary Skills": {"weight": 20, "dependencies": ["Relevant Experience"]},
                    "Industry Expertise": {"weight": 15, "dependencies": ["Relevant Experience"]},
                    "Leadership Capability": {"weight": 15, "dependencies": []},
                    "Execution Track Record": {"weight": 15, "dependencies": ["Leadership Capability"]},
                    "Domain Knowledge": {"weight": 15, "dependencies": ["Industry Expertise"]}
                },
                "Culture/Values": {
                    "Mission Alignment": {"weight": 20, "dependencies": []},
                    "Diversity & Inclusion": {"weight": 20, "dependencies": []},
                    "Team Dynamics": {"weight": 15, "dependencies": ["Communication Effectiveness"]},
                    "Communication Effectiveness": {"weight": 15, "dependencies": []},
                    "Adaptability": {"weight": 15, "dependencies": ["Team Dynamics"]},
                    "Work Ethics & Values": {"weight": 15, "dependencies": ["Mission Alignment"]}
                }
            },
            "Compliance": {
                "Regulatory (India)": {
                    "Data Privacy Compliance": {"weight": 20, "dependencies": []},
                    "Sector-Specific Compliance": {"weight": 20, "dependencies": ["Regulatory Landscape"]},
                    "Tax Compliance": {"weight": 15, "dependencies": []},
                    "Labor Law Compliance": {"weight": 15, "dependencies": []},
                    "Import/Export Regulations": {"weight": 15, "dependencies": ["Regulatory Landscape"]},
                    "Digital India Compliance": {"weight": 15, "dependencies": ["Infrastructure Readiness"]}
                },
                "Sustainability (ESG)": {
                    "Environmental Impact": {"weight": 20, "dependencies": []},
                    "Social Impact (SDGs)": {"weight": 20, "dependencies": []},
                    "Governance Standards": {"weight": 15, "dependencies": ["Ethical Business Practices"]},
                    "Ethical Business Practices": {"weight": 15, "dependencies": []},
                    "Community Engagement": {"weight": 15, "dependencies": ["Social Impact (SDGs)"]},
                    "Carbon Footprint": {"weight": 15, "dependencies": ["Environmental Impact"]}
                },
                "Ecosystem Support (India)": {
                    "Government & Institutional Support": {"weight": 20, "dependencies": ["National Policy Alignment (India)"]},
                    "Investor & Partner Landscape": {"weight": 20, "dependencies": []},
                    "Startup Ecosystem Integration": {"weight": 15, "dependencies": ["Investor & Partner Landscape"]},
                    "Mentorship Availability": {"weight": 15, "dependencies": ["Academic Partnerships"]},
                    "Industry Associations": {"weight": 15, "dependencies": []},
                    "Academic Partnerships": {"weight": 15, "dependencies": ["Academic/Research Contribution"]}
                }
            },
            "Risk & Strategy": {
                "Risk Assessment": {
                    "Technical Risks": {"weight": 20, "dependencies": ["Development Complexity"]},
                    "Market Risks": {"weight": 20, "dependencies": ["Competitive Intensity"]},
                    "Financial Risks": {"weight": 15, "dependencies": ["Cash Flow Sustainability"]},
                    "Competitive Risks": {"weight": 15, "dependencies": ["Competitive Intensity"]},
                    "Regulatory Risks": {"weight": 15, "dependencies": ["Regulatory Landscape"]},
                    "Operational Risks": {"weight": 15, "dependencies": ["Operational Viability"]}
                },
                "Investor Attractiveness": {
                    "Valuation Potential": {"weight": 20, "dependencies": ["Market Size (TAM)"]},
                    "Exit Strategy Viability": {"weight": 20, "dependencies": ["Market Expansion Potential"]},
                    "ROI Potential": {"weight": 15, "dependencies": ["Profitability & Margins"]},
                    "Investment Stage Readiness": {"weight": 15, "dependencies": ["Financial Projections Accuracy"]},
                    "Due Diligence Preparedness": {"weight": 15, "dependencies": ["Governance Standards"]},
                    "Investor Fit": {"weight": 15, "dependencies": ["Investor & Partner Landscape"]}
                },
                "Academic/National Alignment": {
                    "National Policy Alignment (India)": {"weight": 20, "dependencies": []},
                    "Academic/Research Contribution": {"weight": 20, "dependencies": []},
                    "Innovation Ecosystem Impact": {"weight": 15, "dependencies": ["Academic/Research Contribution"]},
                    "Knowledge Transfer Potential": {"weight": 15, "dependencies": ["Academic/Research Contribution"]},
                    "Research Commercialization": {"weight": 15, "dependencies": ["Knowledge Transfer Potential"]},
                    "Educational Value": {"weight": 15, "dependencies": ["Academic/Research Contribution"]}
                }
            }
        }
    
    def _initialize_all_agents(self):
        """Initialize specialized agents (2 for testing, 109 for production)"""
        agent_count = 0
        
        if TEST_MODE:
            print(f"⚠️  TEST MODE: Initializing only {TEST_AGENT_LIMIT} agents for testing")
            print(f"⚠️  All other agents will be SKIPPED to save credits")
        
        for cluster_name, cluster_data in self.evaluation_framework.items():
            if TEST_MODE and agent_count >= TEST_AGENT_LIMIT:
                print(f"✅ Test mode: Reached limit, stopping initialization")
                break
                
            for parameter_name, parameter_data in cluster_data.items():
                if TEST_MODE and agent_count >= TEST_AGENT_LIMIT:
                    break
                    
                for sub_parameter_name, sub_parameter_config in parameter_data.items():
                    # TEST MODE: Stop BEFORE creating more agents
                    if TEST_MODE and agent_count >= TEST_AGENT_LIMIT:
                        break
                    
                    agent_count += 1
                    agent_id = f"agent_{agent_count:03d}_{sub_parameter_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').lower()}"
                    
                    print(f"  Creating agent {agent_count}/{TEST_AGENT_LIMIT if TEST_MODE else '109'}: {sub_parameter_name}")
                    
                    # Create specialized agent
                    agent = self._create_specialized_agent(
                        agent_id=agent_id,
                        cluster=cluster_name,
                        parameter=parameter_name,
                        sub_parameter=sub_parameter_name,
                        config=sub_parameter_config
                    )
                    
                    # Also create the BaseValidationAgent instance for method access
                    from .base_agent import BaseValidationAgent
                    base_agent = BaseValidationAgent(
                        agent_id=agent_id,
                        cluster=cluster_name,
                        parameter=parameter_name,
                        sub_parameter=sub_parameter_name,
                        weight=sub_parameter_config.get('weight', 10),
                        dependencies=sub_parameter_config.get('dependencies', []),
                        llm=self.llm
                    )
                    
                    self.agent_registry[agent_id] = {
                        'agent': agent,  # CrewAI Agent
                        'base_agent': base_agent,  # BaseValidationAgent instance
                        'cluster': cluster_name,
                        'parameter': parameter_name,
                        'sub_parameter': sub_parameter_name,
                        'config': sub_parameter_config
                    }
                    
                    # TEST MODE: Stop immediately after reaching limit
                    if TEST_MODE and agent_count >= TEST_AGENT_LIMIT:
                        print(f"✅ Test mode: Reached {agent_count} agents, stopping now")
                        break
        
        mode_str = "TEST MODE" if TEST_MODE else "PRODUCTION MODE"
        print(f"✅ Initialized {agent_count} specialized validation agents ({mode_str})")
        print(f"📊 Total agents in registry: {len(self.agent_registry)}")
        
        # Test broadcast to verify connection
        try:
            self._broadcast_message("System", f"Initialized {agent_count} validation agents", "system")
        except Exception as e:
            print(f"Broadcast test failed: {e}")
    
    def _agent_step_callback(self, step):
        """Callback function to display real-time agent communication"""
        try:
            if hasattr(step, 'agent') and hasattr(step, 'output'):
                agent_name = getattr(step.agent, 'role', 'Unknown Agent')
                output = str(step.output)[:200] + "..." if len(str(step.output)) > 200 else str(step.output)
                print(f"\n🤖 {agent_name}: {output}")
                # Broadcast to web clients
                self._broadcast_message(agent_name, output, "agent")
            elif hasattr(step, 'message'):
                print(f"\n📢 System: {step.message}")
                # Broadcast to web clients
                self._broadcast_message("System", step.message, "system")
        except Exception as e:
            # Silently handle callback errors to not interrupt the main flow
            pass
    
    def _broadcast_message(self, agent_name, message, message_type="info"):
        """Broadcast message to web clients if available"""
        try:
            # Try to access the global broadcast function
            import builtins
            if hasattr(builtins, 'broadcast_agent_message'):
                builtins.broadcast_agent_message(agent_name, message, message_type)
            else:
                # Try direct import approach
                try:
                    import sys
                    import os
                    
                    # Add the app directory to the path if not already there
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(app_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    
                    # Try to import from the app module
                    if 'app_v3' in sys.modules:
                        broadcast_func = getattr(sys.modules['app_v3'], 'broadcast_agent_message', None)
                        if broadcast_func:
                            broadcast_func(agent_name, message, message_type)
                    else:
                        # Try to import the module
                        import app_v3
                        app_v3.broadcast_agent_message(agent_name, message, message_type)
                        
                except (ImportError, AttributeError, Exception) as e:
                    print(f"Debug: Broadcasting failed: {e}")
                    pass
        except Exception as e:
            print(f"Debug: Broadcasting error: {e}")
            pass
    
    def _create_specialized_agent(self, agent_id: str, cluster: str, parameter: str, 
                                sub_parameter: str, config: Dict[str, Any]) -> Agent:
        """Create a specialized agent for a specific validation parameter"""
        
        # Import BaseValidationAgent here to avoid circular imports
        from .base_agent import BaseValidationAgent
        
        # Create BaseValidationAgent instance
        base_agent = BaseValidationAgent(
            agent_id=agent_id,
            cluster=cluster,
            parameter=parameter,
            sub_parameter=sub_parameter,
            weight=config.get('weight', 10),
            dependencies=config.get('dependencies', []),
            llm=self.llm
        )
        
        # Return the CrewAI Agent created by the base agent
        return base_agent.create_agent()
    
    def _generate_agent_role(self, cluster: str, parameter: str, sub_parameter: str) -> str:
        """Generate specific role description for each agent"""
        role_templates = {
            # Core Idea agents
            "Originality": "Expert in assessing the novelty and uniqueness of business concepts",
            "Differentiation": "Specialist in competitive analysis and unique value proposition evaluation",
            "Innovation Index": "Innovation measurement expert focusing on disruptive potential",
            "Disruptive Potential": "Disruption analysis specialist evaluating market transformation potential",
            "Problem Severity": "Problem validation expert assessing market pain points",
            "Solution Effectiveness": "Solution evaluation specialist measuring problem-solution fit",
            "Market Gap Analysis": "Market gap identification expert analyzing unmet needs",
            "Customer Pain Validation": "Customer research specialist validating pain points",
            "Solution Uniqueness": "Solution differentiation expert evaluating competitive advantages",
            "Intuitive Design": "UX design expert evaluating user experience potential",
            "Accessibility Compliance": "Accessibility specialist ensuring inclusive design",
            "User Interface Quality": "UI/UX expert assessing interface design quality",
            "Mobile Responsiveness": "Mobile experience specialist evaluating cross-device compatibility",
            "Cross-Platform Compatibility": "Platform integration expert assessing technical compatibility",
            
            # Market Opportunity agents
            "Market Size (TAM)": "Market sizing expert calculating total addressable market",
            "Competitive Intensity": "Competition analysis specialist evaluating market dynamics",
            "Market Growth Rate": "Market growth expert analyzing expansion potential",
            "Customer Acquisition Potential": "Customer acquisition specialist evaluating reach strategies",
            "Market Penetration Strategy": "Market entry expert designing penetration approaches",
            "Timing & Market Readiness": "Market timing specialist assessing entry opportunities",
            "Regulatory Landscape": "Regulatory compliance expert for Indian market conditions",
            "Infrastructure Readiness": "Infrastructure assessment specialist evaluating market readiness",
            "Local Market Understanding": "Local market expert analyzing cultural and regional factors",
            "Cultural Adaptation": "Cultural adaptation specialist for Indian market context",
            "Regional Expansion Potential": "Regional growth expert evaluating expansion opportunities",
            "User Engagement": "User engagement specialist measuring interaction potential",
            "Retention Potential": "Customer retention expert evaluating loyalty factors",
            "Customer Satisfaction Metrics": "Customer satisfaction specialist measuring experience quality",
            "Product Stickiness": "Product adoption expert evaluating user dependency",
            "Market Feedback Integration": "Feedback analysis specialist evaluating iteration potential",
            "Viral Coefficient": "Viral growth expert measuring organic expansion potential",
            
            # Default for others
            "default": f"Specialized validation expert for {sub_parameter} assessment"
        }
        
        return role_templates.get(sub_parameter, role_templates["default"])
    
    def _generate_agent_goal(self, cluster: str, parameter: str, sub_parameter: str) -> str:
        """Generate specific goal for each agent"""
        return f"Provide expert evaluation of {sub_parameter} for startup ideas with precise scoring (1.0-5.0), detailed analysis, and actionable insights within the {cluster} evaluation framework."
    
    def _generate_agent_backstory(self, cluster: str, parameter: str, sub_parameter: str) -> str:
        """Generate contextual backstory for each agent"""
        backstory_templates = {
            "Core Idea": "You are a seasoned innovation consultant with deep expertise in evaluating breakthrough ideas and disruptive technologies. You have helped assess hundreds of startups and understand what makes ideas truly innovative.",
            "Market Opportunity": "You are a market research expert with extensive experience in the Indian startup ecosystem. You understand market dynamics, customer behavior, and growth potential in emerging markets.",
            "Execution": "You are a technical and operational expert who has guided numerous startups through execution challenges. You understand the complexities of building and scaling technology solutions.",
            "Business Model": "You are a business strategy expert with deep knowledge of sustainable business models and financial viability. You have experience with venture capital and startup valuations.",
            "Team": "You are an organizational development expert who understands what makes high-performing teams. You have experience in founder coaching and team building for startups.",
            "Compliance": "You are a regulatory and compliance expert with specialized knowledge of Indian business environment, ESG principles, and ecosystem dynamics.",
            "Risk & Strategy": "You are a strategic risk assessment expert who helps startups navigate uncertainties and position themselves for investment and growth opportunities."
        }
        
        base_backstory = backstory_templates.get(cluster, "You are a specialized validation expert with deep domain knowledge.")
        return f"{base_backstory} Your specific expertise lies in {sub_parameter} evaluation, and you collaborate with other specialists to provide comprehensive assessments."
    
    async def validate_idea(self, idea_name: str, idea_concept: str, 
                          custom_weights: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Main validation function using all 109+ agents
        """
        start_time = datetime.now()
        validation_id = f"VAL_{int(start_time.timestamp())}"
        
        print(f"\n🚀 Starting validation with {len(self.agent_registry)} specialized agents...")
        print(f"💡 Idea: {idea_name}")
        print("=" * 80)
        
        # Broadcast validation start
        self._broadcast_message("System", f"🚀 Starting validation with {len(self.agent_registry)} specialized agents", "system")
        self._broadcast_message("System", f"💡 Evaluating: {idea_name}", "info")
        
        # Create tasks for all agents
        tasks = []
        agent_evaluations = []
        
        # Phase 1: Independent evaluations (agents that don't depend on others)
        independent_agents = self._get_independent_agents()
        independent_tasks = await self._create_independent_tasks(
            independent_agents, idea_name, idea_concept
        )
        
        # Phase 2: Dependent evaluations (agents that need input from others)
        dependent_agents = self._get_dependent_agents()
        
        print(f"\n📊 PHASE 1: Independent Agent Analysis ({len(independent_agents)} agents)")
        print("=" * 80)
        # Execute in phases for proper dependency resolution
        phase1_results = await self._execute_agent_phase(independent_tasks)
        
        print(f"\n📊 PHASE 2: Dependent Agent Analysis ({len(dependent_agents)} agents)")
        print("=" * 80)
        phase2_results = await self._execute_dependent_phase(
            dependent_agents, idea_name, idea_concept, phase1_results
        )
        
        # Combine all results
        all_evaluations = phase1_results + phase2_results
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(all_evaluations, custom_weights)
        validation_outcome = self._determine_validation_outcome(overall_score)
        cluster_scores = self._calculate_cluster_scores(all_evaluations)
        collaboration_insights = self._generate_collaboration_insights(all_evaluations)
        consensus_level = self._calculate_consensus_level(all_evaluations)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate comprehensive summaries
        overall_summary = self._generate_overall_summary(idea_name, idea_concept, overall_score, validation_outcome, all_evaluations)
        cluster_summaries = self._generate_cluster_summaries(cluster_scores, all_evaluations)
        key_recommendations = self._generate_key_recommendations(all_evaluations)
        critical_risks = self._generate_critical_risks(all_evaluations)
        market_insights = self._generate_market_insights(all_evaluations)
        
        # Display final summary
        print(f"\n🎉 VALIDATION COMPLETE!")
        print("=" * 80)
        print(f"📊 Overall Score: {overall_score:.2f}/5.0")
        print(f"🏆 Outcome: {validation_outcome.value}")
        print(f"🤖 Agents Consulted: {len(all_evaluations)}")
        print(f"⏱️ Total Time: {processing_time:.2f} seconds")
        print(f"🤝 Consensus Level: {consensus_level:.1%}")
        print("\n📈 Cluster Summary:")
        for cluster, score in cluster_scores.items():
            emoji = "🟢" if score >= 4.0 else "🟡" if score >= 3.0 else "🔴"
            print(f"   {emoji} {cluster}: {score:.2f}/5.0")
        
        print(f"\n📋 Overall Summary:")
        print(overall_summary)
        
        # Broadcast completion
        self._broadcast_message("System", f"🎉 VALIDATION COMPLETE! Overall Score: {overall_score:.2f}/5.0", "success")
        
        return ValidationResult(
            idea_id=f"IDEA_{int(start_time.timestamp())}",
            validation_id=validation_id,
            overall_score=overall_score,
            validation_outcome=validation_outcome,
            agent_evaluations=all_evaluations,
            cluster_scores=cluster_scores,
            collaboration_insights=collaboration_insights,
            total_processing_time=processing_time,
            total_agents_consulted=len(all_evaluations),
            consensus_level=consensus_level,
            timestamp=start_time.isoformat(),
            overall_summary=overall_summary,
            cluster_summaries=cluster_summaries,
            key_recommendations=key_recommendations,
            critical_risks=critical_risks,
            market_insights=market_insights
        )
    
    def _get_independent_agents(self) -> List[str]:
        """Get agents that don't depend on others"""
        independent = []
        for agent_id, agent_info in self.agent_registry.items():
            if not agent_info['config'].get('dependencies'):
                independent.append(agent_id)
        return independent
    
    def _get_dependent_agents(self) -> List[str]:
        """Get agents that depend on others"""
        dependent = []
        for agent_id, agent_info in self.agent_registry.items():
            if agent_info['config'].get('dependencies'):
                dependent.append(agent_id)
        return dependent
    
    async def _create_independent_tasks(self, agent_ids: List[str], 
                                      idea_name: str, idea_concept: str) -> List[Task]:
        """Create tasks for independent agents"""
        tasks = []
        for agent_id in agent_ids:
            agent_info = self.agent_registry[agent_id]
            task = Task(
                description=self._create_task_description(
                    agent_info, idea_name, idea_concept
                ),
                agent=agent_info['agent'],
                expected_output="JSON object with score (1.0-5.0), explanation, assumptions, and confidence_level"
            )
            tasks.append((agent_id, task))
        return tasks
    
    def _create_task_description(self, agent_info: Dict[str, Any], 
                               idea_name: str, idea_concept: str, 
                               dependency_results: Optional[Dict[str, Any]] = None) -> str:
        """Create detailed task description for agent"""
        cluster = agent_info['cluster']
        parameter = agent_info['parameter']
        sub_parameter = agent_info['sub_parameter']
        
        base_description = f"""
        Evaluate the startup idea '{idea_name}' for {sub_parameter} within the {cluster} framework.
        
        Idea Description: {idea_concept}
        
        Your evaluation should consider:
        1. The specific aspect of {sub_parameter}
        2. Indian market context and conditions
        3. Current technology and market trends
        4. Practical implementation challenges
        5. Scalability and sustainability factors
        
        Provide a comprehensive assessment including:
        - Numerical score (1.0-5.0 where 5.0 is excellent)
        - Detailed explanation of your reasoning
        - Key assumptions made in your analysis
        - Confidence level in your assessment (0.0-1.0)
        """
        
        if dependency_results:
            dependency_info = "Previous agent insights to consider:\n"
            for dep_agent, dep_result in dependency_results.items():
                dependency_info += f"- {dep_agent}: {dep_result.get('explanation', 'N/A')}\n"
            base_description += f"\n{dependency_info}"
        
        return base_description
    
    async def _execute_single_agent(self, agent_id: str, task: Task) -> AgentEvaluation:
        """Execute a single agent task"""
        agent_info = self.agent_registry[agent_id]
        start_time = datetime.now()
        
        # Execute the agent with real AI evaluation
        try:
            print(f"\n🔍 {agent_info['sub_parameter']} ({agent_info['cluster']}) - Starting")
            
            # Execute the task with minimal verbosity for speed
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,  # Disable verbose for speed
                output_log_file=False,
                planning=False,
                memory=False
            )
            
            self._broadcast_message("System", f"🚀 {agent_info['sub_parameter']} analyzing...", "system")
            result = crew.kickoff()
            
            # Parse result
            evaluation = self._parse_agent_result(agent_id, result, start_time)
            
            print(f"   ✅ {agent_info['sub_parameter']} completed! Score: {evaluation.assigned_score:.1f}/100")
            self._broadcast_message("System", f"✅ {agent_info['sub_parameter']}: {evaluation.assigned_score:.1f}/100", "success")
            
            return evaluation
            
        except Exception as e:
            print(f"   ❌ {agent_info['sub_parameter']} error: {str(e)[:50]}")
            return self._create_fallback_evaluation(agent_id, start_time)
    
    async def _execute_agent_phase(self, tasks: List[tuple]) -> List[AgentEvaluation]:
        """Execute agent tasks in parallel batches for speed"""
        results = []
        batch_size = 10  # Process 10 agents at a time
        
        print(f"\n🚀 Processing {len(tasks)} agents in parallel batches of {batch_size}")
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            print(f"\n📦 Batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
            
            # Execute batch in parallel
            batch_tasks = [self._execute_single_agent(agent_id, task) for agent_id, task in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions and add valid results
            for result in batch_results:
                if isinstance(result, AgentEvaluation):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"   ⚠️ Batch exception: {str(result)[:50]}")
        
        return results
    
    async def _execute_dependent_phase(self, dependent_agents: List[str], 
                                     idea_name: str, idea_concept: str,
                                     previous_results: List[AgentEvaluation]) -> List[AgentEvaluation]:
        """Execute agents that depend on previous results"""
        results = []
        results_dict = {eval.agent_id: asdict(eval) for eval in previous_results}
        
        for agent_id in dependent_agents:
            agent_info = self.agent_registry[agent_id]
            dependencies = agent_info['config'].get('dependencies', [])
            
            print(f"\n🔍 Starting Dependent Agent: {agent_info['sub_parameter']} ({agent_info['cluster']})")
            print(f"   📋 Dependencies: {', '.join(dependencies) if dependencies else 'None'}")
            
            # Broadcast dependent agent start with challenge context
            challenge_msg = f"🥊 {agent_info['sub_parameter']} agent starting - will challenge {len(dependencies)} previous assessments"
            self._broadcast_message("System", challenge_msg, "challenge")
            
            # Get dependency results
            dependency_results = {}
            for dep in dependencies:
                # Find agent that evaluates this dependency
                for prev_eval in previous_results:
                    if dep.lower().replace(' ', '_') in prev_eval.agent_id.lower():
                        dependency_results[dep] = asdict(prev_eval)
                        print(f"   🔗 Using input from {dep}: Score {prev_eval.assigned_score:.2f}")
                        break
            
            # Create task with dependency context
            task_description = self._create_task_description(
                agent_info, idea_name, idea_concept, dependency_results
            )
            
            task = Task(
                description=task_description,
                agent=agent_info['agent'],
                expected_output="JSON object with score, explanation, assumptions, and confidence_level"
            )
            
            start_time = datetime.now()
            try:
                crew = Crew(
                    agents=[agent_info['agent']],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,  # Enable verbose output for real-time agent communication
                    output_log_file=False,
                    step_callback=self._agent_step_callback,
                    planning=False,  # Disable planning to avoid prompts
                    memory=False  # Disable memory to avoid prompts
                )
                
                print(f"   🚀 Agent {agent_info['sub_parameter']} is analyzing with dependencies...")
                result = crew.kickoff()
                evaluation = self._parse_agent_result(agent_id, result, start_time)
                results.append(evaluation)
                
                # Show completion status
                print(f"   ✅ Agent {agent_info['sub_parameter']} completed!")
                print(f"   📊 Score: {evaluation.assigned_score:.2f}/5.0")
                self._broadcast_message("System", f"✅ {agent_info['sub_parameter']} completed! Score: {evaluation.assigned_score:.2f}/5.0", "success")
                print(f"   💡 Key insight: {evaluation.explanation[:100]}...")
                
            except Exception as e:
                print(f"Error executing dependent agent {agent_id}: {e}")
                evaluation = self._create_fallback_evaluation(agent_id, start_time)
                results.append(evaluation)
        
        return results
    
    def _ensure_string_explanation(self, explanation: Any) -> str:
        """Ensure explanation is always a string, converting lists if needed"""
        if explanation is None:
            return 'Analysis completed'
        if isinstance(explanation, list):
            # Join list items with periods
            return '. '.join(str(item) for item in explanation if item)
        return str(explanation)
    
    def _parse_agent_result(self, agent_id: str, result: Any, start_time: datetime) -> AgentEvaluation:
        """Parse agent execution result into structured format (100-scale scoring)"""
        agent_info = self.agent_registry[agent_id]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        try:
            # Handle CrewAI CrewOutput object
            if hasattr(result, 'raw'):
                result_text = str(result.raw)
            elif hasattr(result, 'output'):
                result_text = str(result.output)
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = str(result)
            
            # Try to parse JSON from text
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
            else:
                # Create from string analysis
                result_data = self._extract_data_from_text(result_text)
            
            # Get score and normalize if needed
            score = float(result_data.get('score', 60.0))
            # If score is in old 5.0 format (<=5), convert to 100
            if score <= 5.0:
                score = score * 20
            # Clamp to valid range
            score = max(0.0, min(100.0, score))
            
            # Ensure explanation is always a string (fix for list issue)
            raw_explanation = result_data.get('explanation', 'Analysis completed')
            explanation_str = self._ensure_string_explanation(raw_explanation)
            
            return AgentEvaluation(
                agent_id=agent_id,
                parameter_name=agent_info['sub_parameter'],
                cluster=agent_info['cluster'],
                sub_cluster=agent_info['parameter'],
                sub_parameter=agent_info['sub_parameter'],
                assigned_score=score,  # Now 0-100 scale
                confidence_level=float(result_data.get('confidence_level', 0.7)),
                explanation=explanation_str,
                assumptions=result_data.get('assumptions', []),
                dependencies=agent_info['config'].get('dependencies', []),
                weight_contribution=agent_info['config'].get('weight', 20),
                processing_time=processing_time,
                timestamp=start_time.isoformat(),
                # Enhanced fields from agent output (bullet points from agents)
                key_insights=result_data.get('key_insights', []),
                strengths=result_data.get('strengths', []),  # NEW from agents
                weaknesses=result_data.get('weaknesses', []),  # NEW from agents
                recommendations=result_data.get('recommendations', []),
                risk_factors=result_data.get('risk_factors', []),
                peer_challenges=result_data.get('peer_challenges', []),
                evidence_gaps=result_data.get('evidence_gaps', []),
                indian_market_considerations=result_data.get('indian_market_considerations', '')
            )
            
        except Exception as e:
            print(f"Error parsing result for {agent_id}: {e}")
            return self._create_fallback_evaluation(agent_id, start_time)
    
    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract evaluation data from text when JSON parsing fails (100-scale)"""
        # Simple text analysis fallback
        score = 60.0  # Default middle score on 100 scale
        explanation = text[:200] if text else "Analysis completed"
        
        # Try to extract score from text
        import re
        score_matches = re.findall(r'score[:\s]*([0-9\.]+)', text.lower())
        if score_matches:
            try:
                score = float(score_matches[0])
                # If score is small (<=5), assume it's old format
                if score <= 5.0:
                    score = score * 20
                score = max(0.0, min(100.0, score))  # Clamp to valid range
            except:
                pass
        
        return {
            'score': score,
            'explanation': explanation,
            'assumptions': ['Extracted from text analysis'],
            'confidence_level': 0.6,
            'key_insights': [],
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'risk_factors': []
        }
    
    def _create_fallback_evaluation(self, agent_id: str, start_time: datetime) -> AgentEvaluation:
        """Create fallback evaluation when agent fails (100-scale)"""
        agent_info = self.agent_registry[agent_id]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentEvaluation(
            agent_id=agent_id,
            parameter_name=agent_info['sub_parameter'],
            cluster=agent_info['cluster'],
            sub_cluster=agent_info['parameter'],
            sub_parameter=agent_info['sub_parameter'],
            assigned_score=60.0,  # Neutral score on 100 scale
            confidence_level=0.5,
            explanation=f"Fallback evaluation for {agent_info['sub_parameter']} due to processing error",
            assumptions=["Fallback evaluation applied"],
            dependencies=agent_info['config'].get('dependencies', []),
            weight_contribution=agent_info['config'].get('weight', 20),
            processing_time=processing_time,
            timestamp=start_time.isoformat(),
            # Enhanced fields with fallback values
            key_insights=[],
            strengths=[],
            weaknesses=[],
            recommendations=[],
            risk_factors=[],
            peer_challenges=[],
            evidence_gaps=[],
            indian_market_considerations=''
        )
    
    def _calculate_overall_score(self, evaluations: List[AgentEvaluation], 
                               custom_weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall score from all agent evaluations (100-scale)"""
        if not evaluations:
            return 60.0  # Neutral score on 100 scale
        
        # Calculate cluster weights
        cluster_weights = {
            "Core Idea": 15,
            "Market Opportunity": 20,
            "Execution": 20,
            "Business Model": 15,
            "Team": 10,
            "Compliance": 10,
            "Risk & Strategy": 10
        }
        
        if custom_weights:
            cluster_weights.update(custom_weights)
        
        # Group by cluster
        cluster_evaluations = {}
        for eval in evaluations:
            if eval.cluster not in cluster_evaluations:
                cluster_evaluations[eval.cluster] = []
            cluster_evaluations[eval.cluster].append(eval)
        
        # Calculate weighted score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for cluster, weight in cluster_weights.items():
            if cluster in cluster_evaluations:
                cluster_evals = cluster_evaluations[cluster]
                cluster_score = sum(eval.assigned_score * eval.weight_contribution 
                                  for eval in cluster_evals) / sum(eval.weight_contribution 
                                                                 for eval in cluster_evals)
                total_weighted_score += cluster_score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 60.0
    
    def _determine_validation_outcome(self, overall_score: float) -> ValidationOutcome:
        """Determine validation outcome based on overall score (100-scale)"""
        if overall_score >= 90:
            return ValidationOutcome.EXCELLENT
        elif overall_score >= 70:
            return ValidationOutcome.GOOD
        elif overall_score >= 50:
            return ValidationOutcome.MODERATE
        elif overall_score >= 30:
            return ValidationOutcome.WEAK
        else:
            return ValidationOutcome.POOR
    
    def _calculate_cluster_scores(self, evaluations: List[AgentEvaluation]) -> Dict[str, float]:
        """Calculate average scores for each cluster"""
        cluster_scores = {}
        cluster_evaluations = {}
        
        for eval in evaluations:
            if eval.cluster not in cluster_evaluations:
                cluster_evaluations[eval.cluster] = []
            cluster_evaluations[eval.cluster].append(eval)
        
        for cluster, evals in cluster_evaluations.items():
            cluster_scores[cluster] = sum(eval.assigned_score for eval in evals) / len(evals)
        
        return cluster_scores
    
    def _generate_collaboration_insights(self, evaluations: List[AgentEvaluation]) -> List[str]:
        """Generate insights from agent collaboration patterns"""
        insights = []
        
        # Analyze consensus patterns
        scores = [eval.assigned_score for eval in evaluations]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        
        if score_variance < 0.5:
            insights.append("High consensus among evaluation agents - consistent assessment across parameters")
        elif score_variance > 2.0:
            insights.append("Significant disagreement among agents - idea shows mixed strengths and weaknesses")
        
        # Analyze dependency relationships
        dependent_agents = [eval for eval in evaluations if eval.dependencies]
        if dependent_agents:
            insights.append(f"Multi-agent collaboration involved {len(dependent_agents)} dependent evaluations")
        
        # Identify standout clusters
        cluster_scores = self._calculate_cluster_scores(evaluations)
        best_cluster = max(cluster_scores.items(), key=lambda x: x[1])
        worst_cluster = min(cluster_scores.items(), key=lambda x: x[1])
        
        insights.append(f"Strongest area: {best_cluster[0]} (Score: {best_cluster[1]:.2f})")
        insights.append(f"Area for improvement: {worst_cluster[0]} (Score: {worst_cluster[1]:.2f})")
        
        return insights
    
    def _calculate_consensus_level(self, evaluations: List[AgentEvaluation]) -> float:
        """Calculate how much the agents agreed with each other"""
        if len(evaluations) < 2:
            return 1.0
        
        scores = [eval.assigned_score for eval in evaluations]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to consensus (lower variance = higher consensus)
        # Max variance would be 4.0 (scores ranging from 1 to 5)
        consensus = max(0.0, 1.0 - (variance / 4.0))
        return consensus
    
    def get_agent_count(self) -> int:
        """Get total number of agents in the system"""
        return len(self.agent_registry)
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the evaluation framework"""
        return {
            "total_agents": len(self.agent_registry),
            "clusters": list(self.evaluation_framework.keys()),
            "framework_version": "CrewAI_v1.0",
            "dependency_count": sum(1 for agent_info in self.agent_registry.values() 
                                  if agent_info['config'].get('dependencies'))
        }
    
    def _generate_overall_summary(self, idea_name: str, idea_concept: str, overall_score: float, 
                                validation_outcome: ValidationOutcome, evaluations: List[AgentEvaluation]) -> str:
        """Generate comprehensive overall summary"""
        score_interpretation = {
            ValidationOutcome.EXCELLENT: "exceptional potential with outstanding market viability",
            ValidationOutcome.GOOD: "strong potential with good market prospects",
            ValidationOutcome.MODERATE: "moderate potential requiring significant improvements",
            ValidationOutcome.WEAK: "limited potential with substantial challenges",
            ValidationOutcome.POOR: "very limited potential with major fundamental issues"
        }
        
        # Get top strengths and weaknesses
        high_scores = [e for e in evaluations if e.assigned_score >= 4.0]
        low_scores = [e for e in evaluations if e.assigned_score <= 2.0]
        
        strengths = [f"{e.sub_parameter} ({e.assigned_score:.1f}/5.0)" for e in high_scores[:3]]
        weaknesses = [f"{e.sub_parameter} ({e.assigned_score:.1f}/5.0)" for e in low_scores[:3]]
        
        # Analyze agent evaluations for specific insights
        high_scores = sorted([e for e in evaluations if e.assigned_score >= 4.0], key=lambda x: x.assigned_score, reverse=True)
        low_scores = sorted([e for e in evaluations if e.assigned_score <= 2.5], key=lambda x: x.assigned_score)
        
        # Extract specific insights from agent explanations
        key_insights = []
        major_concerns = []
        
        for eval in high_scores[:3]:
            # Extract specific positive insights from actual explanations
            explanation_text = self._ensure_string_explanation(eval.explanation) if eval.explanation else f"Strong performance in {eval.sub_parameter}"
            explanation = explanation_text.split('.')[0] if explanation_text else f"Strong performance in {eval.sub_parameter}"
            key_insights.append(f"{eval.sub_parameter} ({eval.assigned_score:.1f}/5.0): {explanation}")
        
        for eval in low_scores[:3]:
            # Extract specific concerns from actual explanations
            explanation_text = self._ensure_string_explanation(eval.explanation) if eval.explanation else f"Challenges identified in {eval.sub_parameter}"
            explanation = explanation_text.split('.')[0] if explanation_text else f"Challenges identified in {eval.sub_parameter}"
            major_concerns.append(f"{eval.sub_parameter} ({eval.assigned_score:.1f}/5.0): {explanation}")
        
        # Generate market context from actual cluster scores
        market_evals = [e for e in evaluations if e.cluster == "Market Opportunity"]
        market_avg = sum(e.assigned_score for e in market_evals) / len(market_evals) if market_evals else 3.0
        
        execution_evals = [e for e in evaluations if e.cluster == "Execution"]
        execution_avg = sum(e.assigned_score for e in execution_evals) / len(execution_evals) if execution_evals else 3.0
        
        business_evals = [e for e in evaluations if e.cluster == "Business Model"]
        business_avg = sum(e.assigned_score for e in business_evals) / len(business_evals) if business_evals else 3.0
        
        summary = f"""The '{idea_name}' concept has undergone comprehensive evaluation by {len(evaluations)} specialized AI agents, achieving an overall score of {overall_score:.2f}/5.0.

CONCEPT OVERVIEW:
{idea_concept[:200]}{'...' if len(idea_concept) > 200 else ''}

KEY STRENGTHS IDENTIFIED:
{chr(10).join(f"• {insight}" for insight in key_insights) if key_insights else "• Limited standout strengths identified across evaluation parameters"}

CRITICAL CHALLENGES:
{chr(10).join(f"• {concern}" for concern in major_concerns) if major_concerns else "• No major fundamental issues identified"}

CLUSTER PERFORMANCE ANALYSIS:
• Market Opportunity: {market_avg:.2f}/5.0 - {'Strong market potential' if market_avg >= 3.5 else 'Moderate market challenges' if market_avg >= 2.5 else 'Significant market barriers'}
• Execution Feasibility: {execution_avg:.2f}/5.0 - {'Implementation appears feasible' if execution_avg >= 3.5 else 'Notable execution challenges' if execution_avg >= 2.5 else 'Major execution barriers'}
• Business Model: {business_avg:.2f}/5.0 - {'Solid business foundation' if business_avg >= 3.5 else 'Business model needs refinement' if business_avg >= 2.5 else 'Fundamental business model issues'}

FINAL ASSESSMENT:
This concept {'demonstrates strong potential for success with proper execution' if overall_score >= 4.0 else 'shows promise but requires significant improvements in key areas' if overall_score >= 3.0 else 'faces substantial challenges that must be addressed before proceeding'}.""".strip()
        
        return summary
    
    def _generate_cluster_summaries(self, cluster_scores: Dict[str, float], evaluations: List[AgentEvaluation]) -> Dict[str, str]:
        """Generate detailed summaries for each cluster"""
        summaries = {}
        
        for cluster, score in cluster_scores.items():
            cluster_evaluations = [e for e in evaluations if e.cluster == cluster]
            
            if score >= 4.0:
                status = "EXCELLENT"
                emoji = "🟢"
            elif score >= 3.0:
                status = "GOOD"
                emoji = "🟡"
            else:
                status = "NEEDS IMPROVEMENT"
                emoji = "🔴"
            
            # Get top and bottom performing parameters
            cluster_scores_list = [(e.sub_parameter, e.assigned_score) for e in cluster_evaluations]
            cluster_scores_list.sort(key=lambda x: x[1], reverse=True)
            
            top_performers = cluster_scores_list[:2]
            bottom_performers = cluster_scores_list[-2:]
            
            # Get detailed insights from actual agent explanations
            top_insights = []
            bottom_insights = []
            
            for param, param_score in top_performers:
                eval_obj = next((e for e in cluster_evaluations if e.sub_parameter == param), None)
                if eval_obj and eval_obj.explanation:
                    explanation_text = self._ensure_string_explanation(eval_obj.explanation)
                    explanation = explanation_text.split('.')[0] if explanation_text else "Strong performance in this area"  # First sentence
                    top_insights.append(f"{param} ({param_score:.1f}/5.0): {explanation}")
                else:
                    top_insights.append(f"{param} ({param_score:.1f}/5.0): Strong performance in this area")
            
            for param, param_score in bottom_performers:
                eval_obj = next((e for e in cluster_evaluations if e.sub_parameter == param), None)
                if eval_obj and eval_obj.explanation:
                    explanation_text = self._ensure_string_explanation(eval_obj.explanation)
                    explanation = explanation_text.split('.')[0] if explanation_text else "Areas for improvement identified"  # First sentence
                    bottom_insights.append(f"{param} ({param_score:.1f}/5.0): {explanation}")
                else:
                    bottom_insights.append(f"{param} ({param_score:.1f}/5.0): Areas for improvement identified")
            
            summary = f"""{emoji} {cluster} - {status} ({score:.2f}/5.0)

STRONG AREAS:
{chr(10).join(f"• {insight}" for insight in top_insights)}

IMPROVEMENT OPPORTUNITIES:
{chr(10).join(f"• {insight}" for insight in bottom_insights)}

CLUSTER ASSESSMENT:
{'This cluster shows exceptional strength and should be leveraged as a key competitive advantage.' if score >= 4.0 else 'This cluster demonstrates solid performance with opportunities for optimization.' if score >= 3.0 else 'This cluster requires focused attention and strategic improvements to reach market viability.'}""".strip()
            
            summaries[cluster] = summary
        
        return summaries
    
    def _generate_key_recommendations(self, evaluations: List[AgentEvaluation]) -> List[str]:
        """Generate key recommendations from agent insights"""
        recommendations = []
        
        # Collect recommendations from all agents
        for eval in evaluations:
            if eval.recommendations:
                recommendations.extend(eval.recommendations)
        
        # If no recommendations from agents, generate based on low-scoring areas
        if not recommendations:
            low_score_areas = sorted([e for e in evaluations if e.assigned_score <= 2.5], key=lambda x: x.assigned_score)
            for eval in low_score_areas[:5]:
                explanation_text = self._ensure_string_explanation(eval.explanation) if eval.explanation else "requires attention"
                explanation = explanation_text.split('.')[0] if explanation_text else "requires attention"
                recommendations.append(f"Improve {eval.sub_parameter}: {explanation}")
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        
        # Sort by frequency and score impact
        recommendation_scores = {}
        for rec in unique_recommendations:
            # Count how many agents mentioned this recommendation
            count = sum(1 for eval in evaluations if eval.recommendations and rec in eval.recommendations)
            # Weight by the agent's score (lower scores = more critical recommendations)
            avg_score = sum(eval.assigned_score for eval in evaluations if eval.recommendations and rec in eval.recommendations) / max(count, 1)
            recommendation_scores[rec] = count * (5.0 - avg_score)  # Higher weight for more critical items
        
        # Sort by priority and return top 10
        sorted_recommendations = sorted(unique_recommendations, key=lambda x: recommendation_scores.get(x, 0), reverse=True)
        return sorted_recommendations[:10]
    
    def _generate_critical_risks(self, evaluations: List[AgentEvaluation]) -> List[str]:
        """Generate critical risks identified by agents"""
        risks = []
        
        # Collect risk factors from all agents
        for eval in evaluations:
            if eval.risk_factors:
                risks.extend(eval.risk_factors)
        
        # If no risks from agents, generate based on low-scoring areas
        if not risks:
            low_score_areas = sorted([e for e in evaluations if e.assigned_score <= 2.0], key=lambda x: x.assigned_score)
            for eval in low_score_areas[:3]:
                explanation_text = self._ensure_string_explanation(eval.explanation) if eval.explanation else "shows concerning results"
                explanation = explanation_text.split('.')[0] if explanation_text else "shows concerning results"
                risks.append(f"Risk in {eval.sub_parameter}: {explanation}")
        
        # Remove duplicates and prioritize
        unique_risks = list(set(risks))
        
        # Sort by frequency and severity
        risk_scores = {}
        for risk in unique_risks:
            # Count how many agents mentioned this risk
            count = sum(1 for eval in evaluations if eval.risk_factors and risk in eval.risk_factors)
            # Weight by the agent's score (lower scores = more critical risks)
            avg_score = sum(eval.assigned_score for eval in evaluations if eval.risk_factors and risk in eval.risk_factors) / max(count, 1)
            risk_scores[risk] = count * (5.0 - avg_score)  # Higher weight for more critical items
        
        # Sort by priority and return top 10
        sorted_risks = sorted(unique_risks, key=lambda x: risk_scores.get(x, 0), reverse=True)
        return sorted_risks[:10]
    
    def _generate_market_insights(self, evaluations: List[AgentEvaluation]) -> List[str]:
        """Generate market insights from agent evaluations"""
        insights = []
        
        # Collect market insights from all agents
        for eval in evaluations:
            if hasattr(eval, 'indian_market_considerations') and eval.indian_market_considerations:
                if isinstance(eval.indian_market_considerations, list):
                    insights.extend(eval.indian_market_considerations)
                else:
                    insights.append(eval.indian_market_considerations)
        
        # Remove duplicates and prioritize
        unique_insights = list(set(insights))
        
        # Sort by frequency and return top 8
        insight_counts = {}
        for insight in unique_insights:
            count = sum(1 for eval in evaluations if hasattr(eval, 'indian_market_considerations') and insight in (eval.indian_market_considerations if isinstance(eval.indian_market_considerations, list) else [eval.indian_market_considerations]))
            insight_counts[insight] = count
        
        sorted_insights = sorted(unique_insights, key=lambda x: insight_counts.get(x, 0), reverse=True)
        return sorted_insights[:8]
