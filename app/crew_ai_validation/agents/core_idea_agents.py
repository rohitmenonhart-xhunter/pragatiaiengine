"""
Core Idea Validation Agents
Specialized agents for evaluating the fundamental strength and innovation potential of ideas.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
from ..base_agent import BaseValidationAgent


class OriginalityAgent(BaseValidationAgent):
    """Agent specialized in evaluating the originality and novelty of ideas"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="originality_specialist",
            cluster="Core Idea",
            parameter="Novelty & Uniqueness", 
            sub_parameter="Originality",
            weight=30,
            dependencies=[],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Innovation Originality Specialist",
            goal="Evaluate the true originality and breakthrough potential of startup ideas with precision and depth",
            backstory="""You are a world-renowned innovation expert with 15+ years of experience in 
            evaluating groundbreaking technologies and business concepts. You have worked with top-tier 
            VCs, patent offices, and R&D departments. Your expertise lies in identifying truly original 
            ideas versus incremental improvements. You understand the nuances of prior art, technology 
            curves, and what constitutes genuine innovation versus marketing hype.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "novelty_assessment": {
                "description": "Assess if the idea is genuinely new or a variation of existing solutions",
                "scoring_rubric": {
                    5: "Completely novel concept with no existing precedent",
                    4: "Significant innovation with some original elements",
                    3: "Moderate novelty with clear improvements over existing solutions",
                    2: "Minor variations of existing approaches",
                    1: "Common or well-established concept"
                }
            },
            "prior_art_analysis": {
                "description": "Evaluate against existing patents, research, and market solutions",
                "factors": ["Patent landscape", "Academic research", "Commercial products", "Open source projects"]
            },
            "innovation_index": {
                "description": "Measure the degree of innovation using established frameworks",
                "frameworks": ["Henderson-Clark model", "Disruptive innovation theory", "Technology S-curves"]
            },
            "market_disruption_potential": {
                "description": "Assess potential to create new markets or transform existing ones",
                "indicators": ["Blue ocean potential", "Market creation capability", "Industry transformation"]
            }
        }
    
    def get_specialized_tools(self) -> List:
        """Tools specific to originality assessment"""
        return []  # Will implement tools later if needed
    
    def get_collaboration_dependencies(self) -> Dict[str, str]:
        """Define how this agent collaborates with others"""
        return {
            "innovation_index_agent": "Shares innovation metrics and benchmarks",
            "differentiation_agent": "Provides originality context for competitive analysis",
            "market_gap_agent": "Informs about novel opportunities in the market"
        }


class DifferentiationAgent(BaseValidationAgent):
    """Agent specialized in evaluating competitive differentiation"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="differentiation_specialist",
            cluster="Core Idea",
            parameter="Novelty & Uniqueness",
            sub_parameter="Differentiation", 
            weight=25,
            dependencies=["Market Gap Analysis"],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Competitive Differentiation Expert",
            goal="Analyze and score how effectively the idea differentiates from existing competition",
            backstory="""You are a strategic business analyst with extensive experience in competitive 
            intelligence and market positioning. You have worked with Fortune 500 companies and startups 
            to identify unique value propositions and sustainable competitive advantages. Your expertise 
            includes Porter's Five Forces, Blue Ocean Strategy, and modern competitive analysis frameworks. 
            You excel at identifying subtle but crucial differentiators that can make or break a business.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "competitive_analysis": {
                "description": "Comprehensive analysis of direct and indirect competitors",
                "dimensions": ["Feature comparison", "Pricing analysis", "Market positioning", "Customer segments"]
            },
            "value_proposition_uniqueness": {
                "description": "Assess the uniqueness and strength of the value proposition",
                "scoring_factors": {
                    5: "Unique value prop with clear competitive moats",
                    4: "Strong differentiation with defendable advantages", 
                    3: "Moderate differentiation with some unique elements",
                    2: "Limited differentiation, mostly feature-based",
                    1: "No clear differentiation from existing solutions"
                }
            },
            "sustainable_advantages": {
                "description": "Evaluate if differentiation can be sustained over time",
                "advantage_types": ["Technology moats", "Network effects", "Brand differentiation", "Cost advantages"]
            },
            "market_positioning": {
                "description": "Assess positioning effectiveness in target market",
                "elements": ["Target audience clarity", "Messaging effectiveness", "Channel strategy", "Pricing position"]
            }
        }
    
    def get_collaboration_dependencies(self) -> Dict[str, str]:
        return {
            "originality_agent": "Uses originality insights for differentiation context",
            "market_gap_agent": "Leverages market gap analysis for positioning",
            "competitive_intensity_agent": "Collaborates on competitive landscape assessment"
        }


class InnovationIndexAgent(BaseValidationAgent):
    """Agent specialized in measuring innovation metrics and benchmarks"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="innovation_index_specialist",
            cluster="Core Idea",
            parameter="Novelty & Uniqueness",
            sub_parameter="Innovation Index",
            weight=25,
            dependencies=[],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Innovation Measurement Expert",
            goal="Quantify and benchmark the innovation level using established metrics and frameworks",
            backstory="""You are a research-oriented innovation expert with a PhD in Innovation Management 
            and 12+ years of experience in technology assessment. You have developed innovation metrics for 
            government agencies, research institutions, and Fortune 500 R&D departments. Your specialty is 
            translating qualitative innovation concepts into quantitative, comparable metrics using established 
            academic and industry frameworks.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "innovation_maturity_assessment": {
                "description": "Measure innovation using Technology Readiness Levels and innovation frameworks",
                "frameworks": ["TRL assessment", "Innovation funnel analysis", "Stage-gate evaluation"]
            },
            "creativity_index": {
                "description": "Assess the creative and imaginative aspects of the solution",
                "metrics": ["Ideation uniqueness", "Problem reframing", "Solution creativity", "Implementation imagination"]
            },
            "technology_advancement": {
                "description": "Evaluate technological advancement level",
                "dimensions": ["Technical complexity", "Scientific novelty", "Engineering innovation", "Integration sophistication"]
            },
            "innovation_benchmarking": {
                "description": "Compare against industry innovation standards",
                "benchmarks": ["Industry averages", "Best-in-class examples", "Academic research standards", "Patent quality metrics"]
            }
        }


class DisruptivePotentialAgent(BaseValidationAgent):
    """Agent specialized in evaluating disruptive innovation potential"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="disruptive_potential_specialist",
            cluster="Core Idea",
            parameter="Novelty & Uniqueness",
            sub_parameter="Disruptive Potential",
            weight=20,
            dependencies=["Technology Maturity"],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Disruptive Innovation Analyst",
            goal="Evaluate the potential for the idea to disrupt existing markets or create new ones",
            backstory="""You are a strategic innovation consultant specializing in Clayton Christensen's 
            disruptive innovation theory and its modern applications. You have 10+ years of experience 
            helping companies identify and respond to disruptive threats and opportunities. You understand 
            the patterns of disruption across industries and can identify early signals of transformative 
            potential. Your expertise includes market transformation analysis, business model innovation, 
            and technology adoption curves.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "disruption_patterns": {
                "description": "Analyze against established disruption patterns",
                "patterns": ["Low-end disruption", "New-market disruption", "Platform disruption", "Business model disruption"]
            },
            "market_transformation_potential": {
                "description": "Assess potential to transform existing markets",
                "factors": ["Industry structure change", "Value chain reconfiguration", "Customer behavior shift", "Ecosystem transformation"]
            },
            "adoption_trajectory": {
                "description": "Evaluate likely adoption pattern and speed",
                "models": ["Technology adoption lifecycle", "Diffusion of innovations", "Network effects acceleration"]
            },
            "incumbent_vulnerability": {
                "description": "Assess how vulnerable current market leaders are to this innovation",
                "vulnerabilities": ["Business model constraints", "Technology limitations", "Customer segments neglected", "Cost structure disadvantages"]
            }
        }


class ProblemSeverityAgent(BaseValidationAgent):
    """Agent specialized in evaluating the severity and prevalence of problems being addressed"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="problem_severity_specialist",
            cluster="Core Idea",
            parameter="Problem-Solution Fit",
            sub_parameter="Problem Severity",
            weight=25,
            dependencies=[],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Problem Validation Expert",
            goal="Assess the severity, prevalence, and urgency of the problem being addressed",
            backstory="""You are a market research expert with deep experience in problem validation 
            and customer development. You have spent 12+ years helping startups and enterprises identify 
            and validate real customer problems. Your expertise includes ethnographic research, jobs-to-be-done 
            framework, and quantitative problem assessment. You excel at distinguishing between perceived 
            problems and actual market pain points that customers will pay to solve.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "problem_prevalence": {
                "description": "Assess how widespread the problem is among target audience",
                "metrics": ["Market size affected", "Frequency of occurrence", "Geographic distribution", "Demographic spread"]
            },
            "pain_intensity": {
                "description": "Measure the intensity of pain felt by those experiencing the problem",
                "indicators": ["Customer complaints", "Workaround efforts", "Time/money spent on alternatives", "Frustration levels"]
            },
            "urgency_assessment": {
                "description": "Evaluate how urgently the problem needs to be solved",
                "factors": ["Time sensitivity", "Consequence severity", "Regulatory pressure", "Competitive pressure"]
            },
            "economic_impact": {
                "description": "Quantify the economic impact of the problem",
                "dimensions": ["Revenue loss", "Cost increase", "Opportunity cost", "Productivity impact"]
            }
        }


# Additional Core Idea agents would follow the same pattern...
# For brevity, I'll create a factory function to generate all agents

class CoreIdeaAgentFactory:
    """Factory for creating all Core Idea validation agents"""
    
    @staticmethod
    def create_all_agents(llm: ChatOpenAI) -> Dict[str, BaseValidationAgent]:
        """Create all Core Idea validation agents"""
        agents = {
            "originality": OriginalityAgent(llm),
            "differentiation": DifferentiationAgent(llm),
            "innovation_index": InnovationIndexAgent(llm),
            "disruptive_potential": DisruptivePotentialAgent(llm),
            "problem_severity": ProblemSeverityAgent(llm),
            # Add more agents here following the same pattern
        }
        
        # Add remaining agents with simplified initialization
        remaining_agents = {
            "solution_effectiveness": {
                "sub_parameter": "Solution Effectiveness",
                "weight": 25,
                "dependencies": ["Technical Feasibility"]
            },
            "market_gap_analysis": {
                "sub_parameter": "Market Gap Analysis", 
                "weight": 20,
                "dependencies": ["Market Size (TAM)"]
            },
            "customer_pain_validation": {
                "sub_parameter": "Customer Pain Validation",
                "weight": 15,
                "dependencies": ["User Engagement"]
            },
            "solution_uniqueness": {
                "sub_parameter": "Solution Uniqueness",
                "weight": 15,
                "dependencies": ["Originality"]
            },
            "intuitive_design": {
                "sub_parameter": "Intuitive Design",
                "weight": 30,
                "dependencies": []
            },
            "accessibility_compliance": {
                "sub_parameter": "Accessibility Compliance",
                "weight": 25,
                "dependencies": ["Regulatory Landscape"]
            },
            "user_interface_quality": {
                "sub_parameter": "User Interface Quality",
                "weight": 20,
                "dependencies": ["Intuitive Design"]
            },
            "mobile_responsiveness": {
                "sub_parameter": "Mobile Responsiveness",
                "weight": 15,
                "dependencies": []
            },
            "cross_platform_compatibility": {
                "sub_parameter": "Cross-Platform Compatibility",
                "weight": 10,
                "dependencies": ["Technical Architecture"]
            }
        }
        
        for agent_key, config in remaining_agents.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Core Idea",
                parameter="Problem-Solution Fit" if "solution" in agent_key or "gap" in agent_key or "pain" in agent_key else "UX/Usability Potential",
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=llm
            )
        
        return agents
