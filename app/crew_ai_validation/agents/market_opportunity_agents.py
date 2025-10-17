"""
Market Opportunity Validation Agents
Specialized agents for evaluating market potential, validation, and opportunity assessment.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
from ..base_agent import BaseValidationAgent


class MarketSizeAgent(BaseValidationAgent):
    """Agent specialized in Total Addressable Market (TAM) analysis"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="market_size_specialist",
            cluster="Market Opportunity",
            parameter="Market Validation",
            sub_parameter="Market Size (TAM)",
            weight=25,
            dependencies=[],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Market Sizing Expert",
            goal="Calculate and validate the Total Addressable Market (TAM) for startup ideas with precision",
            backstory="""You are a market research expert with 12+ years of experience in market sizing 
            and validation. You have worked with top-tier consulting firms, VCs, and market research 
            companies to accurately size markets across various industries. Your expertise includes 
            both top-down and bottom-up market sizing methodologies, with special focus on emerging 
            markets like India. You understand the nuances of Indian market segmentation, purchasing 
            power, and adoption patterns.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "tam_calculation": {
                "description": "Calculate Total Addressable Market using multiple methodologies",
                "methodologies": ["Top-down analysis", "Bottom-up analysis", "Value theory approach"],
                "scoring_rubric": {
                    5: "TAM > ₹10,000 Cr with strong validation",
                    4: "TAM ₹1,000-10,000 Cr with good validation",
                    3: "TAM ₹100-1,000 Cr with moderate validation",
                    2: "TAM ₹10-100 Cr with limited validation",
                    1: "TAM < ₹10 Cr or unclear market size"
                }
            },
            "market_validation": {
                "description": "Validate market size assumptions with real data",
                "data_sources": ["Industry reports", "Government statistics", "Competitor analysis", "Customer surveys"]
            },
            "addressable_segments": {
                "description": "Identify and size addressable customer segments",
                "factors": ["Geographic reach", "Demographic targeting", "Psychographic segments", "Behavioral patterns"]
            },
            "market_growth_potential": {
                "description": "Assess market growth trajectory and expansion potential",
                "indicators": ["Historical growth rates", "Market catalysts", "Technology adoption curves", "Regulatory changes"]
            }
        }
    
    def get_collaboration_dependencies(self) -> Dict[str, str]:
        return {
            "competitive_intensity_agent": "Provides market competition context for sizing",
            "market_growth_agent": "Collaborates on growth rate projections",
            "geographic_specificity_agent": "Informs regional market sizing"
        }


class CompetitiveIntensityAgent(BaseValidationAgent):
    """Agent specialized in competitive landscape analysis"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="competitive_intensity_specialist",
            cluster="Market Opportunity",
            parameter="Market Validation",
            sub_parameter="Competitive Intensity",
            weight=20,
            dependencies=["Market Size (TAM)"],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Competitive Intelligence Expert",
            goal="Analyze competitive landscape intensity and market dynamics",
            backstory="""You are a competitive intelligence specialist with 10+ years of experience 
            analyzing market competition and industry dynamics. You have worked with strategy consulting 
            firms and corporate strategy teams to assess competitive threats and opportunities. Your 
            expertise includes Porter's Five Forces analysis, competitive positioning, and market 
            concentration analysis, with deep knowledge of Indian competitive landscapes across sectors.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "porter_five_forces": {
                "description": "Comprehensive competitive analysis using Porter's framework",
                "forces": ["Competitive rivalry", "Supplier power", "Buyer power", "Threat of substitutes", "Barriers to entry"]
            },
            "market_concentration": {
                "description": "Analyze market concentration and fragmentation",
                "metrics": ["Market share distribution", "HHI index", "Number of players", "Competitive dynamics"],
                "scoring_rubric": {
                    5: "Fragmented market with clear opportunities",
                    4: "Moderate competition with differentiation potential",
                    3: "Balanced competitive landscape",
                    2: "High competition with limited differentiation",
                    1: "Dominated market with high barriers"
                }
            },
            "competitive_positioning": {
                "description": "Assess positioning opportunities relative to competitors",
                "dimensions": ["Price positioning", "Feature differentiation", "Customer segments", "Geographic coverage"]
            },
            "competitive_threats": {
                "description": "Identify and assess potential competitive threats",
                "threat_types": ["Direct competitors", "Indirect competitors", "New entrants", "Substitute products"]
            }
        }


class MarketGrowthRateAgent(BaseValidationAgent):
    """Agent specialized in market growth analysis"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="market_growth_specialist",
            cluster="Market Opportunity",
            parameter="Market Validation",
            sub_parameter="Market Growth Rate",
            weight=20,
            dependencies=["Market Size (TAM)"],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Market Growth Analyst",
            goal="Evaluate market growth rates, trends, and future projections",
            backstory="""You are a market analyst with expertise in growth trend analysis and market 
            forecasting. You have 8+ years of experience with research firms and investment banks, 
            specializing in growth market analysis and trend identification. Your expertise includes 
            time-series analysis, market modeling, and growth driver identification, with particular 
            focus on high-growth emerging markets like India.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "historical_growth_analysis": {
                "description": "Analyze historical market growth patterns",
                "metrics": ["CAGR analysis", "Growth volatility", "Cyclical patterns", "Seasonal trends"]
            },
            "growth_drivers": {
                "description": "Identify and validate key market growth drivers",
                "driver_types": ["Technology adoption", "Regulatory changes", "Economic factors", "Demographic shifts"],
                "scoring_rubric": {
                    5: "Strong growth drivers with >25% CAGR potential",
                    4: "Good growth drivers with 15-25% CAGR potential",
                    3: "Moderate growth drivers with 8-15% CAGR potential",
                    2: "Limited growth drivers with 3-8% CAGR potential",
                    1: "Weak or declining market with <3% CAGR"
                }
            },
            "growth_sustainability": {
                "description": "Assess sustainability of projected growth rates",
                "factors": ["Market maturity", "Competition impact", "Technology lifecycle", "Regulatory stability"]
            },
            "market_catalysts": {
                "description": "Identify potential catalysts for accelerated growth",
                "catalysts": ["Policy changes", "Technology breakthroughs", "Infrastructure development", "Market education"]
            }
        }


class RegulatoryLandscapeAgent(BaseValidationAgent):
    """Agent specialized in Indian regulatory environment analysis"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            agent_id="regulatory_landscape_specialist",
            cluster="Market Opportunity",
            parameter="Geographic Specificity (India)",
            sub_parameter="Regulatory Landscape",
            weight=25,
            dependencies=[],
            llm=llm
        )
    
    def create_agent(self) -> Agent:
        return Agent(
            role="Indian Regulatory Expert",
            goal="Assess regulatory landscape, compliance requirements, and policy impact for Indian startups",
            backstory="""You are a regulatory affairs expert with 15+ years of experience in Indian 
            business regulations and policy analysis. You have worked with law firms, regulatory 
            consultancies, and government bodies to help businesses navigate India's complex regulatory 
            environment. Your expertise includes sector-specific regulations, compliance requirements, 
            policy trends, and regulatory risk assessment across various Indian industries.""",
            verbose=False,
            allow_delegation=True,
            llm=self.llm,
            tools=self.get_specialized_tools()
        )
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        return {
            "compliance_requirements": {
                "description": "Assess regulatory compliance requirements and complexity",
                "areas": ["Business licensing", "Data protection", "Industry-specific regulations", "Tax compliance"],
                "scoring_rubric": {
                    5: "Clear regulatory path with minimal compliance burden",
                    4: "Well-defined regulations with manageable compliance",
                    3: "Standard regulatory requirements with moderate complexity",
                    2: "Complex regulations with significant compliance burden",
                    1: "Unclear or restrictive regulatory environment"
                }
            },
            "policy_support": {
                "description": "Evaluate government policy support for the business model",
                "policies": ["Digital India", "Startup India", "Make in India", "Industry-specific policies"]
            },
            "regulatory_risks": {
                "description": "Identify potential regulatory risks and changes",
                "risk_types": ["Policy changes", "New regulations", "Enforcement variations", "Court decisions"]
            },
            "regulatory_advantages": {
                "description": "Identify regulatory advantages or protections",
                "advantages": ["Government incentives", "Protected sectors", "Preferential policies", "Regulatory sandboxes"]
            }
        }


# Factory for all Market Opportunity agents
class MarketOpportunityAgentFactory:
    """Factory for creating all Market Opportunity validation agents"""
    
    @staticmethod
    def create_all_agents(llm: ChatOpenAI) -> Dict[str, BaseValidationAgent]:
        """Create all Market Opportunity validation agents"""
        agents = {
            "market_size": MarketSizeAgent(llm),
            "competitive_intensity": CompetitiveIntensityAgent(llm),
            "market_growth_rate": MarketGrowthRateAgent(llm),
            "regulatory_landscape": RegulatoryLandscapeAgent(llm),
        }
        
        # Add remaining agents with simplified initialization
        remaining_agents = {
            "customer_acquisition_potential": {
                "parameter": "Market Validation",
                "sub_parameter": "Customer Acquisition Potential",
                "weight": 15,
                "dependencies": ["User Engagement"]
            },
            "market_penetration_strategy": {
                "parameter": "Market Validation",
                "sub_parameter": "Market Penetration Strategy",
                "weight": 10,
                "dependencies": ["Cultural Adaptation"]
            },
            "timing_market_readiness": {
                "parameter": "Market Validation",
                "sub_parameter": "Timing & Market Readiness",
                "weight": 10,
                "dependencies": ["Infrastructure Readiness"]
            },
            "infrastructure_readiness": {
                "parameter": "Geographic Specificity (India)",
                "sub_parameter": "Infrastructure Readiness",
                "weight": 25,
                "dependencies": []
            },
            "local_market_understanding": {
                "parameter": "Geographic Specificity (India)",
                "sub_parameter": "Local Market Understanding",
                "weight": 20,
                "dependencies": ["Cultural Adaptation"]
            },
            "cultural_adaptation": {
                "parameter": "Geographic Specificity (India)",
                "sub_parameter": "Cultural Adaptation",
                "weight": 15,
                "dependencies": []
            },
            "regional_expansion_potential": {
                "parameter": "Geographic Specificity (India)",
                "sub_parameter": "Regional Expansion Potential",
                "weight": 15,
                "dependencies": ["Infrastructure Readiness"]
            },
            "user_engagement": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "User Engagement",
                "weight": 20,
                "dependencies": ["Intuitive Design"]
            },
            "retention_potential": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "Retention Potential",
                "weight": 20,
                "dependencies": ["Product Stickiness"]
            },
            "customer_satisfaction_metrics": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "Customer Satisfaction Metrics",
                "weight": 20,
                "dependencies": ["Solution Effectiveness"]
            },
            "product_stickiness": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "Product Stickiness",
                "weight": 15,
                "dependencies": ["Network Effects"]
            },
            "market_feedback_integration": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "Market Feedback Integration",
                "weight": 15,
                "dependencies": ["Process Efficiency"]
            },
            "viral_coefficient": {
                "parameter": "Product-Market Fit",
                "sub_parameter": "Viral Coefficient",
                "weight": 10,
                "dependencies": ["Network Effects"]
            }
        }
        
        for agent_key, config in remaining_agents.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Market Opportunity",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=llm
            )
        
        return agents
