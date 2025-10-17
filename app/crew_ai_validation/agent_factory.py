"""
Complete Agent Factory for CrewAI Validation System
Creates all 109+ specialized validation agents across all clusters.
"""

from typing import Dict, List
from langchain_openai import ChatOpenAI
from .base_agent import BaseValidationAgent
from .agents.core_idea_agents import CoreIdeaAgentFactory
from .agents.market_opportunity_agents import MarketOpportunityAgentFactory


class ComprehensiveAgentFactory:
    """Factory for creating all validation agents across all clusters"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.all_agents = {}
    
    def create_all_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all 109+ validation agents"""
        
        # Core Idea Cluster (14 agents)
        core_idea_agents = CoreIdeaAgentFactory.create_all_agents(self.llm)
        self.all_agents.update(core_idea_agents)
        
        # Market Opportunity Cluster (17 agents)
        market_agents = MarketOpportunityAgentFactory.create_all_agents(self.llm)
        self.all_agents.update(market_agents)
        
        # Execution Cluster (18 agents)
        execution_agents = self._create_execution_agents()
        self.all_agents.update(execution_agents)
        
        # Business Model Cluster (12 agents)
        business_model_agents = self._create_business_model_agents()
        self.all_agents.update(business_model_agents)
        
        # Team Cluster (12 agents)
        team_agents = self._create_team_agents()
        self.all_agents.update(team_agents)
        
        # Compliance Cluster (18 agents)
        compliance_agents = self._create_compliance_agents()
        self.all_agents.update(compliance_agents)
        
        # Risk & Strategy Cluster (18 agents)
        risk_strategy_agents = self._create_risk_strategy_agents()
        self.all_agents.update(risk_strategy_agents)
        
        print(f"Created {len(self.all_agents)} specialized validation agents")
        return self.all_agents
    
    def _create_execution_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all Execution cluster agents"""
        agents = {}
        
        execution_config = {
            # Technical Feasibility (6 agents)
            "technology_maturity": {
                "parameter": "Technical Feasibility",
                "sub_parameter": "Technology Maturity",
                "weight": 20,
                "dependencies": []
            },
            "scalability_performance": {
                "parameter": "Technical Feasibility", 
                "sub_parameter": "Scalability & Performance",
                "weight": 20,
                "dependencies": ["Technology Maturity"]
            },
            "technical_architecture": {
                "parameter": "Technical Feasibility",
                "sub_parameter": "Technical Architecture", 
                "weight": 15,
                "dependencies": ["Technology Maturity"]
            },
            "development_complexity": {
                "parameter": "Technical Feasibility",
                "sub_parameter": "Development Complexity",
                "weight": 15,
                "dependencies": ["Technical Architecture"]
            },
            "security_framework": {
                "parameter": "Technical Feasibility",
                "sub_parameter": "Security Framework",
                "weight": 15,
                "dependencies": ["Data Privacy Compliance"]
            },
            "api_integration_capability": {
                "parameter": "Technical Feasibility", 
                "sub_parameter": "API Integration Capability",
                "weight": 15,
                "dependencies": ["Technical Architecture"]
            },
            
            # Operational Viability (6 agents)
            "resource_availability": {
                "parameter": "Operational Viability",
                "sub_parameter": "Resource Availability",
                "weight": 20,
                "dependencies": []
            },
            "process_efficiency": {
                "parameter": "Operational Viability",
                "sub_parameter": "Process Efficiency", 
                "weight": 20,
                "dependencies": ["Resource Availability"]
            },
            "supply_chain_management": {
                "parameter": "Operational Viability",
                "sub_parameter": "Supply Chain Management",
                "weight": 15,
                "dependencies": ["Process Efficiency"]
            },
            "quality_assurance": {
                "parameter": "Operational Viability",
                "sub_parameter": "Quality Assurance",
                "weight": 15,
                "dependencies": ["Process Efficiency"]
            },
            "operational_scalability": {
                "parameter": "Operational Viability", 
                "sub_parameter": "Operational Scalability",
                "weight": 15,
                "dependencies": ["Process Efficiency"]
            },
            "cost_structure_optimization": {
                "parameter": "Operational Viability",
                "sub_parameter": "Cost Structure Optimization",
                "weight": 15,
                "dependencies": ["Unit Economics"]
            },
            
            # Scalability Potential (6 agents) 
            "business_model_scalability": {
                "parameter": "Scalability Potential",
                "sub_parameter": "Business Model Scalability",
                "weight": 20,
                "dependencies": ["Financial Viability"]
            },
            "market_expansion_potential": {
                "parameter": "Scalability Potential",
                "sub_parameter": "Market Expansion Potential",
                "weight": 20,
                "dependencies": ["Market Size (TAM)"]
            },
            "technology_scalability": {
                "parameter": "Scalability Potential",
                "sub_parameter": "Technology Scalability",
                "weight": 15,
                "dependencies": ["Scalability & Performance"]
            },
            "execution_operational_scalability": {
                "parameter": "Scalability Potential",
                "sub_parameter": "Operational Scalability",
                "weight": 15,
                "dependencies": ["Process Efficiency"]
            },
            "financial_scalability": {
                "parameter": "Scalability Potential",
                "sub_parameter": "Financial Scalability", 
                "weight": 15,
                "dependencies": ["Revenue Stream Diversity"]
            },
            "international_expansion": {
                "parameter": "Scalability Potential",
                "sub_parameter": "International Expansion",
                "weight": 15,
                "dependencies": ["Cultural Adaptation"]
            }
        }
        
        for agent_key, config in execution_config.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Execution",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=self.llm
            )
        
        return agents
    
    def _create_business_model_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all Business Model cluster agents"""
        agents = {}
        
        business_model_config = {
            # Financial Viability (6 agents)
            "revenue_stream_diversity": {
                "parameter": "Financial Viability",
                "sub_parameter": "Revenue Stream Diversity",
                "weight": 20,
                "dependencies": []
            },
            "profitability_margins": {
                "parameter": "Financial Viability",
                "sub_parameter": "Profitability & Margins",
                "weight": 20,
                "dependencies": ["Unit Economics"]
            },
            "cash_flow_sustainability": {
                "parameter": "Financial Viability",
                "sub_parameter": "Cash Flow Sustainability",
                "weight": 15,
                "dependencies": ["Revenue Stream Diversity"]
            },
            "customer_lifetime_value": {
                "parameter": "Financial Viability",
                "sub_parameter": "Customer Lifetime Value",
                "weight": 15,
                "dependencies": ["Retention Potential"]
            },
            "unit_economics": {
                "parameter": "Financial Viability", 
                "sub_parameter": "Unit Economics",
                "weight": 15,
                "dependencies": ["Revenue Stream Diversity"]
            },
            "financial_projections_accuracy": {
                "parameter": "Financial Viability",
                "sub_parameter": "Financial Projections Accuracy",
                "weight": 15,
                "dependencies": ["Market Size (TAM)"]
            },
            
            # Defensibility (6 agents)
            "intellectual_property": {
                "parameter": "Defensibility",
                "sub_parameter": "Intellectual Property (IP)",
                "weight": 20,
                "dependencies": ["Originality"]
            },
            "network_effects": {
                "parameter": "Defensibility",
                "sub_parameter": "Network Effects",
                "weight": 20,
                "dependencies": ["User Engagement"]
            },
            "brand_moat": {
                "parameter": "Defensibility",
                "sub_parameter": "Brand Moat",
                "weight": 15,
                "dependencies": ["Differentiation"]
            },
            "data_moat": {
                "parameter": "Defensibility",
                "sub_parameter": "Data Moat",
                "weight": 15,
                "dependencies": ["User Engagement"]
            },
            "switching_costs": {
                "parameter": "Defensibility",
                "sub_parameter": "Switching Costs",
                "weight": 15,
                "dependencies": ["Product Stickiness"]
            },
            "regulatory_barriers": {
                "parameter": "Defensibility",
                "sub_parameter": "Regulatory Barriers",
                "weight": 15,
                "dependencies": ["Regulatory Landscape"]
            }
        }
        
        for agent_key, config in business_model_config.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Business Model",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=self.llm
            )
        
        return agents
    
    def _create_team_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all Team cluster agents"""
        agents = {}
        
        team_config = {
            # Founder-Fit (6 agents)
            "relevant_experience": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Relevant Experience",
                "weight": 20,
                "dependencies": []
            },
            "complementary_skills": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Complementary Skills",
                "weight": 20,
                "dependencies": ["Relevant Experience"]
            },
            "industry_expertise": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Industry Expertise",
                "weight": 15,
                "dependencies": ["Relevant Experience"]
            },
            "leadership_capability": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Leadership Capability",
                "weight": 15,
                "dependencies": []
            },
            "execution_track_record": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Execution Track Record",
                "weight": 15,
                "dependencies": ["Leadership Capability"]
            },
            "domain_knowledge": {
                "parameter": "Founder-Fit",
                "sub_parameter": "Domain Knowledge",
                "weight": 15,
                "dependencies": ["Industry Expertise"]
            },
            
            # Culture/Values (6 agents)
            "mission_alignment": {
                "parameter": "Culture/Values",
                "sub_parameter": "Mission Alignment",
                "weight": 20,
                "dependencies": []
            },
            "diversity_inclusion": {
                "parameter": "Culture/Values",
                "sub_parameter": "Diversity & Inclusion",
                "weight": 20,
                "dependencies": []
            },
            "team_dynamics": {
                "parameter": "Culture/Values",
                "sub_parameter": "Team Dynamics",
                "weight": 15,
                "dependencies": ["Communication Effectiveness"]
            },
            "communication_effectiveness": {
                "parameter": "Culture/Values",
                "sub_parameter": "Communication Effectiveness",
                "weight": 15,
                "dependencies": []
            },
            "adaptability": {
                "parameter": "Culture/Values",
                "sub_parameter": "Adaptability",
                "weight": 15,
                "dependencies": ["Team Dynamics"]
            },
            "work_ethics_values": {
                "parameter": "Culture/Values",
                "sub_parameter": "Work Ethics & Values",
                "weight": 15,
                "dependencies": ["Mission Alignment"]
            }
        }
        
        for agent_key, config in team_config.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Team",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=self.llm
            )
        
        return agents
    
    def _create_compliance_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all Compliance cluster agents"""
        agents = {}
        
        compliance_config = {
            # Regulatory (India) (6 agents)
            "data_privacy_compliance": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Data Privacy Compliance",
                "weight": 20,
                "dependencies": []
            },
            "sector_specific_compliance": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Sector-Specific Compliance",
                "weight": 20,
                "dependencies": ["Regulatory Landscape"]
            },
            "tax_compliance": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Tax Compliance",
                "weight": 15,
                "dependencies": []
            },
            "labor_law_compliance": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Labor Law Compliance",
                "weight": 15,
                "dependencies": []
            },
            "import_export_regulations": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Import/Export Regulations",
                "weight": 15,
                "dependencies": ["Regulatory Landscape"]
            },
            "digital_india_compliance": {
                "parameter": "Regulatory (India)",
                "sub_parameter": "Digital India Compliance",
                "weight": 15,
                "dependencies": ["Infrastructure Readiness"]
            },
            
            # Sustainability (ESG) (6 agents)
            "environmental_impact": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Environmental Impact",
                "weight": 20,
                "dependencies": []
            },
            "social_impact_sdgs": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Social Impact (SDGs)",
                "weight": 20,
                "dependencies": []
            },
            "governance_standards": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Governance Standards",
                "weight": 15,
                "dependencies": ["Ethical Business Practices"]
            },
            "ethical_business_practices": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Ethical Business Practices",
                "weight": 15,
                "dependencies": []
            },
            "community_engagement": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Community Engagement",
                "weight": 15,
                "dependencies": ["Social Impact (SDGs)"]
            },
            "carbon_footprint": {
                "parameter": "Sustainability (ESG)",
                "sub_parameter": "Carbon Footprint",
                "weight": 15,
                "dependencies": ["Environmental Impact"]
            },
            
            # Ecosystem Support (India) (6 agents)
            "government_institutional_support": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Government & Institutional Support",
                "weight": 20,
                "dependencies": ["National Policy Alignment (India)"]
            },
            "investor_partner_landscape": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Investor & Partner Landscape",
                "weight": 20,
                "dependencies": []
            },
            "startup_ecosystem_integration": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Startup Ecosystem Integration",
                "weight": 15,
                "dependencies": ["Investor & Partner Landscape"]
            },
            "mentorship_availability": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Mentorship Availability",
                "weight": 15,
                "dependencies": ["Academic Partnerships"]
            },
            "industry_associations": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Industry Associations",
                "weight": 15,
                "dependencies": []
            },
            "academic_partnerships": {
                "parameter": "Ecosystem Support (India)",
                "sub_parameter": "Academic Partnerships",
                "weight": 15,
                "dependencies": ["Academic/Research Contribution"]
            }
        }
        
        for agent_key, config in compliance_config.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Compliance",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=self.llm
            )
        
        return agents
    
    def _create_risk_strategy_agents(self) -> Dict[str, BaseValidationAgent]:
        """Create all Risk & Strategy cluster agents"""
        agents = {}
        
        risk_strategy_config = {
            # Risk Assessment (6 agents)
            "technical_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Technical Risks",
                "weight": 20,
                "dependencies": ["Development Complexity"]
            },
            "market_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Market Risks",
                "weight": 20,
                "dependencies": ["Competitive Intensity"]
            },
            "financial_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Financial Risks",
                "weight": 15,
                "dependencies": ["Cash Flow Sustainability"]
            },
            "competitive_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Competitive Risks",
                "weight": 15,
                "dependencies": ["Competitive Intensity"]
            },
            "regulatory_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Regulatory Risks",
                "weight": 15,
                "dependencies": ["Regulatory Landscape"]
            },
            "operational_risks": {
                "parameter": "Risk Assessment",
                "sub_parameter": "Operational Risks",
                "weight": 15,
                "dependencies": ["Operational Viability"]
            },
            
            # Investor Attractiveness (6 agents)
            "valuation_potential": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "Valuation Potential",
                "weight": 20,
                "dependencies": ["Market Size (TAM)"]
            },
            "exit_strategy_viability": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "Exit Strategy Viability",
                "weight": 20,
                "dependencies": ["Market Expansion Potential"]
            },
            "roi_potential": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "ROI Potential",
                "weight": 15,
                "dependencies": ["Profitability & Margins"]
            },
            "investment_stage_readiness": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "Investment Stage Readiness",
                "weight": 15,
                "dependencies": ["Financial Projections Accuracy"]
            },
            "due_diligence_preparedness": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "Due Diligence Preparedness",
                "weight": 15,
                "dependencies": ["Governance Standards"]
            },
            "investor_fit": {
                "parameter": "Investor Attractiveness",
                "sub_parameter": "Investor Fit",
                "weight": 15,
                "dependencies": ["Investor & Partner Landscape"]
            },
            
            # Academic/National Alignment (6 agents)
            "national_policy_alignment": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "National Policy Alignment (India)",
                "weight": 20,
                "dependencies": []
            },
            "academic_research_contribution": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "Academic/Research Contribution",
                "weight": 20,
                "dependencies": []
            },
            "innovation_ecosystem_impact": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "Innovation Ecosystem Impact",
                "weight": 15,
                "dependencies": ["Academic/Research Contribution"]
            },
            "knowledge_transfer_potential": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "Knowledge Transfer Potential",
                "weight": 15,
                "dependencies": ["Academic/Research Contribution"]
            },
            "research_commercialization": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "Research Commercialization",
                "weight": 15,
                "dependencies": ["Knowledge Transfer Potential"]
            },
            "educational_value": {
                "parameter": "Academic/National Alignment",
                "sub_parameter": "Educational Value",
                "weight": 15,
                "dependencies": ["Academic/Research Contribution"]
            }
        }
        
        for agent_key, config in risk_strategy_config.items():
            agents[agent_key] = BaseValidationAgent(
                agent_id=f"{agent_key}_specialist",
                cluster="Risk & Strategy",
                parameter=config["parameter"],
                sub_parameter=config["sub_parameter"],
                weight=config["weight"],
                dependencies=config["dependencies"],
                llm=self.llm
            )
        
        return agents
    
    def get_agent_count_by_cluster(self) -> Dict[str, int]:
        """Get agent count by cluster"""
        cluster_counts = {}
        for agent in self.all_agents.values():
            cluster = agent.cluster
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        return cluster_counts
    
    def get_total_agent_count(self) -> int:
        """Get total number of agents"""
        return len(self.all_agents)
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency relationships between agents"""
        dependency_graph = {}
        for agent_id, agent in self.all_agents.items():
            dependency_graph[agent_id] = agent.dependencies
        return dependency_graph
