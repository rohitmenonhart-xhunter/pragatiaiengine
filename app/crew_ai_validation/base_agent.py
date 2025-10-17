"""
Base Agent Class for CrewAI Validation System
Provides common functionality for all validation agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from crewai import Agent
from langchain_openai import ChatOpenAI
from datetime import datetime
import json


class BaseValidationAgent(ABC):
    """
    Base class for all validation agents in the CrewAI system.
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, agent_id: str, cluster: str, parameter: str, 
                 sub_parameter: str, weight: float, dependencies: List[str], llm: ChatOpenAI):
        self.agent_id = agent_id
        self.cluster = cluster
        self.parameter = parameter
        self.sub_parameter = sub_parameter
        self.weight = weight
        self.dependencies = dependencies
        self.llm = llm
        self._agent = None
        self._last_evaluation = None
    
    def create_agent(self) -> Agent:
        """Create the CrewAI agent instance. Override in subclasses for customization."""
        if self._agent is None:
            self._agent = Agent(
                role=f"{self.sub_parameter} Validation Specialist",
                goal=self.get_default_goal(),
                backstory=self.get_default_backstory(),
                verbose=True,
                allow_delegation=True,
                llm=self.llm,
                tools=self.get_specialized_tools(),
                max_iter=3,  # Allow multiple iterations for argumentation
                memory=True  # Enable memory for cross-agent discussions
            )
        return self._agent
    
    def get_default_goal(self) -> str:
        """Default goal for the agent"""
        return f"""Provide expert evaluation of {self.sub_parameter} for startup ideas with precise 
        scoring (1.0-5.0), detailed analysis, and actionable insights within the {self.cluster} 
        evaluation framework. Be CRITICAL and RIGOROUS in your assessment. Challenge assumptions, 
        identify weaknesses, and don't hesitate to give low scores for poor ideas. Question other 
        agents' evaluations when they seem too lenient. Focus on Indian market context and provide 
        specific, measurable assessments with HONEST scoring."""
    
    def get_default_backstory(self) -> str:
        """Default backstory for the agent"""
        backstories = {
            "Core Idea": """You are a seasoned innovation consultant with deep expertise in evaluating 
            breakthrough ideas and disruptive technologies. You have helped assess hundreds of startups 
            and understand what makes ideas truly innovative in the Indian market context. You've seen 
            too many overhyped ideas fail and are known for your RIGOROUS, CRITICAL analysis. You don't 
            give participation trophies - you tell the hard truth about idea viability.""",
            
            "Market Opportunity": """You are a market research expert with extensive experience in the 
            Indian startup ecosystem. You understand market dynamics, customer behavior, and growth 
            potential in emerging markets, with specific expertise in Indian consumer and business segments.""",
            
            "Execution": """You are a technical and operational expert who has guided numerous startups 
            through execution challenges. You understand the complexities of building and scaling 
            technology solutions in the Indian infrastructure and talent landscape.""",
            
            "Business Model": """You are a business strategy expert with deep knowledge of sustainable 
            business models and financial viability. You have experience with venture capital and startup 
            valuations in the Indian market, understanding local business dynamics and investor preferences.""",
            
            "Team": """You are an organizational development expert who understands what makes 
            high-performing teams. You have experience in founder coaching and team building for 
            startups, with deep knowledge of Indian workplace culture and talent dynamics.""",
            
            "Compliance": """You are a regulatory and compliance expert with specialized knowledge of 
            Indian business environment, ESG principles, and ecosystem dynamics. You understand the 
            complex regulatory landscape and sustainability requirements for Indian startups.""",
            
            "Risk & Strategy": """You are a strategic risk assessment expert who helps startups navigate 
            uncertainties and position themselves for investment and growth opportunities. You have 
            deep experience with Indian market risks and strategic positioning."""
        }
        
        base_backstory = backstories.get(self.cluster, 
            "You are a specialized validation expert with deep domain knowledge.")
        
        return f"""{base_backstory} Your specific expertise lies in {self.sub_parameter} evaluation, 
        and you collaborate with other specialists to provide comprehensive assessments. You are known 
        for challenging other experts when their assessments seem too optimistic or lack sufficient 
        evidence. You believe in rigorous evaluation and aren't afraid to disagree with colleagues 
        when the data doesn't support their conclusions."""
    
    def get_specialized_tools(self) -> List:
        """Get specialized tools for this agent. Override in subclasses."""
        return []
    
    def _determine_industry_context(self, idea_name: str, idea_concept: str) -> str:
        """Determine the industry context from idea name and concept"""
        text = f"{idea_name} {idea_concept}".lower()
        
        if any(word in text for word in ['food', 'delivery', 'restaurant', 'meal', 'cooking', 'nutrition', 'lunch', 'breakfast', 'dinner', 'snacks']):
            return "Food & Delivery"
        elif any(word in text for word in ['health', 'medical', 'healthcare', 'doctor', 'patient', 'medicine', 'hospital']):
            return "Healthcare"
        elif any(word in text for word in ['education', 'learning', 'school', 'university', 'course', 'student', 'teach', 'academic']):
            return "Education"
        elif any(word in text for word in ['finance', 'banking', 'payment', 'money', 'investment', 'fintech', 'financial']):
            return "Finance"
        elif any(word in text for word in ['ecommerce', 'retail', 'shopping', 'marketplace', 'store', 'buy', 'sell']):
            return "E-commerce"
        elif any(word in text for word in ['manufacturing', 'production', 'factory', 'industrial', 'manufacture']):
            return "Manufacturing"
        elif any(word in text for word in ['tech', 'software', 'app', 'platform', 'digital', 'ai', 'technology']):
            return "Technology"
        else:
            return "General Business"
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get evaluation criteria specific to this agent. Override in subclasses."""
        return {
            "default_criteria": {
                "description": f"Evaluate {self.sub_parameter} thoroughly",
                "scoring_rubric": {
                    5: "Excellent - exceeds expectations significantly",
                    4: "Good - meets expectations with some excellence",
                    3: "Moderate - meets basic expectations",
                    2: "Weak - below expectations, needs improvement", 
                    1: "Poor - significantly below expectations"
                }
            }
        }
    
    def get_collaboration_dependencies(self) -> Dict[str, str]:
        """Get information about collaboration with other agents. Override in subclasses."""
        return {}
    
    def create_evaluation_prompt(self, idea_name: str, idea_concept: str, 
                               dependency_results: Optional[Dict[str, Any]] = None) -> str:
        """Create a comprehensive evaluation prompt for the agent"""
        
        # Determine industry context
        industry_context = self._determine_industry_context(idea_name, idea_concept)
        
        criteria = self.get_evaluation_criteria()
        
        prompt = f"""
        VALIDATION TASK: {self.sub_parameter} Assessment
        
        **IDEA DETAILS:**
        - Name: {idea_name}
        - Concept: {idea_concept}
        - Industry Context: {industry_context}
        
        **YOUR SPECIALIZATION:**
        You are evaluating "{self.sub_parameter}" within the "{self.cluster}" cluster.
        Weight in overall assessment: {self.weight}%
        
        **INDUSTRY CONTEXT AWARENESS:**
        IMPORTANT: Stay focused on your specific parameter within the context of the idea's industry.
        
        **FOR EDUCATIONAL VALUE PARAMETER:**
        - FOOD/FOOD-TECH ideas: Educational value = teaching nutrition, meal planning, cooking skills, healthy eating habits
        - HEALTHCARE ideas: Educational value = medical knowledge, health awareness, treatment education
        - EDUCATION ideas: Educational value = learning outcomes, skill development, knowledge transfer
        - FINANCE ideas: Educational value = financial literacy, investment knowledge, money management
        - E-COMMERCE ideas: Educational value = shopping skills, product knowledge, market awareness
        - MANUFACTURING ideas: Educational value = technical skills, production knowledge, quality standards
        
        **GENERAL GUIDELINES:**
        - For FOOD/TECH ideas: Focus on food industry metrics, not general education
        - For HEALTHCARE ideas: Focus on medical/healthcare industry standards  
        - For EDUCATION ideas: Focus on educational industry requirements
        - For FINANCE ideas: Focus on financial industry regulations and standards
        - For E-COMMERCE ideas: Focus on retail/commerce industry metrics
        - For MANUFACTURING ideas: Focus on production/industrial standards
        
        **EVALUATION CRITERIA:**
        {self._format_criteria(criteria)}
        
        **INDIAN MARKET CONTEXT:**
        Consider these factors specific to the Indian market:
        - Regulatory environment and compliance requirements
        - Infrastructure readiness and digital adoption patterns
        - Cultural preferences and user behavior
        - Economic conditions and purchasing power
        - Government policies and startup ecosystem support
        - Local competition and market dynamics
        
        **SCORING GUIDELINES:**
        - Score: 1.0-5.0 (use decimals for precision)
        - 5.0: Outstanding, exceptional performance
        - 4.0: Strong performance, above market standards
        - 3.0: Acceptable performance, meets basic requirements
        - 2.0: Below expectations, significant concerns
        - 1.0: Major deficiencies, fundamental problems
        
        **DEPENDENCY INSIGHTS & PEER REVIEW:**
        {self._format_dependency_results(dependency_results)}
        
        **CRITICAL EVALUATION GUIDELINES:**
        - Be SKEPTICAL of overly optimistic claims
        - Look for CONCRETE EVIDENCE to support assertions
        - Consider FAILURE SCENARIOS and worst-case outcomes
        - Question underlying assumptions rigorously
        - Compare against REALISTIC industry benchmarks
        - Don't inflate scores without solid justification
        - Remember: Most startups FAIL - assess accordingly
        
        **PEER CHALLENGE PROTOCOL:**
        MANDATORY: You MUST analyze and challenge other agents' assessments when available. For each dependency insight:
        1. If another agent gave a score >3.5, argue why it should be LOWER (be skeptical)
        2. If another agent gave a score <2.5, consider if it should be HIGHER (be balanced)
        3. Point out specific weaknesses in their reasoning
        4. Highlight evidence they missed or misinterpreted
        5. Question their assumptions and methodology
        6. DO NOT simply agree - your job is to provide a different perspective
        
        **REQUIRED OUTPUT FORMAT:**
        Provide a JSON response with exactly this structure:
        {{
            "score": <float between 1.0 and 5.0>,
            "confidence_level": <float between 0.0 and 1.0>,
            "explanation": "<detailed 3-4 sentence explanation of your assessment>",
            "assumptions": ["<assumption 1>", "<assumption 2>", "<assumption 3>"],
            "key_insights": ["<insight 1>", "<insight 2>"],
            "recommendations": ["<recommendation 1>", "<recommendation 2>"],
            "risk_factors": ["<risk 1>", "<risk 2>"],
            "peer_challenges": ["<challenge to other agent 1>", "<challenge to other agent 2>"],
            "evidence_gaps": ["<missing evidence 1>", "<missing evidence 2>"],
            "indian_market_considerations": "<specific factors for Indian market>"
        }}
        
        Ensure your evaluation is thorough, objective, and actionable.
        """
        
        return prompt
    
    def _format_criteria(self, criteria: Dict[str, Any]) -> str:
        """Format evaluation criteria for the prompt"""
        formatted = ""
        for criterion_name, criterion_data in criteria.items():
            formatted += f"\n{criterion_name.upper()}:\n"
            formatted += f"- {criterion_data.get('description', 'No description')}\n"
            
            if 'scoring_rubric' in criterion_data:
                formatted += "Scoring Rubric:\n"
                for score, desc in criterion_data['scoring_rubric'].items():
                    formatted += f"  {score}: {desc}\n"
            
            if 'factors' in criterion_data:
                formatted += f"Key Factors: {', '.join(criterion_data['factors'])}\n"
        
        return formatted
    
    def _format_dependency_results(self, dependency_results: Optional[Dict[str, Any]]) -> str:
        """Format dependency results for the prompt"""
        if not dependency_results:
            return "No dependency insights available for this evaluation."
        
        formatted = "🎯 PREVIOUS AGENT ASSESSMENTS TO ANALYZE & CHALLENGE:\n"
        formatted += "Your job is to scrutinize these assessments and provide counterarguments.\n\n"
        
        for agent_name, result in dependency_results.items():
            score = result.get('assigned_score', result.get('score', 'N/A'))
            explanation = result.get('explanation', 'No explanation available')
            confidence = result.get('confidence_level', 'N/A')
            assumptions = result.get('assumptions', [])
            
            formatted += f"**{agent_name} Assessment:**\n"
            formatted += f"- Score: {score}/5.0 (Confidence: {confidence})\n"
            formatted += f"- Reasoning: {explanation}\n"
            if assumptions:
                formatted += f"- Their Assumptions: {', '.join(assumptions)}\n"
            
            # Add challenge prompts based on score
            if isinstance(score, (int, float)):
                if score > 3.5:
                    formatted += f"⚠️ CHALLENGE THIS: Score seems too optimistic. Find flaws in their reasoning.\n"
                elif score < 2.5:
                    formatted += f"⚠️ CHALLENGE THIS: Score might be too harsh. Consider if they missed positives.\n"
                else:
                    formatted += f"⚠️ CHALLENGE THIS: Question their methodology and evidence.\n"
            formatted += "\n"
        
        return formatted
    
    def validate_output(self, output: Any) -> Dict[str, Any]:
        """Validate and standardize agent output"""
        try:
            if isinstance(output, str):
                # Try to extract JSON from string
                import re
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = self._parse_text_output(output)
            else:
                result = output
            
            # Ensure required fields are present
            standardized = {
                "score": float(result.get("score", 3.0)),
                "confidence_level": float(result.get("confidence_level", 0.7)),
                "explanation": str(result.get("explanation", f"Evaluation completed for {self.sub_parameter}")),
                "assumptions": list(result.get("assumptions", ["Standard market assumptions"])),
                "key_insights": list(result.get("key_insights", [])),
                "recommendations": list(result.get("recommendations", [])),
                "risk_factors": list(result.get("risk_factors", [])),
                "peer_challenges": list(result.get("peer_challenges", [])),
                "evidence_gaps": list(result.get("evidence_gaps", [])),
                "indian_market_considerations": str(result.get("indian_market_considerations", "Standard Indian market factors considered"))
            }
            
            # Validate score range
            standardized["score"] = max(1.0, min(5.0, standardized["score"]))
            standardized["confidence_level"] = max(0.0, min(1.0, standardized["confidence_level"]))
            
            return standardized
            
        except Exception as e:
            print(f"Error validating output for {self.agent_id}: {e}")
            return self._create_fallback_result()
    
    def _parse_text_output(self, text: str) -> Dict[str, Any]:
        """Parse text output when JSON parsing fails"""
        import re
        
        # Try to extract score
        score_match = re.search(r'score[:\s]*([0-9\.]+)', text.lower())
        score = float(score_match.group(1)) if score_match else 3.0
        
        # Extract explanation (first substantial paragraph)
        sentences = text.split('.')[:3]
        explanation = '. '.join(sentences).strip()
        
        return {
            "score": score,
            "confidence_level": 0.6,
            "explanation": explanation or f"Text-based evaluation for {self.sub_parameter}",
            "assumptions": ["Extracted from text analysis"],
            "key_insights": [],
            "recommendations": [],
            "risk_factors": [],
            "indian_market_considerations": "Standard considerations applied"
        }
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when validation fails"""
        return {
            "score": 3.0,
            "confidence_level": 0.5,
            "explanation": f"Fallback evaluation for {self.sub_parameter} due to processing error",
            "assumptions": ["Fallback evaluation applied"],
            "key_insights": ["Requires manual review"],
            "recommendations": ["Review evaluation methodology"],
            "risk_factors": ["Evaluation uncertainty"],
            "indian_market_considerations": "Standard market factors assumed"
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this agent"""
        return {
            "agent_id": self.agent_id,
            "cluster": self.cluster,
            "parameter": self.parameter,
            "sub_parameter": self.sub_parameter,
            "weight": self.weight,
            "dependencies": self.dependencies,
            "evaluation_criteria": self.get_evaluation_criteria(),
            "collaboration_info": self.get_collaboration_dependencies()
        }
    
    def record_evaluation(self, result: Dict[str, Any], processing_time: float):
        """Record the last evaluation for debugging and analysis"""
        self._last_evaluation = {
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_last_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get the last evaluation performed by this agent"""
        return self._last_evaluation


class AgentCollaborationManager:
    """Manages collaboration and dependencies between agents"""
    
    def __init__(self, agents: Dict[str, BaseValidationAgent]):
        self.agents = agents
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a dependency graph for agent execution order"""
        graph = {}
        for agent_id, agent in self.agents.items():
            graph[agent_id] = agent.dependencies
        return graph
    
    def get_execution_order(self) -> List[List[str]]:
        """Get execution order respecting dependencies (topological sort)"""
        # Simple dependency resolution - can be enhanced
        independent = []
        dependent = []
        
        for agent_id, agent in self.agents.items():
            if not agent.dependencies:
                independent.append(agent_id)
            else:
                dependent.append(agent_id)
        
        return [independent, dependent]  # Simplified for now
    
    def resolve_dependencies(self, agent_id: str, completed_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dependencies for a specific agent"""
        agent = self.agents[agent_id]
        dependency_results = {}
        
        for dependency in agent.dependencies:
            # Find which agent evaluated this dependency
            for eval_agent_id, result in completed_evaluations.items():
                eval_agent = self.agents.get(eval_agent_id)
                if eval_agent and dependency.lower().replace(' ', '_') in eval_agent.sub_parameter.lower().replace(' ', '_'):
                    dependency_results[dependency] = result
                    break
        
        return dependency_results
