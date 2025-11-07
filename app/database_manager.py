"""
MongoDB Database Manager for Pragati AI Engine
Handles saving and retrieving validation reports
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MongoDB operations for validation reports"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.mongodb_url = os.getenv("MONGODB_URL")
        self.database_name = os.getenv("DATABASE_NAME", "pragati_ai")
        
        if not self.mongodb_url:
            raise ValueError("MONGODB_URL environment variable is required")
        
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.mongodb_url)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def save_validation_report(self, user_id: str, title: str, validation_result: Dict[str, Any], 
                             idea_name: str, idea_concept: str, source_type: str = "manual") -> str:
        """
        Save a detailed validation report to MongoDB
        
        Args:
            user_id: User identifier
            title: Report title
            validation_result: Complete validation result from agents
            idea_name: Name of the idea
            idea_concept: Concept description
            source_type: "manual" or "pitch_deck"
            
        Returns:
            Report ID (MongoDB ObjectId as string)
        """
        try:
            # Generate detailed report data
            detailed_report = self._generate_detailed_report_data(
                validation_result, idea_name, idea_concept
            )
            
            # Create report document
            report_doc = {
                "_id": ObjectId(),
                "user_id": user_id,
                "title": title,
                "idea_name": idea_name,
                "idea_concept": idea_concept,
                "source_type": source_type,  # "manual" or "pitch_deck"
                "created_at": datetime.now(timezone.utc),
                "overall_score": validation_result.get("overall_score", 0),  # Already in 100 scale from agents
                "validation_outcome": validation_result.get("validation_outcome", "UNKNOWN"),
                "processing_time": validation_result.get("processing_time", 0),
                "agents_consulted": validation_result.get("api_calls_made", 0),
                "consensus_level": validation_result.get("consensus_level", 0) * 100,  # Convert to percentage
                
                # Detailed analysis data
                "detailed_analysis": detailed_report,
                
                # Raw validation data (for reference - keep original 5.0 scale)
                "raw_validation_result": validation_result,
                
                # Store evaluated_data at top level for easy access (agent conversations)
                "evaluated_data": validation_result.get("evaluated_data", {}),
                
                # Metadata
                "version": "3.0.0",
                "system": "Pragati AI Engine"
            }
            
            # Save to MongoDB
            collection = self.db.validation_reports
            result = collection.insert_one(report_doc)
            
            report_id = str(result.inserted_id)
            logger.info(f"✅ Saved validation report: {report_id} for user: {user_id}")
            
            return report_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {e}")
            raise
    
    def _generate_detailed_report_data(self, validation_result: Dict[str, Any], 
                                     idea_name: str, idea_concept: str) -> Dict[str, Any]:
        """Generate detailed report data structure"""
        
        # Extract cluster scores and analysis
        cluster_scores = validation_result.get("cluster_scores", {})
        evaluated_data = validation_result.get("evaluated_data", {})
        
        # Scores are now ALREADY in 100-scale from agents
        overall_score = validation_result.get("overall_score", 0)
        normalized_cluster_scores = cluster_scores  # Already normalized by agents
        
        # Analyze agent arguments and consensus
        agent_arguments = self._analyze_agent_arguments(evaluated_data)
        
        # Identify good and bad areas with ALL parameters
        good_areas, bad_areas, all_weak_parameters = self._identify_performance_areas(normalized_cluster_scores, evaluated_data)
        
        # Generate ALL cluster analyses (ensure all 7 clusters are covered)
        cluster_analyses = self._generate_all_cluster_analyses(normalized_cluster_scores, evaluated_data)
        
        # Generate recommendations and next steps
        recommendations = self._generate_detailed_recommendations(validation_result, bad_areas, all_weak_parameters)
        
        # Pitch deck improvements (if applicable)
        pitch_deck_improvements = self._generate_pitch_deck_improvements(bad_areas, evaluated_data)
        
        detailed_report = {
            "title": f"Validation Report: {idea_name}",
            "executive_summary": {
                "overall_score": overall_score,  # Now out of 100
                "outcome": validation_result.get("validation_outcome", "UNKNOWN"),
                "agents_consulted": validation_result.get("api_calls_made", 0),
                "consensus_level": validation_result.get("consensus_level", 0) * 100,  # Percentage
                "processing_time": validation_result.get("processing_time", 0),
                "summary_points": [
                    f"Overall viability score: {overall_score:.1f}/100",
                    f"Agents consulted: {validation_result.get('api_calls_made', 0)}",
                    f"Validation outcome: {validation_result.get('validation_outcome', 'UNKNOWN')}",
                    f"Top strength: {good_areas[0]['cluster'] if good_areas else 'N/A'}",
                    f"Main concern: {bad_areas[0]['cluster'] if bad_areas else 'N/A'}"
                ]
            },
            
            "parameters_validated": {
                "total_parameters": len(self._flatten_evaluated_data(evaluated_data)),
                "clusters": list(normalized_cluster_scores.keys()),
                "cluster_breakdown": {
                    cluster: {
                        "score": score,  # Now out of 100
                        "parameters": len(evaluated_data.get(cluster, {})),
                        "status": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
                    }
                    for cluster, score in normalized_cluster_scores.items()
                }
            },
            
            # Complete cluster analyses with all parameters
            "cluster_analyses": cluster_analyses,
            
            "agent_arguments": agent_arguments,
            
            "performance_analysis": {
                "good_areas": good_areas,
                "bad_areas": bad_areas,
                "neutral_areas": self._identify_neutral_areas(normalized_cluster_scores),
                "weak_parameters": all_weak_parameters  # All low-scoring parameters
            },
            
            "detailed_recommendations": recommendations,
            
            "next_steps": self._generate_next_steps(validation_result.get("validation_outcome"), bad_areas),
            
            "pitch_deck_improvements": pitch_deck_improvements,
            
            "market_insights": validation_result.get("market_insights", []),
            
            "risk_assessment": {
                "critical_risks": validation_result.get("critical_risks", []),
                "risk_level": self._assess_overall_risk_level(overall_score)  # Use normalized score
            }
        }
        
        return detailed_report
    
    def _analyze_agent_arguments(self, evaluated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how agents argued on different topics"""
        arguments = {
            "high_consensus_topics": [],
            "controversial_topics": [],
            "agent_disagreements": [],
            "consensus_patterns": {}
        }
        
        flattened_data = self._flatten_evaluated_data(evaluated_data)
        
        # Group by score ranges to identify consensus/disagreement
        score_groups = {"high": [], "medium": [], "low": []}
        
        for param_path, data in flattened_data.items():
            score = data.get("assignedScore", 0)
            if score >= 4.0:
                score_groups["high"].append((param_path, data))
            elif score >= 2.0:
                score_groups["medium"].append((param_path, data))
            else:
                score_groups["low"].append((param_path, data))
        
        # Identify controversial topics (medium scores often indicate disagreement)
        for param_path, data in score_groups["medium"]:
            if 2.5 <= data.get("assignedScore", 0) <= 3.5:
                arguments["controversial_topics"].append({
                    "parameter": param_path,
                    "score": data.get("assignedScore", 0),
                    "explanation": data.get("explanation", ""),
                    "reason": "Mixed signals - agents showed moderate agreement"
                })
        
        # High consensus topics
        for param_path, data in score_groups["high"]:
            arguments["high_consensus_topics"].append({
                "parameter": param_path,
                "score": data.get("assignedScore", 0),
                "explanation": data.get("explanation", "")
            })
        
        # Calculate consensus patterns
        arguments["consensus_patterns"] = {
            "strong_areas": len(score_groups["high"]),
            "weak_areas": len(score_groups["low"]),
            "disputed_areas": len([x for x in score_groups["medium"] if 2.5 <= x[1].get("assignedScore", 0) <= 3.5])
        }
        
        return arguments
    
    def _identify_performance_areas(self, cluster_scores: Dict[str, float], 
                                  evaluated_data: Dict[str, Any]) -> tuple:
        """Identify good and bad performance areas (scores are on 100 scale)"""
        good_areas = []
        bad_areas = []
        all_weak_parameters = []
        
        for cluster, score in cluster_scores.items():
            cluster_data = evaluated_data.get(cluster, {})
            
            if score >= 80:
                # Good area (80-100)
                good_areas.append({
                    "cluster": cluster,
                    "score": score,
                    "reason": "Strong performance across multiple parameters",
                    "key_strengths": self._extract_cluster_strengths(cluster_data),
                    "impact": "High positive impact on overall viability"
                })
            elif score <= 50:
                # Bad area (0-50)
                bad_areas.append({
                    "cluster": cluster,
                    "score": score,
                    "reason": "Significant challenges identified",
                    "key_weaknesses": self._extract_cluster_weaknesses(cluster_data),
                    "impact": "Major concern requiring immediate attention",
                    "improvement_priority": "High"
                })
            
            # Extract ALL weak parameters from this cluster
            weak_params = self._extract_all_weak_parameters(cluster, cluster_data)
            all_weak_parameters.extend(weak_params)
        
        # Sort weak parameters by score (lowest first)
        all_weak_parameters.sort(key=lambda x: x['score'])
        
        return good_areas, bad_areas, all_weak_parameters
    
    def _extract_cluster_strengths(self, cluster_data: Dict[str, Any]) -> List[str]:
        """Extract ALL strengths from cluster data (from agent output)"""
        strengths = []
        flattened = self._flatten_cluster_data(cluster_data)
        
        # Extract explicit strengths from agents PLUS high scores
        for param_path, data in flattened.items():
            # Get explicit strengths from agent
            agent_strengths = data.get('strengths', [])
            if agent_strengths:
                for strength in agent_strengths:
                    strengths.append(f"{param_path}: {strength}")
            # Also add high-scoring items
            elif data.get("assignedScore", 0) >= 80:  # 80+ on 100 scale
                strengths.append(f"{param_path} ({data.get('assignedScore', 0):.1f}/100): {data.get('explanation', '')[:100]}")
        
        return strengths  # Return ALL strengths, not just top 3
    
    def _extract_cluster_weaknesses(self, cluster_data: Dict[str, Any]) -> List[str]:
        """Extract ALL weaknesses from cluster data (from agent output)"""
        weaknesses = []
        flattened = self._flatten_cluster_data(cluster_data)
        
        # Extract explicit weaknesses from agents PLUS low scores
        for param_path, data in flattened.items():
            # Get explicit weaknesses from agent
            agent_weaknesses = data.get('weaknesses', [])
            if agent_weaknesses:
                for weakness in agent_weaknesses:
                    weaknesses.append(f"{param_path}: {weakness}")
            # Also add low-scoring items
            elif data.get("assignedScore", 0) < 60:  # Below 60 on 100 scale
                weaknesses.append(f"{param_path} ({data.get('assignedScore', 0):.1f}/100): {data.get('explanation', '')[:150]}")
        
        return weaknesses  # Return ALL weaknesses
    
    def _extract_all_weak_parameters(self, cluster_name: str, cluster_data: Dict[str, Any]) -> List[Dict]:
        """Extract ALL weak parameters from a cluster with detailed info (including agent weaknesses)"""
        weak_params = []
        flattened = self._flatten_cluster_data(cluster_data)
        
        # Find ALL low-scoring parameters OR those with explicit weaknesses
        for param_path, data in flattened.items():
            score = data.get("assignedScore", 0)  # Already in 100 scale from agents
            agent_weaknesses = data.get('weaknesses', [])
            
            # Include if score is low OR agent identified weaknesses
            if score < 60 or agent_weaknesses:
                # Combine explanation with agent weaknesses
                explanation_parts = [data.get('explanation', '')]
                if agent_weaknesses:
                    explanation_parts.extend([f"• {w}" for w in agent_weaknesses])
                full_explanation = " | ".join(explanation_parts)[:300]
                
                weak_params.append({
                    "cluster": cluster_name,
                    "parameter": param_path,
                    "score": score,
                    "explanation": full_explanation,
                    "weaknesses": agent_weaknesses,  # Include explicit weaknesses
                    "severity": "Critical" if score < 40 else "High" if score < 50 else "Moderate"
                })
        
        return weak_params
    
    def _generate_all_cluster_analyses(self, cluster_scores: Dict[str, float], 
                                     evaluated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis for ALL 7 clusters with all parameters"""
        analyses = {}
        
        # Ensure all expected clusters are covered
        all_clusters = [
            "Core Idea", "Market Opportunity", "Execution", 
            "Business Model", "Team", "Compliance", "Risk & Strategy"
        ]
        
        for cluster in all_clusters:
            score = cluster_scores.get(cluster, 0)
            cluster_data = evaluated_data.get(cluster, {})
            
            # Extract all parameters with their scores
            parameters = self._extract_all_parameters_with_scores(cluster_data)
            
            # Categorize parameters
            strong_params = [p for p in parameters if p['score'] >= 80]
            moderate_params = [p for p in parameters if 60 <= p['score'] < 80]
            weak_params = [p for p in parameters if p['score'] < 60]
            
            analyses[cluster] = {
                "score": score,
                "status": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement",
                "total_parameters": len(parameters),
                "parameters": {
                    "strong": strong_params,
                    "moderate": moderate_params,
                    "weak": weak_params
                },
                "summary_points": self._generate_cluster_summary_points(cluster, score, parameters)
            }
        
        return analyses
    
    def _extract_all_parameters_with_scores(self, cluster_data: Dict[str, Any]) -> List[Dict]:
        """Extract all parameters with COMPLETE agent data (100-scale)"""
        parameters = []
        flattened = self._flatten_cluster_data(cluster_data)
        
        for param_path, data in flattened.items():
            score = data.get("assignedScore", 0)  # Already in 100 scale from agents
            
            parameters.append({
                "name": param_path,
                "score": score,
                "explanation": data.get('explanation', 'No explanation provided'),
                "assumptions": data.get('assumptions', []),
                # Include ALL agent insights
                "key_insights": data.get('key_insights', []),
                "strengths": data.get('strengths', []),
                "weaknesses": data.get('weaknesses', []),
                "recommendations": data.get('recommendations', []),
                "risk_factors": data.get('risk_factors', []),
                "indian_market_considerations": data.get('indian_market_considerations', '')
            })
        
        # Sort by score (highest first)
        parameters.sort(key=lambda x: x['score'], reverse=True)
        
        return parameters
    
    def _generate_cluster_summary_points(self, cluster_name: str, score: float, 
                                       parameters: List[Dict]) -> List[str]:
        """Generate concise bullet point summary for a cluster"""
        points = []
        
        # Overall assessment
        if score >= 80:
            points.append(f"{cluster_name} shows excellent performance ({score:.1f}/100)")
        elif score >= 60:
            points.append(f"{cluster_name} demonstrates good performance ({score:.1f}/100) with room for optimization")
        else:
            points.append(f"{cluster_name} requires significant improvement ({score:.1f}/100)")
        
        # Top strength
        if parameters:
            top_param = parameters[0]
            points.append(f"Strongest area: {top_param['name']} ({top_param['score']:.1f}/100)")
        
        # Top weakness
        weak_params = [p for p in parameters if p['score'] < 60]
        if weak_params:
            weakest = weak_params[-1]
            points.append(f"Weakest area: {weakest['name']} ({weakest['score']:.1f}/100)")
        
        # Parameters count
        points.append(f"Total parameters evaluated: {len(parameters)}")
        
        return points
    
    def _identify_neutral_areas(self, cluster_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify neutral performance areas (50-80 range on 100 scale)"""
        neutral_areas = []
        
        for cluster, score in cluster_scores.items():
            if 50 < score < 80:
                neutral_areas.append({
                    "cluster": cluster,
                    "score": score,
                    "status": "Moderate performance with room for improvement"
                })
        
        return neutral_areas
    
    def _generate_detailed_recommendations(self, validation_result: Dict[str, Any], 
                                         bad_areas: List[Dict[str, Any]], 
                                         weak_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed recommendations with bullet points"""
        recommendations = []
        
        # Get existing recommendations from validation
        existing_recs = validation_result.get("key_recommendations", [])
        
        # Add recommendations for bad clusters
        for area in bad_areas:
            recommendations.append({
                "category": area["cluster"],
                "priority": "Critical",
                "score": area["score"],
                "recommendation": f"Immediate attention required for {area['cluster']}",
                "action_items": [
                    f"Current score: {area['score']:.1f}/100 - requires 20+ point improvement",
                    f"Impact: {area.get('impact', 'High')}",
                    *area.get("key_weaknesses", [])
                ]
            })
        
        # Add recommendations for specific weak parameters
        if weak_parameters:
            # Group by severity
            critical_params = [p for p in weak_parameters if p['severity'] == 'Critical']
            high_params = [p for p in weak_parameters if p['severity'] == 'High']
            
            if critical_params:
                recommendations.append({
                    "category": "Critical Parameters",
                    "priority": "Urgent",
                    "recommendation": f"Address {len(critical_params)} critical parameters (score < 40/100)",
                    "action_items": [
                        f"{p['parameter']}: {p['score']:.1f}/100 - {p['explanation'][:100]}..."
                        for p in critical_params[:5]
                    ]
                })
            
            if high_params:
                recommendations.append({
                    "category": "High Priority Parameters",
                    "priority": "High",
                    "recommendation": f"Improve {len(high_params)} parameters (score 40-50/100)",
                    "action_items": [
                        f"{p['parameter']}: {p['score']:.1f}/100 - {p['explanation'][:100]}..."
                        for p in high_params[:5]
                    ]
                })
        
        # Add general recommendations
        for i, rec in enumerate(existing_recs[:3]):
            recommendations.append({
                "category": "General Improvement",
                "priority": "Medium",
                "recommendation": rec,
                "action_items": ["Implement recommended changes", "Monitor progress regularly"],
                "expected_impact": "Positive impact on startup success"
            })
        
        return recommendations
    
    def _generate_next_steps(self, validation_outcome: str, bad_areas: List[Dict[str, Any]]) -> List[str]:
        """Generate specific next steps"""
        next_steps = []
        
        if validation_outcome == "EXCELLENT":
            next_steps = [
                "Proceed with MVP development immediately",
                "Prepare for Series A funding round",
                "Build strategic partnerships",
                "Focus on rapid market entry",
                "Scale team and operations"
            ]
        elif validation_outcome == "GOOD":
            next_steps = [
                "Develop detailed business plan",
                "Address identified weaknesses",
                "Build MVP and test with users",
                "Seek seed funding",
                "Validate market assumptions"
            ]
        elif validation_outcome == "MODERATE":
            next_steps = [
                "Strengthen weak areas before proceeding",
                "Conduct additional market research",
                "Consider pivoting business model",
                "Seek mentorship and advisory support",
                "Build proof of concept"
            ]
        else:
            next_steps = [
                "Fundamental reassessment required",
                "Extensive market validation needed",
                "Consider major pivoting",
                "Address core viability concerns",
                "Seek expert guidance"
            ]
        
        # Add specific steps for bad areas
        for area in bad_areas:
            next_steps.append(f"Urgent: Address {area['cluster']} weaknesses")
        
        return next_steps
    
    def _generate_pitch_deck_improvements(self, bad_areas: List[Dict[str, Any]], 
                                        evaluated_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific pitch deck improvement suggestions"""
        improvements = []
        
        for area in bad_areas:
            cluster = area["cluster"]
            
            if cluster == "Core Idea":
                improvements.append({
                    "slide": "Problem & Solution",
                    "improvement": "Strengthen problem statement and solution uniqueness",
                    "specific_action": "Add more compelling problem validation and differentiation"
                })
            elif cluster == "Market Opportunity":
                improvements.append({
                    "slide": "Market Size & Opportunity",
                    "improvement": "Provide stronger market validation data",
                    "specific_action": "Include TAM/SAM/SOM analysis and competitive landscape"
                })
            elif cluster == "Business Model":
                improvements.append({
                    "slide": "Business Model & Revenue",
                    "improvement": "Clarify revenue streams and financial projections",
                    "specific_action": "Add detailed unit economics and path to profitability"
                })
            elif cluster == "Team":
                improvements.append({
                    "slide": "Team",
                    "improvement": "Highlight relevant experience and expertise",
                    "specific_action": "Emphasize domain knowledge and execution track record"
                })
            elif cluster == "Execution":
                improvements.append({
                    "slide": "Product & Technology",
                    "improvement": "Demonstrate technical feasibility and scalability",
                    "specific_action": "Include technical architecture and development roadmap"
                })
        
        # General improvements
        improvements.extend([
            {
                "slide": "Appendix",
                "improvement": "Add detailed financial projections",
                "specific_action": "Include 3-year P&L, cash flow, and funding requirements"
            },
            {
                "slide": "Market Validation",
                "improvement": "Include customer validation data",
                "specific_action": "Add user interviews, surveys, and pilot program results"
            }
        ])
        
        return improvements
    
    def _assess_overall_risk_level(self, overall_score: float) -> str:
        """Assess overall risk level based on score (100 scale)"""
        if overall_score >= 80:
            return "Low Risk"
        elif overall_score >= 60:
            return "Moderate Risk"
        elif overall_score >= 40:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _flatten_evaluated_data(self, evaluated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested evaluated data structure"""
        flattened = {}
        
        for cluster, cluster_data in evaluated_data.items():
            for parameter, parameter_data in cluster_data.items():
                for sub_parameter, sub_data in parameter_data.items():
                    key = f"{cluster} > {parameter} > {sub_parameter}"
                    flattened[key] = sub_data
        
        return flattened
    
    def _flatten_cluster_data(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten cluster data structure"""
        flattened = {}
        
        for parameter, parameter_data in cluster_data.items():
            for sub_parameter, sub_data in parameter_data.items():
                key = f"{parameter} > {sub_parameter}"
                flattened[key] = sub_data
        
        return flattened
    
    def get_user_reports(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all reports for a user"""
        try:
            collection = self.db.validation_reports
            reports = list(collection.find(
                {"user_id": user_id},
                {
                    "_id": 1,
                    "title": 1,
                    "idea_name": 1,
                    "created_at": 1,
                    "overall_score": 1,
                    "validation_outcome": 1,
                    "source_type": 1
                }
            ).sort("created_at", -1).limit(limit))
            
            # Convert ObjectId to string
            for report in reports:
                report["_id"] = str(report["_id"])
            
            return reports
            
        except Exception as e:
            logger.error(f"❌ Failed to get user reports: {e}")
            return []
    
    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report by ID"""
        try:
            collection = self.db.validation_reports
            report = collection.find_one({"_id": ObjectId(report_id)})
            
            if report:
                report["_id"] = str(report["_id"])
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to get report by ID: {e}")
            return None
    
    def get_ai_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached AI-generated report for a validation report
        Returns None if no cached report exists
        """
        try:
            collection = self.db.validation_reports
            report = collection.find_one(
                {"_id": ObjectId(report_id)},
                {"ai_generated_report": 1, "ai_report_generated_at": 1}
            )
            
            if report and report.get("ai_generated_report"):
                return {
                    "ai_report": report.get("ai_generated_report"),
                    "generated_at": report.get("ai_report_generated_at"),
                    "cached": True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get AI report: {e}")
            return None
    
    def save_ai_report(self, report_id: str, ai_report: Dict[str, Any]) -> bool:
        """
        Save AI-generated report to MongoDB for caching
        Returns True if successful, False otherwise
        """
        try:
            collection = self.db.validation_reports
            result = collection.update_one(
                {"_id": ObjectId(report_id)},
                {
                    "$set": {
                        "ai_generated_report": ai_report,
                        "ai_report_generated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Saved AI report cache for report {report_id}")
                return True
            else:
                logger.warning(f"⚠️ Report {report_id} not found when saving AI report")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to save AI report: {e}")
            return False
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
