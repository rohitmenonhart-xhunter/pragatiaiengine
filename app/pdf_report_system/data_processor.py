"""
Data Processor - Extracts insights from agent conversations
Processes raw validation data into structured report data
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class AgentDataProcessor:
    """Process agent evaluation data into structured report format"""
    
    def __init__(self, report_data: Dict[str, Any]):
        self.report_data = report_data
        # Try to get evaluated_data from multiple possible locations
        self.evaluated_data = (
            report_data.get('evaluated_data') or 
            report_data.get('raw_validation_result', {}).get('evaluated_data') or
            report_data.get('raw_validation_result', {}).get('evaluatedData') or
            report_data.get('validation_result', {}).get('evaluated_data') or
            report_data.get('detailed_analysis', {}).get('evaluated_data') or
            {}
        )
        
        logger.info(f"Evaluated data found: {bool(self.evaluated_data)}")
        if self.evaluated_data:
            logger.info(f"Evaluated data keys: {list(self.evaluated_data.keys())[:5]}...")
        
    def extract_all_agent_conversations(self) -> List[Dict[str, Any]]:
        """
        Extract ALL agent conversations with their insights
        Returns list of conversation objects with bullet points
        Handles both camelCase and snake_case field names
        """
        conversations = []
        
        if not self.evaluated_data or not isinstance(self.evaluated_data, dict):
            logger.warning("No evaluated_data found or invalid format")
            return conversations
        
        # Helper to get value with multiple key options
        def get_value(obj, *keys, default=None):
            for key in keys:
                if obj and key in obj:
                    return obj[key]
            return default
        
        for cluster_name, parameters in self.evaluated_data.items():
            if not isinstance(parameters, dict):
                continue
                
            for param_name, sub_params in parameters.items():
                if not isinstance(sub_params, dict):
                    continue
                    
                for sub_param_name, evaluation in sub_params.items():
                    if not isinstance(evaluation, dict):
                        continue
                    
                    # Handle both camelCase and snake_case
                    score = get_value(evaluation, 'assigned_score', 'assignedScore', 'score', default=0)
                    try:
                        score = float(score) if score else 0
                    except (ValueError, TypeError):
                        score = 0
                    
                    conversation = {
                        'cluster': cluster_name,
                        'parameter': param_name,
                        'sub_parameter': sub_param_name,
                        'score': score,
                        'explanation': get_value(evaluation, 'explanation', 'Explanation', default=''),
                        'strengths': get_value(evaluation, 'strengths', 'Strengths', default=[]) or [],
                        'weaknesses': get_value(evaluation, 'weaknesses', 'Weaknesses', default=[]) or [],
                        'key_insights': get_value(evaluation, 'key_insights', 'keyInsights', 'insights', default=[]) or [],
                        'recommendations': get_value(evaluation, 'recommendations', 'Recommendations', default=[]) or [],
                        'risk_factors': get_value(evaluation, 'risk_factors', 'riskFactors', default=[]) or [],
                        'assumptions': get_value(evaluation, 'assumptions', 'Assumptions', default=[]) or [],
                        'agent_id': get_value(evaluation, 'agent_id', 'agentId', default='')
                    }
                    conversations.append(conversation)
        
        logger.info(f"Extracted {len(conversations)} agent conversations from {len(self.evaluated_data)} clusters")
        return conversations
    
    def group_by_cluster(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by cluster"""
        grouped = {}
        for conv in conversations:
            cluster = conv['cluster']
            if cluster not in grouped:
                grouped[cluster] = []
            grouped[cluster].append(conv)
        return grouped
    
    def group_by_parameter(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by parameter within a cluster"""
        grouped = {}
        for conv in conversations:
            key = f"{conv['cluster']}::{conv['parameter']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(conv)
        return grouped
    
    def calculate_cluster_scores(self, conversations: List[Dict]) -> Dict[str, float]:
        """Calculate average score for each cluster"""
        cluster_scores = {}
        cluster_counts = {}
        
        for conv in conversations:
            cluster = conv['cluster']
            score = conv['score']
            
            if cluster not in cluster_scores:
                cluster_scores[cluster] = 0
                cluster_counts[cluster] = 0
            
            cluster_scores[cluster] += score
            cluster_counts[cluster] += 1
        
        # Calculate averages
        for cluster in cluster_scores:
            if cluster_counts[cluster] > 0:
                cluster_scores[cluster] = cluster_scores[cluster] / cluster_counts[cluster]
        
        return cluster_scores
    
    def extract_strengths_and_weaknesses(self, conversations: List[Dict]) -> Dict[str, List[str]]:
        """Extract all strengths and weaknesses from agent conversations"""
        all_strengths = []
        all_weaknesses = []
        
        for conv in conversations:
            # High-scoring items are strengths
            if conv['score'] >= 70:
                for strength in conv['strengths']:
                    all_strengths.append({
                        'text': strength,
                        'cluster': conv['cluster'],
                        'parameter': conv['sub_parameter'],
                        'score': conv['score']
                    })
            
            # Low-scoring items are weaknesses
            if conv['score'] < 60:
                for weakness in conv['weaknesses']:
                    all_weaknesses.append({
                        'text': weakness,
                        'cluster': conv['cluster'],
                        'parameter': conv['sub_parameter'],
                        'score': conv['score'],
                        'severity': self._get_severity(conv['score'])
                    })
        
        return {
            'strengths': sorted(all_strengths, key=lambda x: x['score'], reverse=True),
            'weaknesses': sorted(all_weaknesses, key=lambda x: x['score'])
        }
    
    def _get_severity(self, score: float) -> str:
        """Determine severity level based on score"""
        if score < 30:
            return 'Critical'
        elif score < 50:
            return 'High'
        else:
            return 'Moderate'
    
    def extract_recommendations(self, conversations: List[Dict]) -> List[Dict[str, Any]]:
        """Extract all recommendations from agents"""
        all_recommendations = []
        
        for conv in conversations:
            for rec in conv['recommendations']:
                all_recommendations.append({
                    'text': rec,
                    'cluster': conv['cluster'],
                    'parameter': conv['sub_parameter'],
                    'priority': 'High' if conv['score'] < 50 else 'Medium' if conv['score'] < 70 else 'Low'
                })
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        all_recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return all_recommendations
    
    def generate_cluster_summary(self, cluster_name: str, conversations: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive summary for a cluster"""
        cluster_convs = [c for c in conversations if c['cluster'] == cluster_name]
        
        if not cluster_convs:
            return {}
        
        # Calculate cluster score
        avg_score = sum(c['score'] for c in cluster_convs) / len(cluster_convs)
        
        # Get strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for conv in cluster_convs:
            if conv['score'] >= 70:
                strengths.extend(conv['strengths'])
            if conv['score'] < 60:
                weaknesses.extend(conv['weaknesses'])
        
        # Get all insights
        all_insights = []
        for conv in cluster_convs:
            all_insights.extend(conv['key_insights'])
        
        return {
            'cluster_name': cluster_name,
            'overall_score': avg_score,
            'status': self._get_status(avg_score),
            'num_parameters': len(cluster_convs),
            'strengths': strengths[:5],  # Top 5
            'weaknesses': weaknesses[:5],  # Top 5
            'key_insights': all_insights[:5],  # Top 5
            'parameters': cluster_convs
        }
    
    def _get_status(self, score: float) -> str:
        """Get status label for score"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Weak'
        else:
            return 'Poor'
    
    def process_complete_report_data(self) -> Dict[str, Any]:
        """Process all data and return complete structure for PDF generation"""
        conversations = self.extract_all_agent_conversations()
        
        if not conversations:
            logger.warning("No agent conversations found in report data")
            return {}
        
        # Group conversations
        grouped_by_cluster = self.group_by_cluster(conversations)
        
        # Calculate scores
        cluster_scores = self.calculate_cluster_scores(conversations)
        
        # Extract insights
        strengths_weaknesses = self.extract_strengths_and_weaknesses(conversations)
        recommendations = self.extract_recommendations(conversations)
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_name in grouped_by_cluster.keys():
            cluster_summaries[cluster_name] = self.generate_cluster_summary(
                cluster_name, conversations
            )
        
        # Overall score
        overall_score = self.report_data.get('overall_score', 0)
        
        return {
            'metadata': {
                'title': self.report_data.get('title', 'Validation Report'),
                'user_id': self.report_data.get('user_id', ''),
                'report_id': str(self.report_data.get('_id', '')),
                'created_at': self.report_data.get('created_at', ''),
                'overall_score': overall_score,
                'validation_outcome': self.report_data.get('validation_outcome', ''),
                'total_agents': len(conversations)
            },
            'cluster_scores': cluster_scores,
            'cluster_summaries': cluster_summaries,
            'strengths': strengths_weaknesses['strengths'],
            'weaknesses': strengths_weaknesses['weaknesses'],
            'recommendations': recommendations,
            'all_conversations': conversations
        }

