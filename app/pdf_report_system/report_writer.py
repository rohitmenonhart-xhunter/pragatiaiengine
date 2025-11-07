"""
AI Report Writer
Reads all agent conversations and writes a comprehensive 20-page report
"""

from typing import Dict, List, Any
import json
import logging
from langchain_openai import ChatOpenAI
import os

logger = logging.getLogger(__name__)


class AIReportWriter:
    """
    AI that reads agent conversations and writes comprehensive reports
    Acts as a senior analyst synthesizing expert opinions
    """
    
    def __init__(self, progress_callback=None):
        """
        Initialize AI Report Writer
        
        Args:
            progress_callback: Optional function to call with progress updates
                              Signature: callback(message: str, progress: float)
        """
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4.1-mini",  # Using gpt-4.1-mini for comprehensive report generation
            max_tokens=4000,
            timeout=120
        )
        self.progress_callback = progress_callback
    
    def _update_progress(self, message: str, progress: float):
        """Send progress update if callback is provided"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def write_comprehensive_report(self, agent_conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Main method: Read all agent conversations and write comprehensive 20-30 page report
        
        Args:
            agent_conversations: List of all agent evaluations with their insights
            metadata: Report metadata (title, score, etc.)
        
        Returns:
            Comprehensive report with all sections: TAM/SAM/SOM, TRL, clusters, conclusion
        """
        total_steps = 7 + len(set(c.get('cluster', 'Unknown') for c in agent_conversations))
        current_step = 0
        
        logger.info(f"AI Report Writer analyzing {len(agent_conversations)} agent conversations...")
        self._update_progress("📊 Analyzing agent conversations...", 0)
        
        # Group conversations by cluster
        clustered_data = self._group_by_cluster(agent_conversations)
        current_step += 1
        self._update_progress(f"📋 Grouped conversations into {len(clustered_data)} clusters", 
                            current_step / total_steps * 100)
        
        # Write report for each cluster (detailed)
        cluster_reports = {}
        for idx, (cluster_name, conversations) in enumerate(clustered_data.items(), 1):
            logger.info(f"Writing detailed analysis for {cluster_name}...")
            self._update_progress(f"✍️ Writing {cluster_name} analysis ({idx}/{len(clustered_data)})...", 
                                (current_step / total_steps) * 100)
            cluster_reports[cluster_name] = self._write_cluster_report(
                cluster_name, conversations, metadata
            )
            current_step += 1
        
        # Write executive summary
        self._update_progress("📝 Writing Executive Summary...", (current_step / total_steps) * 100)
        executive_summary = self._write_executive_summary(
            cluster_reports, agent_conversations, metadata
        )
        current_step += 1
        
        # Write TAM/SAM/SOM analysis
        self._update_progress("📊 Analyzing Market Size (TAM/SAM/SOM)...", (current_step / total_steps) * 100)
        market_analysis = self._write_market_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write TRL analysis
        self._update_progress("🔬 Analyzing Technology Readiness (TRL)...", (current_step / total_steps) * 100)
        trl_analysis = self._write_trl_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write comprehensive pros and cons
        self._update_progress("⚖️ Analyzing Pros and Cons...", (current_step / total_steps) * 100)
        pros_cons = self._write_pros_cons_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write detailed weaknesses analysis
        self._update_progress("🔍 Analyzing Weaknesses...", (current_step / total_steps) * 100)
        weaknesses_analysis = self._write_weaknesses_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write conclusion
        self._update_progress("📋 Writing Conclusion...", (current_step / total_steps) * 100)
        conclusion = self._write_conclusion(
            cluster_reports, metadata, market_analysis, trl_analysis
        )
        current_step += 1
        
        self._update_progress("✅ Report writing complete!", 100)
        
        return {
            'executive_summary': executive_summary,
            'cluster_reports': cluster_reports,
            'market_analysis': market_analysis,
            'trl_analysis': trl_analysis,
            'pros_cons': pros_cons,
            'weaknesses_analysis': weaknesses_analysis,
            'conclusion': conclusion,
            'metadata': metadata
        }
    
    def _group_by_cluster(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by cluster"""
        grouped = {}
        for conv in conversations:
            cluster = conv.get('cluster', 'Unknown')
            if cluster not in grouped:
                grouped[cluster] = []
            grouped[cluster].append(conv)
        return grouped
    
    def _write_cluster_report(self, cluster_name: str, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Write comprehensive analysis for a single cluster
        Based on reading all expert conversations
        """
        
        # Prepare context for AI
        context = self._prepare_cluster_context(cluster_name, conversations)
        
        prompt = f"""You are a senior business analyst writing a comprehensive validation report. You have access to detailed evaluations from {len(conversations)} expert agents who analyzed the "{metadata['title']}" startup idea.

**Your Task**: Write a detailed, professional analysis of the **{cluster_name}** category based on the expert discussions below.

**Expert Agent Conversations**:
{context}

**Write a comprehensive analysis with the following structure**:

1. **Overview** (2-3 bullet points summarizing the cluster)
2. **Detailed Parameter Analysis** (for EACH parameter, write):
   • Parameter name and score
   • Key findings from experts (bullet points)
   • Strengths identified (bullet points if score > 70)
   • Weaknesses identified (bullet points if score < 60)
   • Expert recommendations (bullet points)

3. **Cluster Summary** (2-3 bullet points on overall cluster performance)

**Important Guidelines**:
- Use ONLY bullet points, NO long paragraphs
- Base everything on the expert conversations provided
- Include specific scores mentioned by experts
- Highlight both positive and negative aspects
- Be objective and analytical
- Each bullet point should be ONE clear statement
- Use professional business language

Return ONLY a JSON object with this structure:
{{
  "overview": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "parameters": [
    {{
      "name": "parameter name",
      "score": 75.0,
      "findings": ["finding 1", "finding 2"],
      "strengths": ["strength 1", "strength 2"],
      "weaknesses": ["weakness 1", "weakness 2"],
      "recommendations": ["recommendation 1", "recommendation 2"]
    }}
  ],
  "cluster_summary": ["summary point 1", "summary point 2", "summary point 3"]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            report = json.loads(content)
            
            # Calculate cluster score
            scores = [p['score'] for p in report['parameters'] if 'score' in p]
            avg_score = sum(scores) / len(scores) if scores else 0
            report['cluster_score'] = avg_score
            report['cluster_name'] = cluster_name
            
            logger.info(f"✅ Completed {cluster_name} analysis: {len(report['parameters'])} parameters")
            return report
            
        except Exception as e:
            logger.error(f"Error writing cluster report: {e}")
            # Fallback: return structured data from conversations
            return self._create_fallback_cluster_report(cluster_name, conversations)
    
    def _prepare_cluster_context(self, cluster_name: str, conversations: List[Dict]) -> str:
        """Prepare formatted context from agent conversations"""
        context_parts = []
        
        for i, conv in enumerate(conversations, 1):
            context_parts.append(f"\n--- Expert {i}: {conv.get('sub_parameter', 'Unknown')} Specialist ---")
            context_parts.append(f"Score: {conv.get('score', 0):.1f}/100")
            context_parts.append(f"Assessment: {conv.get('explanation', 'No explanation provided')}")
            
            if conv.get('strengths'):
                context_parts.append("Strengths:")
                for s in conv['strengths']:
                    context_parts.append(f"  • {s}")
            
            if conv.get('weaknesses'):
                context_parts.append("Weaknesses:")
                for w in conv['weaknesses']:
                    context_parts.append(f"  • {w}")
            
            if conv.get('key_insights'):
                context_parts.append("Key Insights:")
                for insight in conv['key_insights']:
                    context_parts.append(f"  • {insight}")
            
            if conv.get('recommendations'):
                context_parts.append("Recommendations:")
                for rec in conv['recommendations']:
                    context_parts.append(f"  • {rec}")
        
        return "\n".join(context_parts)
    
    def _write_executive_summary(self, cluster_reports: Dict, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write executive summary based on all cluster reports"""
        
        # Collect all scores
        all_scores = [conv['score'] for conv in conversations]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Identify top strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for conv in conversations:
            if conv['score'] >= 75:
                for s in conv.get('strengths', []):
                    strengths.append({'text': s, 'score': conv['score'], 'area': conv['sub_parameter']})
            if conv['score'] < 55:
                for w in conv.get('weaknesses', []):
                    weaknesses.append({'text': w, 'score': conv['score'], 'area': conv['sub_parameter']})
        
        # Sort by score
        strengths.sort(key=lambda x: x['score'], reverse=True)
        weaknesses.sort(key=lambda x: x['score'])
        
        prompt = f"""You are writing the Executive Summary for a startup validation report for "{metadata['title']}".

**Overall Score**: {avg_score:.1f}/100
**Validation Outcome**: {metadata['validation_outcome']}

**Top 10 Strengths** (from expert agents):
{self._format_items_for_prompt(strengths[:10])}

**Top 10 Critical Weaknesses** (from expert agents):
{self._format_items_for_prompt(weaknesses[:10])}

**Cluster Performance**:
{self._format_cluster_scores(cluster_reports)}

**Write an Executive Summary with**:
1. **Key Findings** (5-7 bullet points summarizing overall assessment)
2. **Major Strengths** (5-6 bullet points from the strengths above)
3. **Critical Concerns** (5-6 bullet points from the weaknesses above)
4. **Strategic Recommendations** (5-6 bullet points for immediate actions)

Use ONLY bullet points. Be concise and impactful.

Return JSON:
{{
  "key_findings": ["finding 1", "finding 2", ...],
  "major_strengths": ["strength 1", "strength 2", ...],
  "critical_concerns": ["concern 1", "concern 2", ...],
  "strategic_recommendations": ["recommendation 1", "recommendation 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            summary = json.loads(content)
            summary['overall_score'] = avg_score
            summary['outcome'] = metadata['validation_outcome']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error writing executive summary: {e}")
            return self._create_fallback_summary(strengths, weaknesses, avg_score, metadata)
    
    def _write_market_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive TAM/SAM/SOM analysis"""
        
        # Extract market-related conversations
        market_conversations = [c for c in conversations if 'market' in c.get('cluster', '').lower() or 
                              'market' in c.get('sub_parameter', '').lower()]
        
        prompt = f"""You are a market analyst writing a comprehensive market size analysis for "{metadata['title']}".

**Expert Evaluations**:
{self._prepare_cluster_context('Market Analysis', market_conversations if market_conversations else conversations[:10])}

**Write a detailed market analysis with**:

1. **TAM (Total Addressable Market)**:
   - Definition: Total market demand for this product/service globally
   - Estimated size (in USD/INR)
   - Growth rate and trends
   - Key assumptions (3-4 bullet points)

2. **SAM (Serviceable Available Market)**:
   - Definition: Portion of TAM that can be realistically served
   - Estimated size (in USD/INR)
   - Geographic and demographic focus
   - Market accessibility factors (3-4 bullet points)

3. **SOM (Serviceable Obtainable Market)**:
   - Definition: Portion of SAM that can be captured in 3-5 years
   - Estimated size (in USD/INR)
   - Market share assumptions
   - Competitive landscape considerations (3-4 bullet points)

4. **Market Opportunity Summary** (4-5 bullet points)

**Important**:
- Use ONLY bullet points
- Base estimates on expert evaluations
- Include Indian market context
- Be realistic and data-driven
- All numbers should be justified

Return JSON:
{{
  "tam": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "growth_rate": "Growth percentage",
    "assumptions": ["assumption 1", "assumption 2", ...],
    "trends": ["trend 1", "trend 2", ...]
  }},
  "sam": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "geographic_focus": "Primary markets",
    "accessibility_factors": ["factor 1", "factor 2", ...],
    "demographics": ["demographic 1", "demographic 2", ...]
  }},
  "som": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "market_share": "Target market share %",
    "competitive_landscape": ["consideration 1", "consideration 2", ...],
    "capture_strategy": ["strategy 1", "strategy 2", ...]
  }},
  "opportunity_summary": ["summary point 1", "summary point 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Market analysis (TAM/SAM/SOM) completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error writing market analysis: {e}")
            return self._create_fallback_market_analysis(metadata)
    
    def _write_trl_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write Technology Readiness Level (TRL) analysis with timeline"""
        
        # Extract technology/execution related conversations
        tech_conversations = [c for c in conversations if 'execution' in c.get('cluster', '').lower() or 
                            'technology' in c.get('sub_parameter', '').lower() or
                            'technical' in c.get('sub_parameter', '').lower()]
        
        prompt = f"""You are a technology analyst writing a TRL (Technology Readiness Level) analysis for "{metadata['title']}".

**Expert Evaluations on Technology & Execution**:
{self._prepare_cluster_context('Technology Analysis', tech_conversations if tech_conversations else conversations[:10])}

**TRL Levels (1-9)**:
- TRL 1: Basic principles observed
- TRL 2: Technology concept formulated
- TRL 3: Experimental proof of concept
- TRL 4: Technology validated in lab
- TRL 5: Technology validated in relevant environment
- TRL 6: Technology demonstrated in relevant environment
- TRL 7: System prototype demonstration in operational environment
- TRL 8: System complete and qualified
- TRL 9: Actual system proven in operational environment

**Write a comprehensive TRL analysis with**:

1. **Current TRL Assessment**:
   - Current TRL level (1-9)
   - Justification (3-4 bullet points)
   - Key technology components status

2. **TRL Progression Timeline**:
   - TRL 1-3: Timeline and milestones (2-3 bullet points)
   - TRL 4-6: Timeline and milestones (2-3 bullet points)
   - TRL 7-9: Timeline and milestones (2-3 bullet points)
   - Estimated time to market readiness

3. **Technology Risks & Challenges** (4-5 bullet points)

4. **Technology Strengths** (3-4 bullet points)

5. **Recommendations for TRL Advancement** (4-5 bullet points)

**Important**:
- Use ONLY bullet points
- Base assessment on expert evaluations
- Be realistic about timelines
- Include Indian market technology context

Return JSON:
{{
  "current_trl": {{
    "level": 5,
    "justification": ["justification 1", "justification 2", ...],
    "components_status": ["component 1 status", "component 2 status", ...]
  }},
  "timeline": {{
    "trl_1_3": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "trl_4_6": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "trl_7_9": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "time_to_market": "Estimated time"
  }},
  "risks": ["risk 1", "risk 2", ...],
  "strengths": ["strength 1", "strength 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ TRL analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error writing TRL analysis: {e}")
            return self._create_fallback_trl_analysis(metadata)
    
    def _write_pros_cons_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive pros and cons analysis"""
        
        # Collect all pros (strengths) and cons (weaknesses)
        all_pros = []
        all_cons = []
        
        for conv in conversations:
            score = conv.get('score', 0)
            area = conv.get('sub_parameter', 'Unknown')
            
            # Pros from high-scoring areas
            if score >= 70:
                for strength in conv.get('strengths', []):
                    all_pros.append({'text': strength, 'score': score, 'area': area})
                if conv.get('key_insights'):
                    for insight in conv['key_insights']:
                        if 'strong' in insight.lower() or 'positive' in insight.lower():
                            all_pros.append({'text': insight, 'score': score, 'area': area})
            
            # Cons from low-scoring areas
            if score < 60:
                for weakness in conv.get('weaknesses', []):
                    all_cons.append({'text': weakness, 'score': score, 'area': area})
                if conv.get('risk_factors'):
                    for risk in conv['risk_factors']:
                        all_cons.append({'text': risk, 'score': score, 'area': area})
        
        # Sort by score
        all_pros.sort(key=lambda x: x['score'], reverse=True)
        all_cons.sort(key=lambda x: x['score'])
        
        prompt = f"""You are an analyst synthesizing pros and cons for "{metadata['title']}".

**Overall Score**: {metadata['overall_score']:.1f}/100

**Top 15 Strengths** (from expert evaluations):
{self._format_items_for_prompt(all_pros[:15])}

**Top 15 Weaknesses** (from expert evaluations):
{self._format_items_for_prompt(all_cons[:15])}

**Write a comprehensive pros and cons analysis**:

1. **Major Advantages** (8-10 bullet points covering):
   - Market opportunities
   - Technology strengths
   - Business model advantages
   - Competitive advantages
   - Scalability potential

2. **Key Disadvantages** (8-10 bullet points covering):
   - Market challenges
   - Technology gaps
   - Business model concerns
   - Competitive threats
   - Execution risks

3. **Balanced Assessment** (4-5 bullet points weighing pros vs cons)

**Important**:
- Use ONLY bullet points
- Be specific and actionable
- Base on expert evaluations
- Include Indian market context

Return JSON:
{{
  "advantages": ["advantage 1", "advantage 2", ...],
  "disadvantages": ["disadvantage 1", "disadvantage 2", ...],
  "balanced_assessment": ["assessment point 1", "assessment point 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Pros and cons analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error writing pros/cons analysis: {e}")
            return self._create_fallback_pros_cons(all_pros, all_cons)
    
    def _write_weaknesses_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write detailed weaknesses analysis with severity and recommendations"""
        
        # Collect all weaknesses
        all_weaknesses = []
        for conv in conversations:
            score = conv.get('score', 0)
            area = conv.get('sub_parameter', 'Unknown')
            cluster = conv.get('cluster', 'Unknown')
            
            if score < 60:
                for weakness in conv.get('weaknesses', []):
                    severity = "Critical" if score < 40 else "High" if score < 50 else "Moderate"
                    all_weaknesses.append({
                        'text': weakness,
                        'score': score,
                        'area': area,
                        'cluster': cluster,
                        'severity': severity
                    })
        
        # Sort by severity and score
        all_weaknesses.sort(key=lambda x: (x['score'], x['severity'] == 'Critical', x['severity'] == 'High'))
        
        prompt = f"""You are an analyst writing a detailed weaknesses analysis for "{metadata['title']}".

**Overall Score**: {metadata['overall_score']:.1f}/100

**All Identified Weaknesses** (from expert evaluations):
{self._format_weaknesses_for_prompt(all_weaknesses)}

**Write a comprehensive weaknesses analysis**:

1. **Critical Weaknesses** (Score < 40/100):
   - List all critical weaknesses (4-6 bullet points)
   - Impact assessment for each
   - Immediate action required

2. **High Priority Weaknesses** (Score 40-50/100):
   - List high priority weaknesses (5-7 bullet points)
   - Impact assessment
   - Short-term action plan

3. **Moderate Weaknesses** (Score 50-60/100):
   - List moderate weaknesses (4-6 bullet points)
   - Impact assessment
   - Medium-term improvement plan

4. **Weakness Patterns** (3-4 bullet points identifying common themes)

5. **Remediation Strategy** (5-6 bullet points on addressing weaknesses)

**Important**:
- Use ONLY bullet points
- Be specific about each weakness
- Include actionable remediation steps
- Prioritize by severity

Return JSON:
{{
  "critical": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "high_priority": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "moderate": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "patterns": ["pattern 1", "pattern 2", ...],
  "remediation_strategy": ["strategy 1", "strategy 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Weaknesses analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error writing weaknesses analysis: {e}")
            return self._create_fallback_weaknesses(all_weaknesses)
    
    def _write_conclusion(self, cluster_reports: Dict, metadata: Dict, market_analysis: Dict = None, trl_analysis: Dict = None) -> Dict[str, Any]:
        """Write final conclusion and verdict with market and TRL context"""
        
        score = metadata['overall_score']
        
        market_context = ""
        if market_analysis:
            market_context = f"""
**Market Analysis Summary**:
- TAM: {market_analysis.get('tam', {}).get('size', 'N/A')}
- SAM: {market_analysis.get('sam', {}).get('size', 'N/A')}
- SOM: {market_analysis.get('som', {}).get('size', 'N/A')}
"""
        
        trl_context = ""
        if trl_analysis:
            trl_context = f"""
**Technology Readiness**:
- Current TRL: {trl_analysis.get('current_trl', {}).get('level', 'N/A')}
- Time to Market: {trl_analysis.get('timeline', {}).get('time_to_market', 'N/A')}
"""
        
        prompt = f"""Write a comprehensive conclusion for a startup validation report for "{metadata['title']}".

**Overall Score**: {score:.1f}/100

**Cluster Summary**:
{self._format_cluster_scores(cluster_reports)}
{market_context}
{trl_context}

**Write**:
1. **Final Verdict** (3-4 bullet points on investment recommendation considering market size and technology readiness)
2. **Path Forward** (5-6 bullet points on next steps including TRL progression)
3. **Success Factors** (4-5 bullet points on what's needed for success)
4. **Risk Mitigation** (4-5 bullet points on managing key risks)
5. **Market Opportunity Assessment** (3-4 bullet points on TAM/SAM/SOM potential)

Use bullet points only. Be decisive and actionable. Integrate market and technology insights.

Return JSON:
{{
  "final_verdict": ["verdict point 1", "verdict point 2", ...],
  "path_forward": ["next step 1", "next step 2", ...],
  "success_factors": ["factor 1", "factor 2", ...],
  "risk_mitigation": ["mitigation 1", "mitigation 2", ...],
  "market_assessment": ["assessment 1", "assessment 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            conclusion = json.loads(content)
            conclusion['investment_decision'] = self._get_investment_decision(score)
            
            return conclusion
            
        except Exception as e:
            logger.error(f"Error writing conclusion: {e}")
            return self._create_fallback_conclusion(score, metadata)
    
    def _format_items_for_prompt(self, items: List[Dict]) -> str:
        """Format items for LLM prompt"""
        lines = []
        for item in items:
            lines.append(f"• [{item['score']:.0f}/100] {item['area']}: {item['text']}")
        return "\n".join(lines) if lines else "None identified"
    
    def _format_cluster_scores(self, cluster_reports: Dict) -> str:
        """Format cluster scores for prompt"""
        lines = []
        for name, report in cluster_reports.items():
            score = report.get('cluster_score', 0)
            lines.append(f"• {name}: {score:.1f}/100")
        return "\n".join(lines) if lines else "No cluster data"
    
    def _get_investment_decision(self, score: float) -> str:
        """Get investment recommendation"""
        if score >= 75:
            return "STRONG YES - Recommended for investment"
        elif score >= 60:
            return "YES WITH CONDITIONS - Proceed with improvements"
        elif score >= 45:
            return "MAYBE - Significant concerns to address"
        else:
            return "NOT RECOMMENDED - Fundamental issues present"
    
    def _create_fallback_cluster_report(self, cluster_name: str, conversations: List[Dict]) -> Dict:
        """Create fallback report if AI writing fails"""
        parameters = []
        for conv in conversations:
            parameters.append({
                'name': conv.get('sub_parameter', 'Unknown'),
                'score': conv.get('score', 0),
                'findings': [conv.get('explanation', '')],
                'strengths': conv.get('strengths', []),
                'weaknesses': conv.get('weaknesses', []),
                'recommendations': conv.get('recommendations', [])
            })
        
        scores = [p['score'] for p in parameters]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'cluster_name': cluster_name,
            'cluster_score': avg_score,
            'overview': [f"Analysis based on {len(parameters)} expert evaluations"],
            'parameters': parameters,
            'cluster_summary': [f"Average score: {avg_score:.1f}/100"]
        }
    
    def _create_fallback_summary(self, strengths: List, weaknesses: List, score: float, metadata: Dict) -> Dict:
        """Create fallback summary"""
        return {
            'overall_score': score,
            'outcome': metadata['validation_outcome'],
            'key_findings': [f"Overall validation score: {score:.1f}/100"],
            'major_strengths': [s['text'] for s in strengths[:5]],
            'critical_concerns': [w['text'] for w in weaknesses[:5]],
            'strategic_recommendations': ["Review detailed cluster analysis for specific actions"]
        }
    
    def _create_fallback_conclusion(self, score: float, metadata: Dict) -> Dict:
        """Create fallback conclusion"""
        return {
            'investment_decision': self._get_investment_decision(score),
            'final_verdict': [f"Validation score: {score:.1f}/100"],
            'path_forward': ["Review detailed recommendations in each category"],
            'success_factors': ["Address identified weaknesses", "Leverage existing strengths"],
            'risk_mitigation': ["Monitor areas scoring below 50/100"],
            'market_assessment': ["Review market analysis section for detailed TAM/SAM/SOM"]
        }
    
    def _format_weaknesses_for_prompt(self, weaknesses: List[Dict]) -> str:
        """Format weaknesses for LLM prompt"""
        lines = []
        for w in weaknesses:
            lines.append(f"• [{w['severity']}] [{w['score']:.0f}/100] {w['area']} ({w['cluster']}): {w['text']}")
        return "\n".join(lines) if lines else "None identified"
    
    def _create_fallback_market_analysis(self, metadata: Dict) -> Dict:
        """Create fallback market analysis"""
        return {
            'tam': {
                'definition': 'Total Addressable Market - Global market size',
                'size': 'To be determined based on market research',
                'growth_rate': 'N/A',
                'assumptions': ['Market size requires detailed research', 'Growth rate depends on industry trends'],
                'trends': ['Market trends to be analyzed']
            },
            'sam': {
                'definition': 'Serviceable Available Market - Addressable market segment',
                'size': 'To be determined',
                'geographic_focus': 'Primary markets to be identified',
                'accessibility_factors': ['Market accessibility requires analysis'],
                'demographics': ['Target demographics to be defined']
            },
            'som': {
                'definition': 'Serviceable Obtainable Market - Realistic market share',
                'size': 'To be determined',
                'market_share': 'Target market share to be calculated',
                'competitive_landscape': ['Competitive analysis required'],
                'capture_strategy': ['Market capture strategy to be developed']
            },
            'opportunity_summary': ['Market analysis requires detailed research and validation']
        }
    
    def _create_fallback_trl_analysis(self, metadata: Dict) -> Dict:
        """Create fallback TRL analysis"""
        return {
            'current_trl': {
                'level': 3,
                'justification': ['Technology readiness requires technical assessment'],
                'components_status': ['Component status to be evaluated']
            },
            'timeline': {
                'trl_1_3': {
                    'timeframe': 'To be determined',
                    'milestones': ['Initial milestones to be defined']
                },
                'trl_4_6': {
                    'timeframe': 'To be determined',
                    'milestones': ['Development milestones to be planned']
                },
                'trl_7_9': {
                    'timeframe': 'To be determined',
                    'milestones': ['Market readiness milestones to be established']
                },
                'time_to_market': 'To be determined based on TRL progression'
            },
            'risks': ['Technology risks require detailed assessment'],
            'strengths': ['Technology strengths to be identified'],
            'recommendations': ['TRL advancement recommendations require technical review']
        }
    
    def _create_fallback_pros_cons(self, pros: List[Dict], cons: List[Dict]) -> Dict:
        """Create fallback pros/cons analysis"""
        return {
            'advantages': [p['text'] for p in pros[:8]] if pros else ['Advantages to be identified'],
            'disadvantages': [c['text'] for c in cons[:8]] if cons else ['Disadvantages to be identified'],
            'balanced_assessment': ['Balanced assessment requires comprehensive review']
        }
    
    def _create_fallback_weaknesses(self, weaknesses: List[Dict]) -> Dict:
        """Create fallback weaknesses analysis"""
        critical = [w for w in weaknesses if w['severity'] == 'Critical']
        high = [w for w in weaknesses if w['severity'] == 'High']
        moderate = [w for w in weaknesses if w['severity'] == 'Moderate']
        
        return {
            'critical': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Immediate attention required'}
                for w in critical[:6]
            ] if critical else [],
            'high_priority': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Short-term improvement needed'}
                for w in high[:7]
            ] if high else [],
            'moderate': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Medium-term improvement plan'}
                for w in moderate[:6]
            ] if moderate else [],
            'patterns': ['Weakness patterns require detailed analysis'],
            'remediation_strategy': ['Remediation strategy to be developed based on identified weaknesses']
        }

