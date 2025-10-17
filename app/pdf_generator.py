"""
PDF Report Generator for Validation Results
Generates comprehensive PDF reports with summaries and agent insights
"""

import os
from datetime import datetime
from typing import Dict, List, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white, grey
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib import colors


class ValidationReportGenerator:
    """Generate comprehensive PDF reports for validation results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='ReportSectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='ReportSubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='ReportBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            textColor=black,
            fontName='Helvetica'
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ReportScoreText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Cluster style
        self.styles.add(ParagraphStyle(
            name='ReportClusterText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            textColor=black,
            fontName='Helvetica'
        ))
    
    def generate_report(self, validation_result: Any, idea_name: str, idea_concept: str, 
                       output_path: str) -> str:
        """Generate comprehensive PDF report"""
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(idea_name, idea_concept, validation_result))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(validation_result))
        story.append(PageBreak())
        
        # Overall Assessment
        story.extend(self._create_overall_assessment(validation_result))
        story.append(PageBreak())
        
        # Cluster Analysis
        story.extend(self._create_cluster_analysis(validation_result))
        story.append(PageBreak())
        
        # Key Recommendations
        story.extend(self._create_recommendations_section(validation_result))
        story.append(PageBreak())
        
        # Critical Risks
        story.extend(self._create_risks_section(validation_result))
        story.append(PageBreak())
        
        # Market Insights
        story.extend(self._create_market_insights(validation_result))
        story.append(PageBreak())
        
        # Agent Details
        story.extend(self._create_agent_details(validation_result))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_title_page(self, idea_name: str, idea_concept: str, result: Any) -> List:
        """Create title page"""
        elements = []
        
        # Main title
        elements.append(Paragraph("PRAGATI AI VALIDATION REPORT", self.styles['ReportTitle']))
        elements.append(Spacer(1, 20))
        
        # Idea details
        elements.append(Paragraph(f"<b>Idea Name:</b> {idea_name}", self.styles['ReportBodyText']))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"<b>Concept Description:</b>", self.styles['ReportSectionHeader']))
        elements.append(Paragraph(idea_concept, self.styles['ReportBodyText']))
        elements.append(Spacer(1, 30))
        
        # Overall score
        score_color = self._get_score_color(result.overall_score)
        elements.append(Paragraph(f"<b>Overall Validation Score</b>", self.styles['ReportSectionHeader']))
        elements.append(Paragraph(f"<font color='{score_color}'>{result.overall_score:.2f}/5.0</font>", self.styles['ReportScoreText']))
        elements.append(Paragraph(f"<b>Outcome:</b> {result.validation_outcome.value}", self.styles['ReportBodyText']))
        elements.append(Spacer(1, 20))
        
        # Report metadata
        elements.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['ReportBodyText']))
        elements.append(Paragraph(f"<b>Agents Consulted:</b> {result.total_agents_consulted}", self.styles['ReportBodyText']))
        elements.append(Paragraph(f"<b>Processing Time:</b> {result.total_processing_time:.2f} seconds", self.styles['ReportBodyText']))
        elements.append(Paragraph(f"<b>Consensus Level:</b> {result.consensus_level:.1f}%", self.styles['ReportBodyText']))
        
        return elements
    
    def _create_executive_summary(self, result: Any) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        if result.overall_summary:
            elements.append(Paragraph(result.overall_summary, self.styles['ReportBodyText']))
        else:
            elements.append(Paragraph("Comprehensive analysis completed by 109+ specialized AI agents.", self.styles['ReportBodyText']))
        
        return elements
    
    def _create_overall_assessment(self, result: Any) -> List:
        """Create overall assessment section"""
        elements = []
        
        elements.append(Paragraph("OVERALL ASSESSMENT", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Score breakdown
        score_data = [
            ['Metric', 'Value'],
            ['Overall Score', f"{result.overall_score:.2f}/5.0"],
            ['Validation Outcome', result.validation_outcome.value],
            ['Agents Consulted', str(result.total_agents_consulted)],
            ['Consensus Level', f"{result.consensus_level:.1f}%"],
            ['Processing Time', f"{result.total_processing_time:.2f} seconds"]
        ]
        
        score_table = Table(score_data, colWidths=[2*inch, 2*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), white),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_cluster_analysis(self, result: Any) -> List:
        """Create cluster analysis section"""
        elements = []
        
        elements.append(Paragraph("CLUSTER ANALYSIS", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Cluster scores table
        cluster_data = [['Cluster', 'Score', 'Status']]
        for cluster, score in result.cluster_scores.items():
            status = self._get_status_text(score)
            cluster_data.append([cluster, f"{score:.2f}/5.0", status])
        
        cluster_table = Table(cluster_data, colWidths=[2*inch, 1*inch, 1.5*inch])
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), white),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(cluster_table)
        elements.append(Spacer(1, 20))
        
        # Detailed cluster summaries
        if result.cluster_summaries:
            elements.append(Paragraph("DETAILED CLUSTER SUMMARIES", self.styles['ReportSubsectionHeader']))
            elements.append(Spacer(1, 8))
            
            for cluster, summary in result.cluster_summaries.items():
                elements.append(Paragraph(f"<b>{cluster}</b>", self.styles['ReportClusterText']))
                elements.append(Paragraph(summary, self.styles['ReportBodyText']))
                elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_recommendations_section(self, result: Any) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("KEY RECOMMENDATIONS", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        if result.key_recommendations:
            for i, rec in enumerate(result.key_recommendations, 1):
                elements.append(Paragraph(f"{i}. {rec}", self.styles['ReportBodyText']))
                elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph("No specific recommendations generated.", self.styles['ReportBodyText']))
        
        return elements
    
    def _create_risks_section(self, result: Any) -> List:
        """Create critical risks section"""
        elements = []
        
        elements.append(Paragraph("CRITICAL RISKS", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        if result.critical_risks:
            for i, risk in enumerate(result.critical_risks, 1):
                elements.append(Paragraph(f"{i}. {risk}", self.styles['ReportBodyText']))
                elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph("No critical risks identified.", self.styles['ReportBodyText']))
        
        return elements
    
    def _create_market_insights(self, result: Any) -> List:
        """Create market insights section"""
        elements = []
        
        elements.append(Paragraph("MARKET INSIGHTS", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        if result.market_insights:
            for i, insight in enumerate(result.market_insights, 1):
                elements.append(Paragraph(f"{i}. {insight}", self.styles['ReportBodyText']))
                elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph("No specific market insights generated.", self.styles['ReportBodyText']))
        
        return elements
    
    def _create_agent_details(self, result: Any) -> List:
        """Create agent details section"""
        elements = []
        
        elements.append(Paragraph("AGENT EVALUATION DETAILS", self.styles['ReportSectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Top and bottom performing agents
        sorted_agents = sorted(result.agent_evaluations, key=lambda x: x.assigned_score, reverse=True)
        
        elements.append(Paragraph("TOP PERFORMING PARAMETERS", self.styles['ReportSubsectionHeader']))
        for i, agent in enumerate(sorted_agents[:5], 1):
            elements.append(Paragraph(f"{i}. {agent.sub_parameter} - {agent.assigned_score:.2f}/5.0", self.styles['ReportBodyText']))
            elements.append(Paragraph(f"   {agent.explanation}", self.styles['ReportBodyText']))
            elements.append(Spacer(1, 8))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("AREAS NEEDING IMPROVEMENT", self.styles['ReportSubsectionHeader']))
        for i, agent in enumerate(sorted_agents[-5:], 1):
            elements.append(Paragraph(f"{i}. {agent.sub_parameter} - {agent.assigned_score:.2f}/5.0", self.styles['ReportBodyText']))
            elements.append(Paragraph(f"   {agent.explanation}", self.styles['ReportBodyText']))
            elements.append(Spacer(1, 8))
        
        return elements
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score"""
        if score >= 4.0:
            return "green"
        elif score >= 3.0:
            return "orange"
        else:
            return "red"
    
    def _get_status_text(self, score: float) -> str:
        """Get status text based on score"""
        if score >= 4.0:
            return "EXCELLENT"
        elif score >= 3.0:
            return "GOOD"
        elif score >= 2.0:
            return "MODERATE"
        else:
            return "NEEDS IMPROVEMENT"
