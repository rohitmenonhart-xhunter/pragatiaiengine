"""
Enhanced PDF Report Generator for Validation Reports
Generates detailed, professional PDF reports
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether, Image
)
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)


class ReportPDFGenerator:
    """Generate detailed PDF reports for idea validation"""
    
    # Color scheme
    PRIMARY_COLOR = HexColor('#4F46E5')  # Indigo
    SECONDARY_COLOR = HexColor('#06B6D4')  # Cyan
    SUCCESS_COLOR = HexColor('#10B981')  # Green
    WARNING_COLOR = HexColor('#F59E0B')  # Amber
    DANGER_COLOR = HexColor('#EF4444')  # Red
    BACKGROUND_LIGHT = HexColor('#F8FAFC')
    TEXT_PRIMARY = HexColor('#1E293B')
    TEXT_SECONDARY = HexColor('#64748B')
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.PRIMARY_COLOR,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section Header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=self.PRIMARY_COLOR,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=self.PRIMARY_COLOR,
            borderPadding=0,
            leftIndent=0
        ))
        
        # Subsection Header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=self.TEXT_PRIMARY,
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=self.TEXT_PRIMARY,
            alignment=TA_LEFT,
            spaceAfter=6,
            leading=14
        ))
        
        # Score display
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            fontSize=36,
            textColor=self.PRIMARY_COLOR,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=10
        ))
        
        # Metadata
        self.styles.add(ParagraphStyle(
            name='MetadataText',
            fontSize=9,
            textColor=self.TEXT_SECONDARY,
            alignment=TA_LEFT,
            spaceAfter=4
        ))
    
    def _get_score_color(self, score):
        """Get color based on score value"""
        if score >= 80:
            return self.SUCCESS_COLOR
        elif score >= 60:
            return self.SECONDARY_COLOR
        elif score >= 40:
            return self.WARNING_COLOR
        else:
            return self.DANGER_COLOR
    
    def _create_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page"""
        canvas_obj.saveState()
        
        # Footer
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(self.TEXT_SECONDARY)
        canvas_obj.drawString(
            inch, 
            0.5 * inch, 
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        )
        canvas_obj.drawRightString(
            letter[0] - inch, 
            0.5 * inch, 
            f"Page {canvas_obj.getPageNumber()}"
        )
        
        canvas_obj.restoreState()
    
    def generate_pdf(self, report_data):
        """
        Generate PDF from report data
        
        Args:
            report_data: Dictionary containing report information
            
        Returns:
            BytesIO object containing PDF data
        """
        try:
            logger.info(f"Starting PDF generation. Report has keys: {list(report_data.keys())}")
            
            # Validate report data
            if not report_data:
                raise ValueError("Report data is empty")
            
            # Check if detailed_analysis exists
            detailed_analysis = report_data.get('detailed_analysis', {})
            if not detailed_analysis:
                logger.warning("No detailed_analysis found in report_data. Creating minimal report.")
            else:
                logger.info(f"detailed_analysis has keys: {list(detailed_analysis.keys())}")
            
            buffer = io.BytesIO()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build document content
            story = []
            
            # Title page
            logger.info("Creating title page...")
            title_elements = self._create_title_page(report_data)
            if title_elements:
                story.extend(title_elements)
                story.append(PageBreak())
            else:
                logger.error("Title page creation returned empty!")
            
            # Executive Summary
            logger.info("Creating executive summary...")
            summary_elements = self._create_executive_summary(report_data)
            if summary_elements:
                story.extend(summary_elements)
                story.append(PageBreak())
            else:
                logger.warning("Executive summary is empty")
            
            # Detailed Analysis
            logger.info("Creating detailed analysis...")
            analysis_elements = self._create_detailed_analysis(report_data)
            if analysis_elements:
                story.extend(analysis_elements)
            else:
                logger.warning("Detailed analysis is empty")
            
            # Recommendations
            if detailed_analysis.get('detailed_recommendations') or detailed_analysis.get('next_steps'):
                logger.info("Creating recommendations...")
                story.append(PageBreak())
                story.extend(self._create_recommendations(report_data))
            
            # Ensure we have content
            if not story:
                raise ValueError("No content generated for PDF. Story is empty!")
            
            logger.info(f"Story has {len(story)} elements. Building PDF...")
            
            # Build PDF
            doc.build(story, onFirstPage=self._create_header_footer, 
                     onLaterPages=self._create_header_footer)
            
            buffer.seek(0)
            logger.info(f"PDF generated successfully. Buffer size: {buffer.getbuffer().nbytes} bytes")
            return buffer
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}", exc_info=True)
            # Create a minimal error PDF instead of raising
            return self._create_error_pdf(str(e))
    
    def _create_title_page(self, report_data):
        """Create title page elements"""
        elements = []
        
        try:
            # Add spacing
            elements.append(Spacer(1, 1.5 * inch))
            
            # Main title
            title = Paragraph(
                "IDEA VALIDATION REPORT",
                self.styles['CustomTitle']
            )
            elements.append(title)
            elements.append(Spacer(1, 0.3 * inch))
            
            # Idea title
            idea_title = report_data.get('title', report_data.get('idea_name', 'Untitled Idea'))
            title_para = Paragraph(
                f"<b>{idea_title}</b>",
                ParagraphStyle(
                    'IdeaTitle',
                    parent=self.styles['Heading2'],
                    fontSize=18,
                    textColor=self.TEXT_PRIMARY,
                    alignment=TA_CENTER,
                    spaceAfter=20
                )
            )
            elements.append(title_para)
            
            # Overall score with colored background
            score = report_data.get('overall_score', 0)
            # If score is still in 5.0 format, convert
            if score <= 5.0 and score > 0:
                score = score * 20
            score_color = self._get_score_color(score)
            
            score_data = [[
                Paragraph(
                    f"<b>Overall Score</b><br/><font size='36' color='#{score_color.hexval()}'>{score:.1f}/100</font>",
                    ParagraphStyle(
                        'ScoreDisplay',
                        alignment=TA_CENTER,
                        fontSize=12,
                        spaceAfter=10
                    )
                )
            ]]
            
            score_table = Table(score_data, colWidths=[4 * inch])
            score_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), self.BACKGROUND_LIGHT),
                ('BOX', (0, 0), (-1, -1), 2, score_color),
                ('TOPPADDING', (0, 0), (-1, -1), 20),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
            ]))
            
            elements.append(score_table)
            elements.append(Spacer(1, 0.5 * inch))
            
            # Metadata table
            metadata = [
                ['Report ID:', report_data.get('_id', 'N/A')],
                ['User ID:', report_data.get('user_id', 'N/A')],
                ['Generated:', report_data.get('created_at', datetime.now().isoformat())],
            ]
            
            metadata_table = Table(metadata, colWidths=[1.5 * inch, 3 * inch])
            metadata_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('TEXTCOLOR', (0, 0), (0, -1), self.TEXT_SECONDARY),
                ('TEXTCOLOR', (1, 0), (1, -1), self.TEXT_PRIMARY),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            
            elements.append(metadata_table)
            
            return elements
        except Exception as e:
            logger.error(f"Error creating title page: {e}")
            # Return minimal title page
            return [
                Spacer(1, 2 * inch),
                Paragraph("IDEA VALIDATION REPORT", self.styles['CustomTitle']),
                Spacer(1, 0.5 * inch),
                Paragraph("Error loading report details", self.styles['CustomBody'])
            ]
    
    def _create_executive_summary(self, report_data):
        """Create executive summary section"""
        elements = []
        
        try:
            # Section header
            elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            elements.append(Spacer(1, 0.1 * inch))
            
            # Get detailed analysis data
            detailed_analysis = report_data.get('detailed_analysis', {})
            exec_summary = detailed_analysis.get('executive_summary', {})
            
            # If no detailed analysis, create a basic summary from raw data
            if not exec_summary:
                logger.warning("No executive_summary in detailed_analysis, using raw data")
                score = report_data.get('overall_score', 0)
                if score <= 5.0:
                    score = score * 20
                elements.append(Paragraph(f"<b>Overall Score:</b> {score:.1f}/100", self.styles['CustomBody']))
                elements.append(Paragraph(f"<b>Outcome:</b> {report_data.get('validation_outcome', 'N/A')}", self.styles['CustomBody']))
                return elements
            
            # Summary points
            summary_points = exec_summary.get('summary_points', [])
            if summary_points:
                elements.append(Paragraph("<b>Key Findings:</b>", self.styles['SubsectionHeader']))
                for point in summary_points:
                    elements.append(Paragraph(f"• {point}", self.styles['CustomBody']))
                elements.append(Spacer(1, 0.15 * inch))
            
            # Validation outcome
            outcome = exec_summary.get('outcome', '')
            if outcome:
                elements.append(Paragraph(f"<b>Validation Outcome:</b> {outcome}", self.styles['CustomBody']))
                elements.append(Spacer(1, 0.15 * inch))
            
            # Key strengths from good areas
            performance_analysis = detailed_analysis.get('performance_analysis', {})
            good_areas = performance_analysis.get('good_areas', [])
            if good_areas:
                elements.append(Paragraph("<b>Key Strengths:</b>", self.styles['SubsectionHeader']))
                for area in good_areas[:3]:
                    elements.append(Paragraph(
                        f"• <b>{area['cluster']}</b> ({area['score']:.1f}/100): {area['reason']}", 
                        self.styles['CustomBody']
                    ))
                elements.append(Spacer(1, 0.15 * inch))
            
            # Areas for improvement from weak parameters - SHOW MORE
            weak_parameters = performance_analysis.get('weak_parameters', [])
            if weak_parameters:
                elements.append(Paragraph("<b>Critical Areas for Improvement:</b>", self.styles['SubsectionHeader']))
                
                # Group by severity
                critical = [p for p in weak_parameters if p['severity'] == 'Critical']
                high = [p for p in weak_parameters if p['severity'] == 'High']
                moderate = [p for p in weak_parameters if p['severity'] == 'Moderate']
                
                if critical:
                    elements.append(Paragraph(f"<b>🚨 Critical Issues ({len(critical)}):</b>", self.styles['CustomBody']))
                    for param in critical[:10]:  # Show up to 10 critical
                        weaknesses = param.get('weaknesses', [])
                        if weaknesses:
                            elements.append(Paragraph(f"• <b>{param['parameter']}</b> ({param['score']:.1f}/100):", self.styles['CustomBody']))
                            for weakness in weaknesses[:2]:
                                elements.append(Paragraph(f"  └─ {weakness}", self.styles['CustomBody']))
                        else:
                            elements.append(Paragraph(
                                f"• {param['parameter']} ({param['score']:.1f}/100)", 
                                self.styles['CustomBody']
                            ))
                
                if high:
                    elements.append(Paragraph(f"<b>⚠️ High Priority ({len(high)}):</b>", self.styles['CustomBody']))
                    for param in high[:8]:  # Show up to 8 high priority
                        elements.append(Paragraph(
                            f"• {param['parameter']} ({param['score']:.1f}/100)", 
                            self.styles['CustomBody']
                        ))
                
                if moderate:
                    elements.append(Paragraph(f"<b>⚡ Moderate Priority ({len(moderate)} items)</b>", self.styles['CustomBody']))
                
                elements.append(Spacer(1, 0.15 * inch))
            
            return elements
        except Exception as e:
            logger.error(f"Error creating executive summary: {e}")
            return [
                Paragraph("Executive Summary", self.styles['SectionHeader']),
                Paragraph("Error loading summary data", self.styles['CustomBody'])
            ]
    
    def _create_detailed_analysis(self, report_data):
        """Create detailed analysis section - ALL 7 clusters with parameters"""
        elements = []
        
        try:
            elements.append(Paragraph("Detailed Analysis by Cluster", self.styles['SectionHeader']))
            elements.append(Spacer(1, 0.1 * inch))
            
            # Get cluster analyses
            detailed_analysis = report_data.get('detailed_analysis', {})
            cluster_analyses = detailed_analysis.get('cluster_analyses', {})
            
            if not cluster_analyses:
                logger.warning("No cluster_analyses found in detailed_analysis")
                elements.append(Paragraph("Detailed analysis data is being processed. Please generate a new report.", self.styles['CustomBody']))
                return elements
            
            # Ensure all 7 clusters are covered
            all_clusters = [
                "Core Idea", "Market Opportunity", "Execution", 
                "Business Model", "Team", "Compliance", "Risk & Strategy"
            ]
            
            # Create detailed analysis for each cluster
            for cluster_name in all_clusters:
                cluster_data = cluster_analyses.get(cluster_name, {})
                if cluster_data:
                    try:
                        cluster_elements = self._create_cluster_analysis_section(cluster_name, cluster_data)
                        elements.extend(cluster_elements)
                        elements.append(Spacer(1, 0.2 * inch))
                    except Exception as cluster_error:
                        logger.error(f"Error creating cluster section for {cluster_name}: {cluster_error}")
                        elements.append(Paragraph(f"<b>{cluster_name}:</b> Data unavailable", self.styles['CustomBody']))
            
            return elements
        except Exception as e:
            logger.error(f"Error creating detailed analysis: {e}")
            return [
                Paragraph("Detailed Analysis", self.styles['SectionHeader']),
                Paragraph("Error loading detailed analysis", self.styles['CustomBody'])
            ]
    
    def _create_cluster_analysis_section(self, cluster_name, cluster_data):
        """Create COMPREHENSIVE section for a cluster with ALL parameters and agent insights"""
        elements = []
        
        score = cluster_data.get('score', 0)
        status = cluster_data.get('status', 'Unknown')
        
        # Cluster header with score
        score_color = self._get_score_color(score)
        header_text = f"<font color='#{self.PRIMARY_COLOR.hexval()}'><b>{cluster_name}</b></font> - <font color='#{score_color.hexval()}'><b>{score:.1f}/100</b></font> - {status}"
        
        elements.append(Paragraph(header_text, self.styles['SubsectionHeader']))
        
        # Summary points
        summary_points = cluster_data.get('summary_points', [])
        if summary_points:
            for point in summary_points:
                elements.append(Paragraph(f"• {point}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.08 * inch))
        
        # Parameters breakdown - SHOW ALL PARAMETERS
        parameters = cluster_data.get('parameters', {})
        all_params = parameters.get('strong', []) + parameters.get('moderate', []) + parameters.get('weak', [])
        
        if all_params:
            elements.append(Paragraph(f"<b>Detailed Parameter Analysis ({len(all_params)} parameters evaluated):</b>", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.06 * inch))
            
            for param in all_params:
                param_score = param.get('score', 0)
                param_name = param.get('name', 'Unknown')
                
                # Color code by score
                if param_score >= 80:
                    marker = "✓✓"
                    status_text = "Excellent"
                elif param_score >= 60:
                    marker = "✓"
                    status_text = "Good"
                else:
                    marker = "✗"
                    status_text = "Needs Improvement"
                
                # Parameter header
                elements.append(Paragraph(
                    f"<b>{marker} {param_name}: {param_score:.1f}/100</b> - {status_text}",
                    self.styles['CustomBody']
                ))
                
                # Explanation
                explanation = param.get('explanation', '')
                if explanation:
                    elements.append(Paragraph(f"  └─ {explanation[:150]}{'...' if len(explanation) > 150 else ''}", self.styles['CustomBody']))
                
                # Agent strengths (if any)
                strengths = param.get('strengths', [])
                if strengths:
                    elements.append(Paragraph("  <b>Strengths:</b>", self.styles['CustomBody']))
                    for strength in strengths:
                        elements.append(Paragraph(f"    • {strength}", self.styles['CustomBody']))
                
                # Agent weaknesses (if any)
                weaknesses = param.get('weaknesses', [])
                if weaknesses:
                    elements.append(Paragraph("  <b>Weaknesses:</b>", self.styles['CustomBody']))
                    for weakness in weaknesses:
                        elements.append(Paragraph(f"    • {weakness}", self.styles['CustomBody']))
                
                # Key insights (if any)
                insights = param.get('key_insights', [])
                if insights:
                    for insight in insights[:2]:  # Top 2 insights
                        elements.append(Paragraph(f"  💡 {insight}", self.styles['CustomBody']))
                
                # Recommendations (if any)
                recommendations = param.get('recommendations', [])
                if recommendations:
                    for rec in recommendations[:2]:  # Top 2 recommendations
                        elements.append(Paragraph(f"  → {rec}", self.styles['CustomBody']))
                
                elements.append(Spacer(1, 0.08 * inch))
        
        return elements
    
    def _create_criterion_section(self, criterion_data):
        """Create section for a single criterion (legacy support)"""
        elements = []
        
        criterion_name = criterion_data.get('criterion', 'Unknown Criterion')
        score = criterion_data.get('score', 0)
        
        # Criterion header with score
        score_color = self._get_score_color(score)
        header_text = f"<font color='#{self.PRIMARY_COLOR.hexval()}'><b>{criterion_name}</b></font> - <font color='#{score_color.hexval()}'><b>{score}/100</b></font>"
        
        elements.append(Paragraph(header_text, self.styles['SubsectionHeader']))
        
        # Reasoning
        reasoning = criterion_data.get('reasoning', '')
        if reasoning:
            elements.append(Paragraph("<b>Analysis:</b>", self.styles['CustomBody']))
            elements.append(Paragraph(reasoning, self.styles['CustomBody']))
            elements.append(Spacer(1, 0.08 * inch))
        
        # Suggestions
        suggestions = criterion_data.get('suggestions', [])
        if suggestions:
            elements.append(Paragraph("<b>Suggestions:</b>", self.styles['CustomBody']))
            for suggestion in suggestions:
                elements.append(Paragraph(f"• {suggestion}", self.styles['CustomBody']))
        
        return elements
    
    def _create_recommendations(self, report_data):
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommendations & Next Steps", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get recommendations from detailed analysis
        detailed_analysis = report_data.get('detailed_analysis', {})
        recommendations = detailed_analysis.get('detailed_recommendations', [])
        next_steps = detailed_analysis.get('next_steps', [])
        
        if not recommendations and not next_steps:
            elements.append(Paragraph("No specific recommendations available.", self.styles['CustomBody']))
            return elements
        
        # Detailed recommendations
        if recommendations:
            for rec in recommendations:
                category = rec.get('category', 'General')
                priority = rec.get('priority', 'Medium')
                recommendation = rec.get('recommendation', '')
                action_items = rec.get('action_items', [])
                
                # Priority color coding
                priority_color = self.DANGER_COLOR if priority in ['Critical', 'Urgent'] else self.WARNING_COLOR if priority == 'High' else self.PRIMARY_COLOR
                
                elements.append(Paragraph(
                    f"<font color='#{priority_color.hexval()}'><b>[{priority}]</b></font> <b>{category}:</b> {recommendation}",
                    self.styles['CustomBody']
                ))
                
                if action_items:
                    for item in action_items[:5]:  # Limit to 5 items per recommendation
                        elements.append(Paragraph(f"  • {item}", self.styles['CustomBody']))
                
                elements.append(Spacer(1, 0.1 * inch))
        
        # Next steps
        if next_steps:
            elements.append(Paragraph("<b>Immediate Next Steps:</b>", self.styles['SubsectionHeader']))
            for i, step in enumerate(next_steps[:10], 1):  # Top 10 steps
                step_text = f"<b>{i}.</b> {step}"
                elements.append(Paragraph(step_text, self.styles['CustomBody']))
                elements.append(Spacer(1, 0.06 * inch))
        
        return elements


    def _create_error_pdf(self, error_message):
        """Create a simple error PDF when main generation fails"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            
            story.append(Paragraph("PDF Generation Error", self.styles['CustomTitle']))
            story.append(Spacer(1, 0.5 * inch))
            story.append(Paragraph(f"An error occurred while generating the report:", self.styles['CustomBody']))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Error: {error_message}", self.styles['CustomBody']))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Please check the server logs for more details.", self.styles['CustomBody']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
        except:
            # If even error PDF fails, return minimal buffer
            buffer = io.BytesIO()
            buffer.write(b"PDF Generation Failed")
            buffer.seek(0)
            return buffer


def generate_report_pdf(report_data):
    """
    Utility function to generate PDF report
    
    Args:
        report_data: Dictionary containing report information
        
    Returns:
        BytesIO object containing PDF data
    """
    generator = ReportPDFGenerator()
    return generator.generate_pdf(report_data)

