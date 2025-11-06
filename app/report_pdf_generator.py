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
            story.extend(self._create_title_page(report_data))
            story.append(PageBreak())
            
            # Executive Summary
            story.extend(self._create_executive_summary(report_data))
            story.append(PageBreak())
            
            # Detailed Analysis
            story.extend(self._create_detailed_analysis(report_data))
            
            # Recommendations
            if report_data.get('validation_result', {}).get('next_steps'):
                story.append(PageBreak())
                story.extend(self._create_recommendations(report_data))
            
            # Build PDF
            doc.build(story, onFirstPage=self._create_header_footer, 
                     onLaterPages=self._create_header_footer)
            
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def _create_title_page(self, report_data):
        """Create title page elements"""
        elements = []
        
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
        idea_title = report_data.get('title', 'Untitled Idea')
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
        score = report_data.get('validation_result', {}).get('overall_score', 0)
        score_color = self._get_score_color(score)
        
        score_data = [[
            Paragraph(
                f"<b>Overall Score</b><br/><font size='36' color='#{score_color.hexval()}'>{score}/100</font>",
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
    
    def _create_executive_summary(self, report_data):
        """Create executive summary section"""
        elements = []
        
        # Section header
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Validation outcome
        outcome = report_data.get('validation_result', {}).get('validation_outcome', '')
        if outcome:
            elements.append(Paragraph("<b>Validation Outcome:</b>", self.styles['SubsectionHeader']))
            elements.append(Paragraph(outcome, self.styles['CustomBody']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Key strengths
        strengths = report_data.get('validation_result', {}).get('strengths', [])
        if strengths:
            elements.append(Paragraph("<b>Key Strengths:</b>", self.styles['SubsectionHeader']))
            for strength in strengths[:5]:  # Top 5
                elements.append(Paragraph(f"• {strength}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Areas for improvement
        weaknesses = report_data.get('validation_result', {}).get('weaknesses', [])
        if weaknesses:
            elements.append(Paragraph("<b>Areas for Improvement:</b>", self.styles['SubsectionHeader']))
            for weakness in weaknesses[:5]:  # Top 5
                elements.append(Paragraph(f"• {weakness}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.15 * inch))
        
        return elements
    
    def _create_detailed_analysis(self, report_data):
        """Create detailed analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get evaluated criteria
        evaluated_data = report_data.get('validation_result', {}).get('evaluated_data', [])
        
        if not evaluated_data:
            elements.append(Paragraph("No detailed analysis available.", self.styles['CustomBody']))
            return elements
        
        # Create detailed analysis for each criterion
        for criterion_data in evaluated_data:
            criterion_elements = self._create_criterion_section(criterion_data)
            elements.extend(criterion_elements)
            elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _create_criterion_section(self, criterion_data):
        """Create section for a single criterion"""
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
        
        next_steps = report_data.get('validation_result', {}).get('next_steps', [])
        
        if not next_steps:
            elements.append(Paragraph("No specific recommendations available.", self.styles['CustomBody']))
            return elements
        
        for i, step in enumerate(next_steps, 1):
            step_text = f"<b>{i}.</b> {step}"
            elements.append(Paragraph(step_text, self.styles['CustomBody']))
            elements.append(Spacer(1, 0.08 * inch))
        
        return elements


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

