"""
Title Page Section Generator
"""

from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
from typing import List, Dict, Any
from datetime import datetime

from .colors import ReportColors


class TitlePageGenerator:
    """Generate professional title page"""
    
    def __init__(self, colors: ReportColors, styles: Dict):
        self.colors = colors
        self.styles = styles
    
    def generate(self, processed_data: Dict[str, Any]) -> List:
        """Generate title page elements"""
        elements = []
        metadata = processed_data['metadata']
        
        # Add spacing from top
        elements.append(Spacer(1, 1.5 * inch))
        
        # Company/Idea Title
        title = metadata['title']
        elements.append(Paragraph(
            f"<b>{title}</b>",
            ParagraphStyle(
                'ReportTitle',
                fontSize=36,
                textColor=self.colors.PRIMARY,
                alignment=TA_CENTER,
                spaceAfter=15,
                fontName='Helvetica-Bold'
            )
        ))
        
        # Subtitle
        elements.append(Paragraph(
            "AI-Powered Validation Report",
            ParagraphStyle(
                'Subtitle',
                fontSize=18,
                textColor=self.colors.TEXT_SECONDARY,
                alignment=TA_CENTER,
                spaceAfter=10
            )
        ))
        
        # Agent count
        elements.append(Paragraph(
            f"Analyzed by {metadata['total_agents']} Specialized AI Agents",
            ParagraphStyle(
                'AgentCount',
                fontSize=12,
                textColor=self.colors.TEXT_LIGHT,
                alignment=TA_CENTER,
                spaceAfter=40
            )
        ))
        
        # Score Card
        score = metadata['overall_score']
        score_color = self.colors.get_score_color(score)
        outcome = metadata['validation_outcome']
        
        score_content = Paragraph(
            f'''<para align="center">
            <font size="60" color="#{score_color.hexval()}"><b>{score:.1f}</b></font>
            <font size="24" color="#{self.colors.TEXT_SECONDARY.hexval()}">/100</font>
            <br/><br/>
            <font size="16" color="#{score_color.hexval()}"><b>{outcome}</b></font>
            </para>''',
            self.styles['Normal']
        )
        
        score_table = Table([[score_content]], colWidths=[5.5 * inch])
        score_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 4, score_color),
            ('ROUNDEDCORNERS', [15, 15, 15, 15]),
            ('TOPPADDING', (0, 0), (-1, -1), 35),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 35),
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 0.6 * inch))
        
        # Metadata table
        created_at = metadata['created_at']
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at).strftime('%B %d, %Y at %I:%M %p')
            except:
                created_at = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        else:
            created_at = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        report_id = metadata['report_id'][:16] + '...' if len(metadata['report_id']) > 16 else metadata['report_id']
        
        metadata_data = [
            ['Report ID:', report_id],
            ['Generated:', created_at],
            ['User ID:', metadata['user_id'] or 'N/A']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.8 * inch, 3.7 * inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('TEXTCOLOR', (0, 0), (0, -1), self.colors.TEXT_SECONDARY),
            ('TEXTCOLOR', (1, 0), (1, -1), self.colors.TEXT_PRIMARY),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        
        elements.append(metadata_table)
        
        # Page break
        elements.append(PageBreak())
        
        return elements

