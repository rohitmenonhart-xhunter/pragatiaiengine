"""
Modern, Modular PDF Report Generator for Pragati AI
Beautiful, professional report design with advanced features
"""

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, Frame, PageTemplate
)
from reportlab.pdfgen import canvas
import logging

logger = logging.getLogger(__name__)

# Modern Color Palette
class ModernColors:
    """Beautiful, modern color scheme"""
    # Primary Colors
    PRIMARY = colors.HexColor('#2563EB')  # Blue
    SECONDARY = colors.HexColor('#7C3AED')  # Purple
    ACCENT = colors.HexColor('#10B981')  # Green
    
    # Score Colors (Gradient system)
    EXCELLENT = colors.HexColor('#059669')  # Emerald green
    GOOD = colors.HexColor('#10B981')  # Green
    MODERATE = colors.HexColor('#F59E0B')  # Amber
    WEAK = colors.HexColor('#EF4444')  # Red
    POOR = colors.HexColor('#DC2626')  # Dark red
    
    # Pros/Cons Colors
    PROS_BG = colors.HexColor('#ECFDF5')  # Light green
    PROS_BORDER = colors.HexColor('#10B981')  # Green
    CONS_BG = colors.HexColor('#FEF2F2')  # Light red
    CONS_BORDER = colors.HexColor('#EF4444')  # Red
    
    # UI Colors
    BACKGROUND = colors.HexColor('#F9FAFB')  # Light gray
    BORDER_LIGHT = colors.HexColor('#E5E7EB')  # Gray
    BORDER = colors.HexColor('#D1D5DB')  # Darker gray
    TEXT_PRIMARY = colors.HexColor('#111827')  # Almost black
    TEXT_SECONDARY = colors.HexColor('#6B7280')  # Gray
    TEXT_LIGHT = colors.HexColor('#9CA3AF')  # Light gray
    
    # Category Colors (for different clusters)
    CLUSTER_COLORS = {
        'Core Idea': colors.HexColor('#3B82F6'),  # Blue
        'Market Opportunity': colors.HexColor('#10B981'),  # Green
        'Execution': colors.HexColor('#8B5CF6'),  # Purple
        'Business Model': colors.HexColor('#F59E0B'),  # Amber
        'Team': colors.HexColor('#EC4899'),  # Pink
        'Compliance': colors.HexColor('#06B6D4'),  # Cyan
        'Risk & Strategy': colors.HexColor('#EF4444')  # Red
    }


class PDFSectionBase:
    """Base class for PDF sections"""
    
    def __init__(self, styles: Dict, colors: ModernColors):
        self.styles = styles
        self.colors = colors
    
    def create_section(self, report_data: Dict) -> List:
        """Create section elements - must be implemented by subclasses"""
        raise NotImplementedError


class TitlePageSection(PDFSectionBase):
    """Modern title page with key metrics"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Company/Idea Title
        title = report_data.get('title', 'Untitled Idea')
        elements.append(Spacer(1, 1.5 * inch))
        elements.append(Paragraph(
            f"<b>{title}</b>",
            ParagraphStyle(
                'TitleStyle',
                parent=self.styles['Title'],
                fontSize=32,
                textColor=self.colors.PRIMARY,
                alignment=TA_CENTER,
                spaceAfter=20
            )
        ))
        
        # Subtitle
        elements.append(Paragraph(
            "Comprehensive Validation Report",
            ParagraphStyle(
                'SubtitleStyle',
                fontSize=16,
                textColor=self.colors.TEXT_SECONDARY,
                alignment=TA_CENTER,
                spaceAfter=30
            )
        ))
        
        # Score Card - Prominent Display
        score = report_data.get('overall_score', 0)
        if score <= 5.0 and score > 0:
            score = score * 20  # Convert 5-scale to 100-scale if needed
        
        score_color = self._get_score_color(score)
        outcome = report_data.get('validation_outcome', 'N/A')
        
        # Create beautiful score card
        score_data = [[
            Paragraph(
                f'''<para align="center">
                <font size="48" color="#{score_color.hexval()}"><b>{score:.1f}</b></font>
                <font size="20" color="#{self.colors.TEXT_SECONDARY.hexval()}">/100</font>
                <br/><br/>
                <font size="14" color="#{self.colors.TEXT_SECONDARY.hexval()}">{outcome}</font>
                </para>''',
                self.styles['Normal']
            )
        ]]
        
        score_table = Table(score_data, colWidths=[5 * inch])
        score_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 3, score_color),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('TOPPADDING', (0, 0), (-1, -1), 30),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 30),
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Metadata in a clean table
        created_at = report_data.get('created_at', datetime.now().isoformat())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at).strftime('%B %d, %Y')
            except:
                created_at = datetime.now().strftime('%B %d, %Y')
        
        metadata = [
            ['Report ID:', str(report_data.get('_id', 'N/A'))[:20] + '...'],
            ['Generated:', created_at],
            ['User ID:', report_data.get('user_id', 'N/A')]
        ]
        
        metadata_table = Table(metadata, colWidths=[1.5 * inch, 3.5 * inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('TEXTCOLOR', (0, 0), (0, -1), self.colors.TEXT_SECONDARY),
            ('TEXTCOLOR', (1, 0), (1, -1), self.colors.TEXT_PRIMARY),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(metadata_table)
        elements.append(PageBreak())
        
        return elements
    
    def _get_score_color(self, score: float):
        """Get color based on score (100-scale)"""
        if score >= 80:
            return self.colors.EXCELLENT
        elif score >= 60:
            return self.colors.GOOD
        elif score >= 40:
            return self.colors.MODERATE
        elif score >= 20:
            return self.colors.WEAK
        else:
            return self.colors.POOR


class ExecutiveSummarySection(PDFSectionBase):
    """Executive summary with key findings, strengths, and weaknesses"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "Executive Summary",
            ParagraphStyle(
                'SectionHeader',
                fontSize=24,
                textColor=self.colors.PRIMARY,
                spaceAfter=20,
                spaceBefore=10,
                fontName='Helvetica-Bold'
            )
        ))
        
        detailed_analysis = report_data.get('detailed_analysis', {})
        exec_summary = detailed_analysis.get('executive_summary', {})
        
        if not exec_summary:
            elements.append(Paragraph(
                "Executive summary is being generated. Please check back later.",
                self.styles['Normal']
            ))
            return elements
        
        # Key Findings
        summary_points = exec_summary.get('summary_points', [])
        if summary_points:
            elements.append(Paragraph(
                "<b>📊 Key Findings:</b>",
                ParagraphStyle('SubHeader', fontSize=14, textColor=self.colors.TEXT_PRIMARY, 
                             spaceAfter=10, fontName='Helvetica-Bold')
            ))
            for point in summary_points:
                elements.append(Paragraph(
                    f"• {point}",
                    ParagraphStyle('BulletPoint', fontSize=11, leftIndent=20, spaceAfter=6)
                ))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Key Strengths (Pros) - Green themed
        performance_analysis = detailed_analysis.get('performance_analysis', {})
        good_areas = performance_analysis.get('good_areas', [])
        
        if good_areas:
            pros_elements = []
            pros_elements.append(Paragraph(
                "<b>✅ Key Strengths</b>",
                ParagraphStyle('ProsHeader', fontSize=14, textColor=self.colors.PROS_BORDER,
                             spaceAfter=8, fontName='Helvetica-Bold')
            ))
            
            for area in good_areas[:5]:  # Top 5
                pros_elements.append(Paragraph(
                    f"• <b>{area['cluster']}</b> ({area['score']:.1f}/100): {area['reason']}",
                    ParagraphStyle('ProsBullet', fontSize=10, leftIndent=15, spaceAfter=5,
                                 textColor=self.colors.TEXT_PRIMARY)
                ))
            
            # Wrap in colored box
            pros_table = Table([[pros_elements]], colWidths=[6.5 * inch])
            pros_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.PROS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.PROS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(pros_table)
            elements.append(Spacer(1, 0.15 * inch))
        
        # Critical Areas (Cons) - Red themed
        weak_parameters = performance_analysis.get('weak_parameters', [])
        
        if weak_parameters:
            cons_elements = []
            cons_elements.append(Paragraph(
                "<b>⚠️ Critical Areas for Improvement</b>",
                ParagraphStyle('ConsHeader', fontSize=14, textColor=self.colors.CONS_BORDER,
                             spaceAfter=8, fontName='Helvetica-Bold')
            ))
            
            # Group by severity
            critical = [p for p in weak_parameters if p.get('severity') == 'Critical']
            high = [p for p in weak_parameters if p.get('severity') == 'High']
            
            if critical:
                cons_elements.append(Paragraph(
                    f"<b>🚨 Critical Issues ({len(critical)}):</b>",
                    ParagraphStyle('CriticalLabel', fontSize=11, leftIndent=10, 
                                 textColor=self.colors.WEAK, spaceAfter=4, fontName='Helvetica-Bold')
                ))
                for param in critical[:5]:
                    cons_elements.append(Paragraph(
                        f"• {param['parameter']} ({param['score']:.1f}/100)",
                        ParagraphStyle('CriticalBullet', fontSize=10, leftIndent=20, spaceAfter=4)
                    ))
            
            if high:
                cons_elements.append(Paragraph(
                    f"<b>⚡ High Priority ({len(high)}):</b>",
                    ParagraphStyle('HighLabel', fontSize=11, leftIndent=10,
                                 textColor=self.colors.MODERATE, spaceAfter=4, fontName='Helvetica-Bold')
                ))
                for param in high[:5]:
                    cons_elements.append(Paragraph(
                        f"• {param['parameter']} ({param['score']:.1f}/100)",
                        ParagraphStyle('HighBullet', fontSize=10, leftIndent=20, spaceAfter=4)
                    ))
            
            # Wrap in colored box
            cons_table = Table([[cons_elements]], colWidths=[6.5 * inch])
            cons_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.CONS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.CONS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(cons_table)
        
        elements.append(PageBreak())
        return elements


class TAMSAMSOMSection(PDFSectionBase):
    """TAM/SAM/SOM explanation and breakdown"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "Market Size Analysis: TAM, SAM & SOM",
            ParagraphStyle(
                'SectionHeader',
                fontSize=20,
                textColor=self.colors.PRIMARY,
                spaceAfter=15,
                fontName='Helvetica-Bold'
            )
        ))
        
        # Explanation boxes
        definitions = [
            {
                'title': 'TAM (Total Addressable Market)',
                'definition': 'The total market demand for a product or service. Represents the maximum revenue opportunity if you achieved 100% market share.',
                'color': self.colors.CLUSTER_COLORS['Market Opportunity']
            },
            {
                'title': 'SAM (Serviceable Addressable Market)',
                'definition': 'The segment of the TAM that your product/service can realistically reach given geographical, regulatory, and competitive limitations.',
                'color': self.colors.CLUSTER_COLORS['Business Model']
            },
            {
                'title': 'SOM (Serviceable Obtainable Market)',
                'definition': 'The portion of SAM that you can realistically capture in the near term (1-3 years) considering competition, resources, and market penetration strategy.',
                'color': self.colors.CLUSTER_COLORS['Execution']
            }
        ]
        
        for definition in definitions:
            box_elements = []
            box_elements.append(Paragraph(
                f"<b>{definition['title']}</b>",
                ParagraphStyle('DefTitle', fontSize=13, textColor=definition['color'],
                             fontName='Helvetica-Bold', spaceAfter=6)
            ))
            box_elements.append(Paragraph(
                definition['definition'],
                ParagraphStyle('DefText', fontSize=10, spaceAfter=4)
            ))
            
            box_table = Table([[box_elements]], colWidths=[6.5 * inch])
            box_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
                ('BOX', (0, 0), (-1, -1), 2, definition['color']),
                ('ROUNDEDCORNERS', [6, 6, 6, 6]),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(box_table)
            elements.append(Spacer(1, 0.1 * inch))
        
        # Add visual representation (funnel)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            "<b>Market Funnel Visualization:</b>",
            ParagraphStyle('FunnelHeader', fontSize=12, fontName='Helvetica-Bold', spaceAfter=10)
        ))
        
        funnel_data = [
            ['TAM', '← Largest: Total market universe'],
            ['    SAM', '← Medium: Your serviceable segment'],
            ['        SOM', '← Smallest: Your achievable target']
        ]
        
        funnel_table = Table(funnel_data, colWidths=[1.5 * inch, 4 * inch])
        funnel_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, 0), self.colors.CLUSTER_COLORS['Market Opportunity']),
            ('TEXTCOLOR', (0, 1), (0, 1), self.colors.CLUSTER_COLORS['Business Model']),
            ('TEXTCOLOR', (0, 2), (0, 2), self.colors.CLUSTER_COLORS['Execution']),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(funnel_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements


class DetailedAnalysisSection(PDFSectionBase):
    """Detailed cluster-by-cluster analysis with clear demarkation"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "Detailed Analysis by Category",
            ParagraphStyle(
                'SectionHeader',
                fontSize=22,
                textColor=self.colors.PRIMARY,
                spaceAfter=20,
                fontName='Helvetica-Bold'
            )
        ))
        
        detailed_analysis = report_data.get('detailed_analysis', {})
        cluster_analyses = detailed_analysis.get('cluster_analyses', {})
        
        if not cluster_analyses:
            elements.append(Paragraph(
                "Detailed analysis is being processed.",
                self.styles['Normal']
            ))
            return elements
        
        # All 7 clusters
        all_clusters = [
            "Core Idea", "Market Opportunity", "Execution",
            "Business Model", "Team", "Compliance", "Risk & Strategy"
        ]
        
        for cluster_name in all_clusters:
            cluster_data = cluster_analyses.get(cluster_name, {})
            if cluster_data:
                elements.extend(self._create_cluster_section(cluster_name, cluster_data))
        
        elements.append(PageBreak())
        return elements
    
    def _create_cluster_section(self, cluster_name: str, cluster_data: Dict) -> List:
        """Create a beautiful section for each cluster"""
        elements = []
        
        # Get cluster color
        cluster_color = self.colors.CLUSTER_COLORS.get(cluster_name, self.colors.PRIMARY)
        
        # Cluster header with colored bar
        score = cluster_data.get('score', 0)
        status = cluster_data.get('status', 'N/A')
        
        header_data = [[
            Paragraph(
                f"<b>{cluster_name}</b>",
                ParagraphStyle('ClusterTitle', fontSize=16, textColor=cluster_color, 
                             fontName='Helvetica-Bold')
            ),
            Paragraph(
                f"<b>{score:.1f}/100</b> - {status}",
                ParagraphStyle('ClusterScore', fontSize=14, textColor=cluster_color,
                             alignment=TA_RIGHT, fontName='Helvetica-Bold')
            )
        ]]
        
        header_table = Table(header_data, colWidths=[4 * inch, 2.5 * inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 3, cluster_color),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(header_table)
        elements.append(Spacer(1, 0.15 * inch))
        
        # Summary points
        summary_points = cluster_data.get('summary_points', [])
        if summary_points:
            for point in summary_points[:3]:  # Top 3 points
                elements.append(Paragraph(
                    f"• {point}",
                    ParagraphStyle('ClusterBullet', fontSize=10, leftIndent=15, spaceAfter=5)
                ))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Parameters breakdown with clear demarkation
        all_parameters = cluster_data.get('all_parameters', [])
        if all_parameters:
            # Group by status: Strong, Moderate, Weak
            strong_params = [p for p in all_parameters if p.get('normalized_score', 0) >= 80]
            moderate_params = [p for p in all_parameters if 50 <= p.get('normalized_score', 0) < 80]
            weak_params = [p for p in all_parameters if p.get('normalized_score', 0) < 50]
            
            # Strong parameters (Green)
            if strong_params:
                elements.append(Paragraph(
                    f"<b>✅ Strong Areas ({len(strong_params)})</b>",
                    ParagraphStyle('ParamGroup', fontSize=11, textColor=self.colors.GOOD,
                                 fontName='Helvetica-Bold', spaceAfter=6, leftIndent=10)
                ))
                
                for param in strong_params[:5]:  # Show top 5
                    param_name = param.get('parameter', 'Unknown')
                    param_score = param.get('normalized_score', 0)
                    
                    param_data = [[
                        Paragraph(f"<b>{param_name}</b>", 
                                ParagraphStyle('ParamName', fontSize=10, fontName='Helvetica-Bold')),
                        Paragraph(f"<b>{param_score:.1f}/100</b>",
                                ParagraphStyle('ParamScore', fontSize=10, alignment=TA_RIGHT,
                                             textColor=self.colors.GOOD, fontName='Helvetica-Bold'))
                    ]]
                    
                    param_table = Table(param_data, colWidths=[5 * inch, 1 * inch])
                    param_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), self.colors.PROS_BG),
                        ('BOX', (0, 0), (-1, -1), 1, self.colors.PROS_BORDER),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    
                    elements.append(param_table)
                    elements.append(Spacer(1, 0.05 * inch))
                
                elements.append(Spacer(1, 0.1 * inch))
            
            # Weak parameters (Red)
            if weak_params:
                elements.append(Paragraph(
                    f"<b>⚠️ Areas Needing Improvement ({len(weak_params)})</b>",
                    ParagraphStyle('ParamGroup', fontSize=11, textColor=self.colors.WEAK,
                                 fontName='Helvetica-Bold', spaceAfter=6, leftIndent=10)
                ))
                
                for param in weak_params[:5]:  # Show all weak ones (critical)
                    param_name = param.get('parameter', 'Unknown')
                    param_score = param.get('normalized_score', 0)
                    weaknesses = param.get('agent_weaknesses', [])
                    
                    param_elements = []
                    param_elements.append(Paragraph(
                        f"<b>{param_name}</b> - <font color='#{self.colors.WEAK.hexval()}'>{param_score:.1f}/100</font>",
                        ParagraphStyle('ParamName', fontSize=10, fontName='Helvetica-Bold', spaceAfter=4)
                    ))
                    
                    # Show weaknesses
                    if weaknesses:
                        for weakness in weaknesses[:2]:
                            param_elements.append(Paragraph(
                                f"└─ {weakness}",
                                ParagraphStyle('Weakness', fontSize=9, leftIndent=10, spaceAfter=2,
                                             textColor=self.colors.TEXT_SECONDARY)
                            ))
                    
                    param_box = Table([[param_elements]], colWidths=[6 * inch])
                    param_box.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), self.colors.CONS_BG),
                        ('BOX', (0, 0), (-1, -1), 1, self.colors.CONS_BORDER),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    
                    elements.append(param_box)
                    elements.append(Spacer(1, 0.05 * inch))
        
        elements.append(Spacer(1, 0.25 * inch))
        return elements


class TRLTimelineSection(PDFSectionBase):
    """Technology Readiness Level timeline with AI-generated milestones"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "Technology Readiness Level (TRL) Timeline",
            ParagraphStyle(
                'SectionHeader',
                fontSize=20,
                textColor=self.colors.PRIMARY,
                spaceAfter=15,
                fontName='Helvetica-Bold'
            )
        ))
        
        # TRL Explanation
        elements.append(Paragraph(
            "Technology Readiness Levels (TRL) are a systematic measurement system to assess the maturity of a particular technology, from basic research (TRL 1) to full commercial deployment (TRL 9).",
            ParagraphStyle('Explanation', fontSize=10, textColor=self.colors.TEXT_SECONDARY, 
                         spaceAfter=12, alignment=TA_JUSTIFY)
        ))
        
        # TRL Levels with timeline
        trl_levels = [
            {'level': 'TRL 1-3', 'stage': 'Research & Proof of Concept', 'duration': '6-12 months', 
             'color': self.colors.POOR},
            {'level': 'TRL 4-6', 'stage': 'Development & Validation', 'duration': '12-18 months',
             'color': self.colors.MODERATE},
            {'level': 'TRL 7-8', 'stage': 'Demonstration & Testing', 'duration': '12-24 months',
             'color': self.colors.GOOD},
            {'level': 'TRL 9', 'stage': 'Commercial Deployment', 'duration': '24+ months',
             'color': self.colors.EXCELLENT}
        ]
        
        for i, trl in enumerate(trl_levels):
            trl_data = [[
                Paragraph(f"<b>{trl['level']}</b>",
                        ParagraphStyle('TRLLevel', fontSize=12, fontName='Helvetica-Bold',
                                     textColor=trl['color'])),
                Paragraph(f"<b>{trl['stage']}</b>",
                        ParagraphStyle('TRLStage', fontSize=11, fontName='Helvetica-Bold')),
                Paragraph(f"Est. {trl['duration']}",
                        ParagraphStyle('TRLDuration', fontSize=10, 
                                     textColor=self.colors.TEXT_SECONDARY, alignment=TA_RIGHT))
            ]]
            
            trl_table = Table(trl_data, colWidths=[1.2 * inch, 3.5 * inch, 1.8 * inch])
            trl_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
                ('BOX', (0, 0), (-1, -1), 2, trl['color']),
                ('ROUNDEDCORNERS', [6, 6, 6, 6]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            elements.append(trl_table)
            
            # Add arrow between stages
            if i < len(trl_levels) - 1:
                elements.append(Paragraph(
                    "↓",
                    ParagraphStyle('Arrow', fontSize=14, alignment=TA_CENTER, spaceAfter=2)
                ))
        
        elements.append(Spacer(1, 0.3 * inch))
        return elements


class ConclusionSection(PDFSectionBase):
    """Comprehensive conclusion with final verdict and recommendations"""
    
    def create_section(self, report_data: Dict) -> List:
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "Conclusion & Final Verdict",
            ParagraphStyle(
                'SectionHeader',
                fontSize=22,
                textColor=self.colors.PRIMARY,
                spaceAfter=20,
                fontName='Helvetica-Bold'
            )
        ))
        
        # Final Assessment
        score = report_data.get('overall_score', 0)
        if score <= 5.0 and score > 0:
            score = score * 20
        
        outcome = report_data.get('validation_outcome', 'N/A')
        
        # Verdict box
        if score >= 70:
            verdict = "✅ <b>RECOMMENDED FOR INVESTMENT</b>"
            verdict_text = "This idea demonstrates strong potential across multiple dimensions. With proper execution and resource allocation, it has a high likelihood of success."
            verdict_color = self.colors.EXCELLENT
        elif score >= 50:
            verdict = "⚠️ <b>PROCEED WITH CAUTION</b>"
            verdict_text = "This idea shows promise but has notable challenges that must be addressed. Recommend targeted improvements before significant investment."
            verdict_color = self.colors.MODERATE
        else:
            verdict = "⛔ <b>NOT RECOMMENDED</b>"
            verdict_text = "This idea faces substantial challenges across multiple critical areas. Significant pivots or foundational improvements are required."
            verdict_color = self.colors.WEAK
        
        verdict_elements = []
        verdict_elements.append(Paragraph(
            verdict,
            ParagraphStyle('Verdict', fontSize=16, textColor=verdict_color, 
                         fontName='Helvetica-Bold', spaceAfter=10, alignment=TA_CENTER)
        ))
        verdict_elements.append(Paragraph(
            verdict_text,
            ParagraphStyle('VerdictText', fontSize=11, spaceAfter=8, alignment=TA_JUSTIFY)
        ))
        
        verdict_table = Table([[verdict_elements]], colWidths=[6.5 * inch])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 3, verdict_color),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(verdict_table)
        elements.append(Spacer(1, 0.25 * inch))
        
        # Key Recommendations
        detailed_analysis = report_data.get('detailed_analysis', {})
        recommendations = detailed_analysis.get('detailed_recommendations', {})
        
        if recommendations:
            elements.append(Paragraph(
                "<b>Priority Actions:</b>",
                ParagraphStyle('RecHeader', fontSize=14, fontName='Helvetica-Bold',
                             textColor=self.colors.PRIMARY, spaceAfter=10)
            ))
            
            immediate_actions = recommendations.get('immediate_actions', [])
            for action in immediate_actions[:5]:
                elements.append(Paragraph(
                    f"• {action}",
                    ParagraphStyle('ActionBullet', fontSize=11, leftIndent=15, spaceAfter=6)
                ))
        
        # Closing statement
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            "This comprehensive analysis was conducted by Pragati AI Engine using 109+ specialized validation agents across 7 key clusters. For questions or detailed consultation, please contact your advisor.",
            ParagraphStyle('Footer', fontSize=9, textColor=self.colors.TEXT_LIGHT,
                         alignment=TA_CENTER, spaceAfter=10)
        ))
        
        return elements


class ModernPDFGenerator:
    """Main PDF generator coordinator"""
    
    def __init__(self):
        self.colors = ModernColors()
        self.styles = self._create_styles()
        
        # Initialize sections
        self.sections = {
            'title': TitlePageSection(self.styles, self.colors),
            'executive_summary': ExecutiveSummarySection(self.styles, self.colors),
            'tam_sam_som': TAMSAMSOMSection(self.styles, self.colors),
            'detailed_analysis': DetailedAnalysisSection(self.styles, self.colors),
            'trl_timeline': TRLTimelineSection(self.styles, self.colors),
            'conclusion': ConclusionSection(self.styles, self.colors)
        }
    
    def _create_styles(self) -> Dict:
        """Create modern paragraph styles"""
        styles = getSampleStyleSheet()
        
        # Add custom styles
        styles.add(ParagraphStyle(
            name='ModernTitle',
            fontSize=28,
            textColor=self.colors.PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='ModernBody',
            fontSize=11,
            textColor=self.colors.TEXT_PRIMARY,
            alignment=TA_LEFT,
            spaceAfter=8,
            leading=14
        ))
        
        return styles
    
    def generate(self, report_data: Dict) -> BytesIO:
        """Generate complete PDF report"""
        buffer = BytesIO()
        
        try:
            logger.info("Starting modern PDF generation...")
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch,
                title=report_data.get('title', 'Validation Report'),
                author='Pragati AI Engine'
            )
            
            elements = []
            
            # Add all sections in order
            logger.info("Adding title page...")
            elements.extend(self.sections['title'].create_section(report_data))
            
            logger.info("Adding executive summary...")
            elements.extend(self.sections['executive_summary'].create_section(report_data))
            
            logger.info("Adding TAM/SAM/SOM explanation...")
            elements.extend(self.sections['tam_sam_som'].create_section(report_data))
            elements.append(PageBreak())
            
            logger.info("Adding detailed analysis...")
            elements.extend(self.sections['detailed_analysis'].create_section(report_data))
            
            logger.info("Adding TRL timeline...")
            elements.extend(self.sections['trl_timeline'].create_section(report_data))
            elements.append(PageBreak())
            
            logger.info("Adding conclusion...")
            elements.extend(self.sections['conclusion'].create_section(report_data))
            
            # Build PDF
            logger.info("Building PDF document...")
            doc.build(elements)
            buffer.seek(0)
            
            pdf_size = buffer.getbuffer().nbytes
            logger.info(f"✅ PDF generated successfully! Size: {pdf_size:,} bytes ({pdf_size/1024:.1f} KB)")
            return buffer
            
        except Exception as e:
            logger.error(f"❌ Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback error PDF
            buffer = BytesIO()
            from reportlab.pdfgen import canvas as pdf_canvas
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, "PDF Generation Error")
            c.drawString(100, 730, f"Error: {str(e)[:100]}")
            c.drawString(100, 710, "Please contact support for assistance.")
            c.save()
            buffer.seek(0)
            return buffer


def generate_modern_pdf(report_data: Dict) -> BytesIO:
    """Main entry point for PDF generation"""
    generator = ModernPDFGenerator()
    return generator.generate(report_data)

