"""
Main Report Generator
Coordinates all sections and generates final PDF
"""

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from typing import Dict, Any
import logging

from .colors import ReportColors
from .data_processor import AgentDataProcessor
from .sections_title import TitlePageGenerator
from .report_writer import AIReportWriter

logger = logging.getLogger(__name__)


def generate_validation_report(report_data: Dict[str, Any], progress_callback=None) -> BytesIO:
    """
    Main entry point for PDF generation
    
    Flow:
    1. Extract agent conversations from database
    2. AI reads conversations and writes comprehensive report
    3. Generate beautiful PDF from AI-written report
    
    Args:
        report_data: Report data from database
        progress_callback: Optional function to call with progress updates
                          Signature: callback(message: str, progress: float)
    """
    logger.info("🚀 Starting comprehensive report generation...")
    
    try:
        # Step 1: Process agent conversations
        if progress_callback:
            progress_callback("📂 Processing agent conversations...", 5)
        processor = AgentDataProcessor(report_data)
        processed_data = processor.process_complete_report_data()
        
        if not processed_data or not processed_data.get('all_conversations'):
            logger.warning("No agent conversations found, using detailed_analysis from database")
            # Fallback: Use existing detailed_analysis structure
            return _generate_pdf_from_detailed_analysis(report_data)
        
        logger.info(f"📊 Extracted {len(processed_data['all_conversations'])} agent conversations")
        
        # Step 2: AI writes comprehensive report by reading agent conversations
        logger.info("✍️  AI is writing comprehensive report from expert discussions...")
        writer = AIReportWriter(progress_callback=progress_callback)
        ai_written_report = writer.write_comprehensive_report(
            processed_data['all_conversations'],
            processed_data['metadata']
        )
        
        logger.info("✅ AI completed report writing")
        
        # Step 3: Merge AI-written report with processed data
        processed_data['ai_report'] = ai_written_report
        
        # Step 4: Generate beautiful PDF
        if progress_callback:
            progress_callback("🎨 Generating PDF document...", 95)
        logger.info("🎨 Generating beautiful PDF...")
        generator = PDFReportGenerator()
        pdf_buffer = generator.generate(processed_data)
        
        if progress_callback:
            progress_callback("✅ PDF ready for download!", 100)
        logger.info("✅ PDF generation completed successfully")
        return pdf_buffer
        
    except Exception as e:
        logger.error(f"❌ Error generating PDF: {e}")
        if progress_callback:
            progress_callback(f"❌ Error: {str(e)}", 0)
        import traceback
        traceback.print_exc()
        return _create_error_pdf(f"Error: {str(e)}")


class PDFReportGenerator:
    """Generates complete PDF report"""
    
    def __init__(self):
        self.colors = ReportColors()
        self.styles = self._create_styles()
    
    def _create_styles(self) -> Dict:
        """Create paragraph styles"""
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            fontSize=22,
            textColor=self.colors.PRIMARY,
            spaceAfter=18,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='SubHeader',
            fontSize=16,
            textColor=self.colors.TEXT_PRIMARY,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='BulletPoint',
            fontSize=11,
            leftIndent=20,
            spaceAfter=6,
            textColor=self.colors.TEXT_PRIMARY
        ))
        
        styles.add(ParagraphStyle(
            name='SubBullet',
            fontSize=10,
            leftIndent=35,
            spaceAfter=4,
            textColor=self.colors.TEXT_SECONDARY
        ))
        
        return styles
    
    def generate(self, processed_data: Dict[str, Any]) -> BytesIO:
        """Generate complete PDF"""
        buffer = BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=processed_data['metadata']['title'],
            author='Pragati AI Engine'
        )
        
        elements = []
        
        # Title Page
        title_gen = TitlePageGenerator(self.colors, self.styles)
        elements.extend(title_gen.generate(processed_data))
        
        # Executive Summary
        elements.extend(self._generate_executive_summary(processed_data))
        
        # Market Analysis (TAM/SAM/SOM)
        elements.extend(self._generate_market_analysis(processed_data))
        
        # TRL Analysis
        elements.extend(self._generate_trl_analysis(processed_data))
        
        # Detailed Analysis by Cluster
        elements.extend(self._generate_cluster_analysis(processed_data))
        
        # Pros and Cons Analysis
        elements.extend(self._generate_pros_cons(processed_data))
        
        # Weaknesses Analysis
        elements.extend(self._generate_weaknesses_analysis(processed_data))
        
        # Recommendations
        elements.extend(self._generate_recommendations(processed_data))
        
        # Conclusion
        elements.extend(self._generate_conclusion(processed_data))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return buffer
    
    def _generate_executive_summary(self, data: Dict) -> list:
        """Generate executive summary section using AI-written content"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get AI-written summary
        ai_summary = data.get('ai_report', {}).get('executive_summary', {})
        score = data['metadata']['overall_score']
        
        # Overall Assessment
        elements.append(Paragraph(
            f"<b>Overall Validation Score:</b> <font color='#{self.colors.get_score_color(score).hexval()}'>{score:.1f}/100</font>",
            self.styles['SubHeader']
        ))
        elements.append(Spacer(1, 0.15 * inch))
        
        # Key Findings
        key_findings = ai_summary.get('key_findings', [])
        if key_findings:
            elements.append(Paragraph("<b>📊 Key Findings</b>", self.styles['SubHeader']))
            for finding in key_findings:
                elements.append(Paragraph(f"• {finding}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Major Strengths (Green Box)
        strengths = ai_summary.get('major_strengths', [])
        if strengths:
            elements.append(Paragraph("<b>✅ Major Strengths</b>", self.styles['SubHeader']))
            
            strength_bullets = []
            for s in strengths:
                strength_bullets.append(Paragraph(f"• {s}", self.styles['BulletPoint']))
            
            strength_table = Table([[strength_bullets]], colWidths=[6.5 * inch])
            strength_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.PROS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.PROS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(strength_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Critical Concerns (Red Box)
        concerns = ai_summary.get('critical_concerns', [])
        if concerns:
            elements.append(Paragraph("<b>⚠️ Critical Concerns</b>", self.styles['SubHeader']))
            
            concern_bullets = []
            for c in concerns:
                concern_bullets.append(Paragraph(f"• {c}", self.styles['BulletPoint']))
            
            concern_table = Table([[concern_bullets]], colWidths=[6.5 * inch])
            concern_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.CONS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.CONS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(concern_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Strategic Recommendations
        recommendations = ai_summary.get('strategic_recommendations', [])
        if recommendations:
            elements.append(Paragraph("<b>🎯 Strategic Recommendations</b>", self.styles['SubHeader']))
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", self.styles['BulletPoint']))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_cluster_analysis(self, data: Dict) -> list:
        """Generate detailed cluster-by-cluster analysis using AI-written content"""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis by Category", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get AI-written cluster reports
        cluster_reports = data.get('ai_report', {}).get('cluster_reports', {})
        
        if not cluster_reports:
            # Fallback to original summaries
            for cluster_name, summary in data['cluster_summaries'].items():
                elements.extend(self._generate_single_cluster(cluster_name, summary))
        else:
            # Use AI-written reports
            for cluster_name, ai_report in cluster_reports.items():
                elements.extend(self._generate_ai_written_cluster(cluster_name, ai_report))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_ai_written_cluster(self, cluster_name: str, ai_report: Dict) -> list:
        """Generate cluster section from AI-written report"""
        elements = []
        
        cluster_color = self.colors.get_cluster_color(cluster_name)
        score = ai_report.get('cluster_score', 0)
        
        # Cluster Header
        status = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        header_content = [[
            Paragraph(f"<b>{cluster_name}</b>",
                    ParagraphStyle('ClusterTitle', fontSize=16, textColor=cluster_color,
                                 fontName='Helvetica-Bold')),
            Paragraph(f"<b>{score:.1f}/100</b> - {status}",
                    ParagraphStyle('ClusterScore', fontSize=14, textColor=cluster_color,
                                 alignment=TA_CENTER, fontName='Helvetica-Bold'))
        ]]
        
        header_table = Table(header_content, colWidths=[4.5 * inch, 2 * inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 3, cluster_color),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.15 * inch))
        
        # Overview
        overview = ai_report.get('overview', [])
        if overview:
            elements.append(Paragraph("<b>Overview:</b>", self.styles['SubHeader']))
            for point in overview:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Parameters
        parameters = ai_report.get('parameters', [])
        for param in parameters:
            elements.extend(self._generate_ai_parameter(param, cluster_color))
        
        # Cluster Summary
        summary = ai_report.get('cluster_summary', [])
        if summary:
            elements.append(Paragraph("<b>Summary:</b>", self.styles['SubHeader']))
            for point in summary:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 0.3 * inch))
        return elements
    
    def _generate_ai_parameter(self, param: Dict, cluster_color) -> list:
        """Generate parameter section from AI-written content"""
        elements = []
        
        name = param.get('name', 'Unknown')
        score = param.get('score', 0)
        score_color = self.colors.get_score_color(score)
        
        # Parameter header
        param_header = f"<b>{name}</b> - <font color='#{score_color.hexval()}'>{score:.1f}/100</font>"
        elements.append(Paragraph(param_header,
            ParagraphStyle('ParamHeader', fontSize=12, leftIndent=10, spaceAfter=6, fontName='Helvetica-Bold')))
        
        # Findings
        findings = param.get('findings', [])
        if findings:
            for finding in findings:
                elements.append(Paragraph(f"• {finding}",
                    ParagraphStyle('Finding', fontSize=10, leftIndent=25, spaceAfter=4)))
        
        # Strengths (Green)
        strengths = param.get('strengths', [])
        if strengths:
            for strength in strengths:
                elements.append(Paragraph(f"  ✓ {strength}",
                    ParagraphStyle('Strength', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.PROS_TEXT)))
        
        # Weaknesses (Red)
        weaknesses = param.get('weaknesses', [])
        if weaknesses:
            for weakness in weaknesses:
                elements.append(Paragraph(f"  ✗ {weakness}",
                    ParagraphStyle('Weakness', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.CONS_TEXT)))
        
        # Recommendations
        recommendations = param.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                elements.append(Paragraph(f"  → {rec}",
                    ParagraphStyle('Recommendation', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.TEXT_SECONDARY)))
        
        elements.append(Spacer(1, 0.12 * inch))
        return elements
    
    def _generate_market_analysis(self, data: Dict) -> list:
        """Generate TAM/SAM/SOM market analysis section"""
        elements = []
        
        elements.append(Paragraph("Market Size Analysis (TAM/SAM/SOM)", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        market_analysis = data.get('ai_report', {}).get('market_analysis', {})
        
        if not market_analysis:
            elements.append(Paragraph("Market analysis data not available.", self.styles['BodyText']))
            elements.append(PageBreak())
            return elements
        
        # TAM Section
        tam = market_analysis.get('tam', {})
        if tam:
            elements.append(Paragraph("<b>📊 TAM (Total Addressable Market)</b>", self.styles['SubHeader']))
            elements.append(Paragraph(f"<b>Definition:</b> {tam.get('definition', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Estimated Size:</b> {tam.get('size', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Growth Rate:</b> {tam.get('growth_rate', 'N/A')}", self.styles['BodyText']))
            
            if tam.get('assumptions'):
                elements.append(Paragraph("<b>Key Assumptions:</b>", self.styles['BodyText']))
                for assumption in tam['assumptions']:
                    elements.append(Paragraph(f"• {assumption}", self.styles['BulletPoint']))
            
            if tam.get('trends'):
                elements.append(Paragraph("<b>Market Trends:</b>", self.styles['BodyText']))
                for trend in tam['trends']:
                    elements.append(Paragraph(f"• {trend}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        # SAM Section
        sam = market_analysis.get('sam', {})
        if sam:
            elements.append(Paragraph("<b>🎯 SAM (Serviceable Available Market)</b>", self.styles['SubHeader']))
            elements.append(Paragraph(f"<b>Definition:</b> {sam.get('definition', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Estimated Size:</b> {sam.get('size', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Geographic Focus:</b> {sam.get('geographic_focus', 'N/A')}", self.styles['BodyText']))
            
            if sam.get('accessibility_factors'):
                elements.append(Paragraph("<b>Accessibility Factors:</b>", self.styles['BodyText']))
                for factor in sam['accessibility_factors']:
                    elements.append(Paragraph(f"• {factor}", self.styles['BulletPoint']))
            
            if sam.get('demographics'):
                elements.append(Paragraph("<b>Target Demographics:</b>", self.styles['BodyText']))
                for demo in sam['demographics']:
                    elements.append(Paragraph(f"• {demo}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        # SOM Section
        som = market_analysis.get('som', {})
        if som:
            elements.append(Paragraph("<b>🚀 SOM (Serviceable Obtainable Market)</b>", self.styles['SubHeader']))
            elements.append(Paragraph(f"<b>Definition:</b> {som.get('definition', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Estimated Size:</b> {som.get('size', 'N/A')}", self.styles['BodyText']))
            elements.append(Paragraph(f"<b>Target Market Share:</b> {som.get('market_share', 'N/A')}", self.styles['BodyText']))
            
            if som.get('competitive_landscape'):
                elements.append(Paragraph("<b>Competitive Landscape:</b>", self.styles['BodyText']))
                for consideration in som['competitive_landscape']:
                    elements.append(Paragraph(f"• {consideration}", self.styles['BulletPoint']))
            
            if som.get('capture_strategy'):
                elements.append(Paragraph("<b>Market Capture Strategy:</b>", self.styles['BodyText']))
                for strategy in som['capture_strategy']:
                    elements.append(Paragraph(f"• {strategy}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        # Opportunity Summary
        opportunity_summary = market_analysis.get('opportunity_summary', [])
        if opportunity_summary:
            elements.append(Paragraph("<b>💡 Market Opportunity Summary</b>", self.styles['SubHeader']))
            for point in opportunity_summary:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_trl_analysis(self, data: Dict) -> list:
        """Generate TRL (Technology Readiness Level) analysis section"""
        elements = []
        
        elements.append(Paragraph("Technology Readiness Level (TRL) Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        trl_analysis = data.get('ai_report', {}).get('trl_analysis', {})
        
        if not trl_analysis:
            elements.append(Paragraph("TRL analysis data not available.", self.styles['BodyText']))
            elements.append(PageBreak())
            return elements
        
        # Current TRL
        current_trl = trl_analysis.get('current_trl', {})
        if current_trl:
            trl_level = current_trl.get('level', 'N/A')
            elements.append(Paragraph(f"<b>Current TRL Level: {trl_level}/9</b>", self.styles['SubHeader']))
            
            if current_trl.get('justification'):
                elements.append(Paragraph("<b>Justification:</b>", self.styles['BodyText']))
                for justification in current_trl['justification']:
                    elements.append(Paragraph(f"• {justification}", self.styles['BulletPoint']))
            
            if current_trl.get('components_status'):
                elements.append(Paragraph("<b>Technology Components Status:</b>", self.styles['BodyText']))
                for status in current_trl['components_status']:
                    elements.append(Paragraph(f"• {status}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        # TRL Timeline
        timeline = trl_analysis.get('timeline', {})
        if timeline:
            elements.append(Paragraph("<b>⏱️ TRL Progression Timeline</b>", self.styles['SubHeader']))
            
            # TRL 1-3
            trl_1_3 = timeline.get('trl_1_3', {})
            if trl_1_3:
                elements.append(Paragraph(f"<b>TRL 1-3 (Concept Development):</b> {trl_1_3.get('timeframe', 'N/A')}", self.styles['BodyText']))
                if trl_1_3.get('milestones'):
                    for milestone in trl_1_3['milestones']:
                        elements.append(Paragraph(f"  • {milestone}", self.styles['BulletPoint']))
            
            # TRL 4-6
            trl_4_6 = timeline.get('trl_4_6', {})
            if trl_4_6:
                elements.append(Paragraph(f"<b>TRL 4-6 (Validation & Testing):</b> {trl_4_6.get('timeframe', 'N/A')}", self.styles['BodyText']))
                if trl_4_6.get('milestones'):
                    for milestone in trl_4_6['milestones']:
                        elements.append(Paragraph(f"  • {milestone}", self.styles['BulletPoint']))
            
            # TRL 7-9
            trl_7_9 = timeline.get('trl_7_9', {})
            if trl_7_9:
                elements.append(Paragraph(f"<b>TRL 7-9 (Market Readiness):</b> {trl_7_9.get('timeframe', 'N/A')}", self.styles['BodyText']))
                if trl_7_9.get('milestones'):
                    for milestone in trl_7_9['milestones']:
                        elements.append(Paragraph(f"  • {milestone}", self.styles['BulletPoint']))
            
            time_to_market = timeline.get('time_to_market', 'N/A')
            elements.append(Paragraph(f"<b>Estimated Time to Market:</b> {time_to_market}", self.styles['BodyText']))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Risks
        risks = trl_analysis.get('risks', [])
        if risks:
            elements.append(Paragraph("<b>⚠️ Technology Risks & Challenges</b>", self.styles['SubHeader']))
            for risk in risks:
                elements.append(Paragraph(f"• {risk}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Strengths
        strengths = trl_analysis.get('strengths', [])
        if strengths:
            elements.append(Paragraph("<b>✅ Technology Strengths</b>", self.styles['SubHeader']))
            for strength in strengths:
                elements.append(Paragraph(f"• {strength}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Recommendations
        recommendations = trl_analysis.get('recommendations', [])
        if recommendations:
            elements.append(Paragraph("<b>🎯 Recommendations for TRL Advancement</b>", self.styles['SubHeader']))
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", self.styles['BulletPoint']))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_pros_cons(self, data: Dict) -> list:
        """Generate pros and cons analysis section"""
        elements = []
        
        elements.append(Paragraph("Comprehensive Pros and Cons Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        pros_cons = data.get('ai_report', {}).get('pros_cons', {})
        
        if not pros_cons:
            elements.append(Paragraph("Pros and cons analysis not available.", self.styles['BodyText']))
            elements.append(PageBreak())
            return elements
        
        # Advantages (Pros)
        advantages = pros_cons.get('advantages', [])
        if advantages:
            elements.append(Paragraph("<b>✅ Major Advantages</b>", self.styles['SubHeader']))
            
            advantage_bullets = []
            for adv in advantages:
                advantage_bullets.append(Paragraph(f"• {adv}", self.styles['BulletPoint']))
            
            advantage_table = Table([[advantage_bullets]], colWidths=[6.5 * inch])
            advantage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.PROS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.PROS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(advantage_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Disadvantages (Cons)
        disadvantages = pros_cons.get('disadvantages', [])
        if disadvantages:
            elements.append(Paragraph("<b>❌ Key Disadvantages</b>", self.styles['SubHeader']))
            
            disadvantage_bullets = []
            for dis in disadvantages:
                disadvantage_bullets.append(Paragraph(f"• {dis}", self.styles['BulletPoint']))
            
            disadvantage_table = Table([[disadvantage_bullets]], colWidths=[6.5 * inch])
            disadvantage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.CONS_BG),
                ('BOX', (0, 0), (-1, -1), 2, self.colors.CONS_BORDER),
                ('ROUNDEDCORNERS', [8, 8, 8, 8]),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(disadvantage_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Balanced Assessment
        balanced = pros_cons.get('balanced_assessment', [])
        if balanced:
            elements.append(Paragraph("<b>⚖️ Balanced Assessment</b>", self.styles['SubHeader']))
            for assessment in balanced:
                elements.append(Paragraph(f"• {assessment}", self.styles['BulletPoint']))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_weaknesses_analysis(self, data: Dict) -> list:
        """Generate detailed weaknesses analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Weaknesses Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        weaknesses_analysis = data.get('ai_report', {}).get('weaknesses_analysis', {})
        
        if not weaknesses_analysis:
            elements.append(Paragraph("Weaknesses analysis not available.", self.styles['BodyText']))
            elements.append(PageBreak())
            return elements
        
        # Critical Weaknesses
        critical = weaknesses_analysis.get('critical', [])
        if critical:
            elements.append(Paragraph("<b>🔴 Critical Weaknesses (Score < 40/100)</b>", self.styles['SubHeader']))
            for weakness_item in critical:
                weakness = weakness_item.get('weakness', '')
                impact = weakness_item.get('impact', '')
                action = weakness_item.get('action', '')
                elements.append(Paragraph(f"<b>• {weakness}</b>", self.styles['BulletPoint']))
                if impact:
                    elements.append(Paragraph(f"  <i>Impact:</i> {impact}", 
                        ParagraphStyle('Impact', fontSize=9, leftIndent=25, textColor=self.colors.TEXT_SECONDARY)))
                if action:
                    elements.append(Paragraph(f"  <i>Action Required:</i> {action}", 
                        ParagraphStyle('Action', fontSize=9, leftIndent=25, textColor=self.colors.CONS_BORDER)))
            elements.append(Spacer(1, 0.2 * inch))
        
        # High Priority Weaknesses
        high_priority = weaknesses_analysis.get('high_priority', [])
        if high_priority:
            elements.append(Paragraph("<b>🟠 High Priority Weaknesses (Score 40-50/100)</b>", self.styles['SubHeader']))
            for weakness_item in high_priority:
                weakness = weakness_item.get('weakness', '')
                impact = weakness_item.get('impact', '')
                action = weakness_item.get('action', '')
                elements.append(Paragraph(f"<b>• {weakness}</b>", self.styles['BulletPoint']))
                if impact:
                    elements.append(Paragraph(f"  <i>Impact:</i> {impact}", 
                        ParagraphStyle('Impact', fontSize=9, leftIndent=25, textColor=self.colors.TEXT_SECONDARY)))
                if action:
                    elements.append(Paragraph(f"  <i>Action Required:</i> {action}", 
                        ParagraphStyle('Action', fontSize=9, leftIndent=25, textColor=self.colors.CONS_BORDER)))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Moderate Weaknesses
        moderate = weaknesses_analysis.get('moderate', [])
        if moderate:
            elements.append(Paragraph("<b>🟡 Moderate Weaknesses (Score 50-60/100)</b>", self.styles['SubHeader']))
            for weakness_item in moderate:
                weakness = weakness_item.get('weakness', '')
                impact = weakness_item.get('impact', '')
                action = weakness_item.get('action', '')
                elements.append(Paragraph(f"<b>• {weakness}</b>", self.styles['BulletPoint']))
                if impact:
                    elements.append(Paragraph(f"  <i>Impact:</i> {impact}", 
                        ParagraphStyle('Impact', fontSize=9, leftIndent=25, textColor=self.colors.TEXT_SECONDARY)))
                if action:
                    elements.append(Paragraph(f"  <i>Action Required:</i> {action}", 
                        ParagraphStyle('Action', fontSize=9, leftIndent=25, textColor=self.colors.CONS_BORDER)))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Weakness Patterns
        patterns = weaknesses_analysis.get('patterns', [])
        if patterns:
            elements.append(Paragraph("<b>📊 Weakness Patterns</b>", self.styles['SubHeader']))
            for pattern in patterns:
                elements.append(Paragraph(f"• {pattern}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Remediation Strategy
        remediation = weaknesses_analysis.get('remediation_strategy', [])
        if remediation:
            elements.append(Paragraph("<b>🔧 Remediation Strategy</b>", self.styles['SubHeader']))
            for strategy in remediation:
                elements.append(Paragraph(f"• {strategy}", self.styles['BulletPoint']))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_single_cluster(self, cluster_name: str, summary: Dict) -> list:
        """Generate analysis for a single cluster with ALL parameters"""
        elements = []
        
        cluster_color = self.colors.get_cluster_color(cluster_name)
        score = summary['overall_score']
        status = summary['status']
        
        # Cluster Header
        header_content = [[
            Paragraph(f"<b>{cluster_name}</b>",
                    ParagraphStyle('ClusterTitle', fontSize=16, textColor=cluster_color,
                                 fontName='Helvetica-Bold')),
            Paragraph(f"<b>{score:.1f}/100</b> - {status}",
                    ParagraphStyle('ClusterScore', fontSize=14, textColor=cluster_color,
                                 alignment=TA_CENTER, fontName='Helvetica-Bold'))
        ]]
        
        header_table = Table(header_content, colWidths=[4.5 * inch, 2 * inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
            ('BOX', (0, 0), (-1, -1), 3, cluster_color),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.15 * inch))
        
        # Parameters with bullet points
        for param in summary['parameters']:
            elements.extend(self._generate_parameter_details(param))
        
        elements.append(Spacer(1, 0.25 * inch))
        return elements
    
    def _generate_parameter_details(self, param: Dict) -> list:
        """Generate detailed bullet points for a parameter"""
        elements = []
        
        score = param['score']
        score_color = self.colors.get_score_color(score)
        
        # Parameter name with score
        param_header = f"<b>{param['sub_parameter']}</b> - <font color='#{score_color.hexval()}'>{score:.1f}/100</font>"
        elements.append(Paragraph(param_header,
            ParagraphStyle('ParamHeader', fontSize=12, leftIndent=10, spaceAfter=6, fontName='Helvetica-Bold')))
        
        # Explanation
        if param['explanation']:
            elements.append(Paragraph(f"• {param['explanation']}",
                ParagraphStyle('Explanation', fontSize=10, leftIndent=20, spaceAfter=4)))
        
        # Strengths (if high score)
        if score >= 70 and param['strengths']:
            for strength in param['strengths'][:3]:
                elements.append(Paragraph(f"  ✓ {strength}",
                    ParagraphStyle('Strength', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.PROS_TEXT)))
        
        # Weaknesses (if low score)
        if score < 60 and param['weaknesses']:
            for weakness in param['weaknesses'][:3]:
                elements.append(Paragraph(f"  ✗ {weakness}",
                    ParagraphStyle('Weakness', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.CONS_TEXT)))
        
        # Key insights
        if param['key_insights']:
            for insight in param['key_insights'][:2]:
                elements.append(Paragraph(f"  → {insight}",
                    ParagraphStyle('Insight', fontSize=9, leftIndent=30, spaceAfter=3,
                                 textColor=self.colors.TEXT_SECONDARY)))
        
        elements.append(Spacer(1, 0.1 * inch))
        return elements
    
    def _generate_recommendations(self, data: Dict) -> list:
        """Generate recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommendations & Action Items", self.styles['SectionHeader']))
        
        recommendations = data['recommendations'][:10]  # Top 10
        
        if recommendations:
            # Group by priority
            high_priority = [r for r in recommendations if r['priority'] == 'High']
            medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
            
            if high_priority:
                elements.append(Paragraph("<b>🔴 High Priority Actions:</b>", self.styles['SubHeader']))
                for rec in high_priority:
                    elements.append(Paragraph(
                        f"• {rec['text']}",
                        self.styles['BulletPoint']
                    ))
                elements.append(Spacer(1, 0.15 * inch))
            
            if medium_priority:
                elements.append(Paragraph("<b>🟡 Medium Priority Actions:</b>", self.styles['SubHeader']))
                for rec in medium_priority[:5]:
                    elements.append(Paragraph(
                        f"• {rec['text']}",
                        self.styles['BulletPoint']
                    ))
        
        elements.append(PageBreak())
        return elements
    
    def _generate_conclusion(self, data: Dict) -> list:
        """Generate conclusion section using AI-written content"""
        elements = []
        
        elements.append(Paragraph("Conclusion & Final Verdict", self.styles['SectionHeader']))
        
        # Get AI-written conclusion
        ai_conclusion = data.get('ai_report', {}).get('conclusion', {})
        score = data['metadata']['overall_score']
        
        # Investment Decision
        decision = ai_conclusion.get('investment_decision', '')
        verdict_color = self.colors.get_score_color(score)
        
        if decision:
            verdict_elements = []
            verdict_elements.append(Paragraph(f"<b>{decision}</b>",
                    ParagraphStyle('Decision', fontSize=16, textColor=verdict_color,
                                 fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
            
            # Final Verdict points
            verdict_points = ai_conclusion.get('final_verdict', [])
            for point in verdict_points:
                verdict_elements.append(Paragraph(f"• {point}",
                    ParagraphStyle('VerdictPoint', fontSize=10, leftIndent=15, spaceAfter=4)))
            
            verdict_table = Table([[verdict_elements]], colWidths=[6.5 * inch])
            verdict_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors.BACKGROUND),
                ('BOX', (0, 0), (-1, -1), 3, verdict_color),
                ('ROUNDEDCORNERS', [12, 12, 12, 12]),
                ('LEFTPADDING', (0, 0), (-1, -1), 20),
                ('RIGHTPADDING', (0, 0), (-1, -1), 20),
                ('TOPPADDING', (0, 0), (-1, -1), 18),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
            ]))
            elements.append(verdict_table)
            elements.append(Spacer(1, 0.25 * inch))
        
        # Path Forward
        path_forward = ai_conclusion.get('path_forward', [])
        if path_forward:
            elements.append(Paragraph("<b>📍 Path Forward</b>", self.styles['SubHeader']))
            for point in path_forward:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Success Factors
        success_factors = ai_conclusion.get('success_factors', [])
        if success_factors:
            elements.append(Paragraph("<b>🎯 Success Factors</b>", self.styles['SubHeader']))
            for point in success_factors:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Risk Mitigation
        risk_mitigation = ai_conclusion.get('risk_mitigation', [])
        if risk_mitigation:
            elements.append(Paragraph("<b>🛡️ Risk Mitigation</b>", self.styles['SubHeader']))
            for point in risk_mitigation:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.15 * inch))
        
        # Market Opportunity Assessment
        market_assessment = ai_conclusion.get('market_assessment', [])
        if market_assessment:
            elements.append(Paragraph("<b>📈 Market Opportunity Assessment</b>", self.styles['SubHeader']))
            for point in market_assessment:
                elements.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 0.3 * inch))
        
        # Footer
        elements.append(Paragraph(
            f"This comprehensive analysis was conducted by Pragati AI Engine using {data['metadata']['total_agents']} specialized validation agents. "
            "All assessments are based on expert AI agent discussions and comprehensive analysis.",
            ParagraphStyle('Footer', fontSize=9, textColor=self.colors.TEXT_LIGHT,
                         alignment=TA_CENTER)
        ))
        
        return elements


def _generate_pdf_from_detailed_analysis(report_data: Dict[str, Any]) -> BytesIO:
    """
    Fallback: Generate PDF from detailed_analysis (for old reports)
    Without AI writing step
    """
    logger.info("📄 Generating PDF from detailed_analysis (legacy format)")
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        elements = []
        colors = ReportColors()
        styles = getSampleStyleSheet()
        
        # Title
        title = report_data.get('title', 'Validation Report')
        score = report_data.get('overall_score', 0)
        
        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph(f"<b>{title}</b>",
            ParagraphStyle('Title', fontSize=28, alignment=TA_CENTER, 
                         textColor=colors.PRIMARY, fontName='Helvetica-Bold', spaceAfter=20)))
        
        elements.append(Paragraph(f"Score: {score:.1f}/100",
            ParagraphStyle('Score', fontSize=24, alignment=TA_CENTER, 
                         textColor=colors.get_score_color(score), spaceAfter=40)))
        
        elements.append(PageBreak())
        
        # Get detailed analysis
        detailed = report_data.get('detailed_analysis', {})
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary",
            ParagraphStyle('Header', fontSize=20, fontName='Helvetica-Bold', 
                         textColor=colors.PRIMARY, spaceAfter=15)))
        
        exec_summary = detailed.get('executive_summary', {})
        summary_points = exec_summary.get('summary_points', [])
        for point in summary_points:
            elements.append(Paragraph(f"• {point}",
                ParagraphStyle('Bullet', fontSize=11, leftIndent=20, spaceAfter=6)))
        
        elements.append(Spacer(1, 0.3 * inch))
        
        # Performance Analysis
        perf = detailed.get('performance_analysis', {})
        
        # Good Areas
        good_areas = perf.get('good_areas', [])
        if good_areas:
            elements.append(Paragraph("✅ Strengths",
                ParagraphStyle('SubHeader', fontSize=14, fontName='Helvetica-Bold', 
                             textColor=colors.GOOD, spaceAfter=10)))
            for area in good_areas[:5]:
                elements.append(Paragraph(f"• {area.get('cluster')}: {area.get('reason')}",
                    ParagraphStyle('Bullet', fontSize=10, leftIndent=20, spaceAfter=5)))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Bad Areas
        bad_areas = perf.get('bad_areas', [])
        if bad_areas:
            elements.append(Paragraph("⚠️ Areas for Improvement",
                ParagraphStyle('SubHeader', fontSize=14, fontName='Helvetica-Bold', 
                             textColor=colors.WEAK, spaceAfter=10)))
            for area in bad_areas[:5]:
                elements.append(Paragraph(f"• {area.get('cluster')}: {area.get('reason')}",
                    ParagraphStyle('Bullet', fontSize=10, leftIndent=20, spaceAfter=5)))
        
        elements.append(PageBreak())
        
        # Cluster Analyses
        cluster_analyses = detailed.get('cluster_analyses', {})
        if cluster_analyses:
            elements.append(Paragraph("Detailed Analysis",
                ParagraphStyle('Header', fontSize=20, fontName='Helvetica-Bold', 
                             textColor=colors.PRIMARY, spaceAfter=15)))
            
            for cluster_name, cluster_data in cluster_analyses.items():
                cluster_color = colors.get_cluster_color(cluster_name)
                cluster_score = cluster_data.get('score', 0)
                
                elements.append(Paragraph(f"<b>{cluster_name}</b> ({cluster_score:.1f}/100)",
                    ParagraphStyle('ClusterHeader', fontSize=14, fontName='Helvetica-Bold',
                                 textColor=cluster_color, spaceAfter=10)))
                
                # Summary points
                for point in cluster_data.get('summary_points', []):
                    elements.append(Paragraph(f"• {point}",
                        ParagraphStyle('Bullet', fontSize=10, leftIndent=20, spaceAfter=4)))
                
                elements.append(Spacer(1, 0.15 * inch))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        logger.info("✅ Fallback PDF generated successfully")
        return buffer
        
    except Exception as e:
        logger.error(f"Fallback PDF generation failed: {e}")
        return _create_error_pdf(f"PDF generation error: {str(e)}")


def _create_error_pdf(error_message: str) -> BytesIO:
    """Create a simple error PDF"""
    buffer = BytesIO()
    
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "PDF Generation Error")
    c.drawString(100, 730, error_message[:100])
    c.drawString(100, 710, "Please contact support for assistance.")
    c.save()
    
    buffer.seek(0)
    return buffer

