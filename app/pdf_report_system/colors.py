"""
Color scheme for PDF reports
"""

from reportlab.lib import colors


class ReportColors:
    """Professional color palette for validation reports"""
    
    # Primary Brand Colors
    PRIMARY = colors.HexColor('#2563EB')  # Blue
    SECONDARY = colors.HexColor('#7C3AED')  # Purple
    ACCENT = colors.HexColor('#10B981')  # Green
    
    # Score-based Colors (100-point scale)
    EXCELLENT = colors.HexColor('#059669')  # 80-100: Dark green
    GOOD = colors.HexColor('#10B981')  # 60-79: Green
    MODERATE = colors.HexColor('#F59E0B')  # 40-59: Amber
    WEAK = colors.HexColor('#EF4444')  # 20-39: Red
    POOR = colors.HexColor('#DC2626')  # 0-19: Dark red
    
    # Pros/Cons Colors
    PROS_BG = colors.HexColor('#ECFDF5')  # Light green background
    PROS_BORDER = colors.HexColor('#10B981')  # Green border
    PROS_TEXT = colors.HexColor('#047857')  # Dark green text
    
    CONS_BG = colors.HexColor('#FEF2F2')  # Light red background
    CONS_BORDER = colors.HexColor('#EF4444')  # Red border
    CONS_TEXT = colors.HexColor('#B91C1C')  # Dark red text
    
    # Neutral UI Colors
    BACKGROUND = colors.HexColor('#F9FAFB')
    BACKGROUND_DARK = colors.HexColor('#F3F4F6')
    BORDER_LIGHT = colors.HexColor('#E5E7EB')
    BORDER = colors.HexColor('#D1D5DB')
    TEXT_PRIMARY = colors.HexColor('#111827')
    TEXT_SECONDARY = colors.HexColor('#6B7280')
    TEXT_LIGHT = colors.HexColor('#9CA3AF')
    WHITE = colors.white
    
    # Cluster-specific Colors
    CLUSTER_COLORS = {
        'Core Idea': colors.HexColor('#3B82F6'),  # Blue
        'Market Opportunity': colors.HexColor('#10B981'),  # Green
        'Execution': colors.HexColor('#8B5CF6'),  # Purple
        'Business Model': colors.HexColor('#F59E0B'),  # Amber
        'Team': colors.HexColor('#EC4899'),  # Pink
        'Compliance': colors.HexColor('#06B6D4'),  # Cyan
        'Risk & Strategy': colors.HexColor('#EF4444')  # Red
    }
    
    @classmethod
    def get_score_color(cls, score: float) -> colors.Color:
        """Get color based on score (0-100 scale)"""
        if score >= 80:
            return cls.EXCELLENT
        elif score >= 60:
            return cls.GOOD
        elif score >= 40:
            return cls.MODERATE
        elif score >= 20:
            return cls.WEAK
        else:
            return cls.POOR
    
    @classmethod
    def get_cluster_color(cls, cluster_name: str) -> colors.Color:
        """Get color for a specific cluster"""
        return cls.CLUSTER_COLORS.get(cluster_name, cls.PRIMARY)

