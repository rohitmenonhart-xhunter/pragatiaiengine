"""
User Profile Manager
Creates and manages entrepreneur profiles based on psychometric assessments
Integrates profiles with idea validation for personalized recommendations
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from database_manager import get_database_manager

logger = logging.getLogger(__name__)


class UserProfileManager:
    """
    Manages user profiles created from psychometric assessments
    Provides profile-aware validation recommendations
    """
    
    def __init__(self):
        """Initialize the user profile manager"""
        self.db_manager = get_database_manager()
        if not self.db_manager:
            logger.warning("⚠️ Database manager not available - profiles will not be persisted")
        logger.info("✅ User Profile Manager initialized")
    
    def create_profile_from_psychometric(
        self,
        user_id: str,
        evaluation_result: Dict
    ) -> Dict:
        """
        Create or update user profile from psychometric evaluation
        
        Args:
            user_id: Unique user identifier
            evaluation_result: Complete psychometric evaluation result
            
        Returns:
            User profile dictionary
        """
        try:
            logger.info(f"📊 Creating profile for user: {user_id}")
            
            # Extract key information from evaluation
            dimension_scores = evaluation_result.get('dimension_scores', {})
            entrepreneurial_fit = evaluation_result.get('entrepreneurial_fit', {})
            strengths = evaluation_result.get('strengths', [])
            weaknesses = evaluation_result.get('areas_for_development', [])
            personality_profile = evaluation_result.get('personality_profile', '')
            
            # Calculate profile metadata
            profile = {
                "user_id": user_id,
                "user_name": evaluation_result.get('user_name', 'Unknown'),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                
                # Psychometric scores
                "psychometric_scores": dimension_scores,
                "overall_psychometric_score": evaluation_result.get('overall_score', 0),
                
                # Entrepreneurial assessment
                "entrepreneurial_fit": entrepreneurial_fit.get('overall_fit', 'Medium'),
                "fit_score": entrepreneurial_fit.get('fit_score', 50),
                "ideal_role": entrepreneurial_fit.get('ideal_role', 'Entrepreneur'),
                "ideal_venture_type": entrepreneurial_fit.get('ideal_venture_type', 'Various'),
                
                # Strengths and weaknesses
                "top_strengths": strengths[:5],  # Top 5 strengths
                "development_areas": weaknesses[:5],  # Top 5 areas
                
                # Profile summary
                "personality_profile": personality_profile,
                
                # Validation preferences (based on profile)
                "validation_focus_areas": self._determine_focus_areas(dimension_scores),
                "risk_tolerance_level": self._categorize_risk_tolerance(
                    dimension_scores.get('risk_tolerance', 5)
                ),
                
                # Recommendations context
                "detailed_insights": evaluation_result.get('detailed_insights', {}),
                "recommendations": evaluation_result.get('recommendations', []),
                
                # Profile status
                "profile_completeness": self._calculate_completeness(evaluation_result),
                "profile_version": "1.0",
                
                # Reference to evaluation
                "evaluation_id": evaluation_result.get('evaluation_id'),
                "assessment_date": evaluation_result.get('evaluated_at')
            }
            
            # Save to database
            if self.db_manager is not None and self.db_manager.db is not None:
                try:
                    # Check if profile exists
                    existing_profile = self.db_manager.db.user_profiles.find_one(
                        {"user_id": user_id}
                    )
                    
                    if existing_profile:
                        # Update existing profile
                        profile['created_at'] = existing_profile.get('created_at')
                        profile['validation_history'] = existing_profile.get('validation_history', [])
                        
                        self.db_manager.db.user_profiles.update_one(
                            {"user_id": user_id},
                            {"$set": profile}
                        )
                        logger.info(f"✅ Updated existing profile for user: {user_id}")
                    else:
                        # Create new profile
                        profile['validation_history'] = []
                        self.db_manager.db.user_profiles.insert_one(profile)
                        logger.info(f"✅ Created new profile for user: {user_id}")
                
                except Exception as e:
                    logger.error(f"❌ Failed to save profile to database: {e}")
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to create profile: {e}")
            raise
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve user profile
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User profile or None if not found
        """
        try:
            if self.db_manager is None or self.db_manager.db is None:
                return None
            
            profile = self.db_manager.db.user_profiles.find_one({"user_id": user_id})
            
            if profile:
                profile['_id'] = str(profile['_id'])
                logger.info(f"✅ Retrieved profile for user: {user_id}")
                return profile
            else:
                logger.info(f"ℹ️ No profile found for user: {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve profile: {e}")
            return None
    
    def add_validation_to_history(
        self,
        user_id: str,
        idea_name: str,
        validation_result: Dict,
        report_id: str
    ):
        """
        Add validation result to user's history
        
        Args:
            user_id: Unique user identifier
            idea_name: Name of the validated idea
            validation_result: Validation result
            report_id: Report ID in database
        """
        try:
            if self.db_manager is None or self.db_manager.db is None:
                return
            
            validation_entry = {
                "idea_name": idea_name,
                "report_id": report_id,
                "validated_at": datetime.now().isoformat(),
                "overall_score": validation_result.get('overall_score', 0),
                "validation_outcome": validation_result.get('validation_outcome', 'N/A')
            }
            
            self.db_manager.db.user_profiles.update_one(
                {"user_id": user_id},
                {
                    "$push": {"validation_history": validation_entry},
                    "$set": {"last_updated": datetime.now().isoformat()}
                }
            )
            
            logger.info(f"✅ Added validation to history for user: {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add validation to history: {e}")
    
    def get_personalized_validation_context(self, user_id: str) -> Dict:
        """
        Get personalized context for idea validation based on user profile
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Context dictionary for validation customization
        """
        try:
            profile = self.get_profile(user_id)
            
            if not profile:
                return {
                    "has_profile": False,
                    "message": "No psychometric profile available. Standard validation will be performed."
                }
            
            # Build personalized context
            context = {
                "has_profile": True,
                "user_name": profile.get('user_name', 'User'),
                
                # Entrepreneur characteristics
                "entrepreneurial_fit": profile.get('entrepreneurial_fit'),
                "fit_score": profile.get('fit_score', 50),
                "ideal_role": profile.get('ideal_role'),
                
                # Key strengths to leverage
                "strengths": profile.get('top_strengths', []),
                
                # Areas needing extra attention
                "weak_areas": profile.get('development_areas', []),
                
                # Validation customization
                "focus_areas": profile.get('validation_focus_areas', []),
                "risk_tolerance": profile.get('risk_tolerance_level', 'Medium'),
                
                # Dimension scores for weighted validation
                "psychometric_scores": profile.get('psychometric_scores', {}),
                
                # Personality insights
                "personality_summary": profile.get('personality_profile', ''),
                "leadership_style": profile.get('detailed_insights', {}).get('leadership_style', ''),
                "decision_making_pattern": profile.get('detailed_insights', {}).get('decision_making_pattern', ''),
                
                # Previous validations
                "validation_count": len(profile.get('validation_history', [])),
                "previous_ideas": [v.get('idea_name') for v in profile.get('validation_history', [])[-3:]]
            }
            
            logger.info(f"✅ Generated personalized context for user: {user_id}")
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to generate personalized context: {e}")
            return {
                "has_profile": False,
                "error": str(e)
            }
    
    def _determine_focus_areas(self, dimension_scores: Dict) -> List[str]:
        """
        Determine which validation areas need extra focus based on weak dimensions
        
        Args:
            dimension_scores: Dictionary of dimension scores
            
        Returns:
            List of focus areas for validation
        """
        focus_areas = []
        
        # Map dimensions to validation focus areas
        dimension_to_focus = {
            'leadership': 'Team & Leadership Evaluation',
            'risk_tolerance': 'Risk Assessment & Mitigation',
            'resilience': 'Sustainability & Long-term Viability',
            'innovation': 'Innovation & Differentiation',
            'decision_making': 'Business Model & Strategy',
            'emotional_intelligence': 'Stakeholder Management',
            'persistence': 'Execution Capability',
            'strategic_thinking': 'Market Strategy & Positioning',
            'communication': 'Go-to-Market & Sales',
            'problem_solving': 'Problem-Solution Fit'
        }
        
        # Identify weak dimensions (score < 5)
        for dimension, score in dimension_scores.items():
            if score < 5 and dimension in dimension_to_focus:
                focus_areas.append(dimension_to_focus[dimension])
        
        # If no weak areas, focus on top performers for leverage
        if not focus_areas:
            sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
            for dimension, _ in sorted_dims[:3]:
                if dimension in dimension_to_focus:
                    focus_areas.append(dimension_to_focus[dimension])
        
        return focus_areas
    
    def _categorize_risk_tolerance(self, risk_score: float) -> str:
        """
        Categorize risk tolerance level
        
        Args:
            risk_score: Risk tolerance score (0-10)
            
        Returns:
            Category: Low, Medium, High, Very High
        """
        if risk_score >= 8:
            return "Very High"
        elif risk_score >= 6:
            return "High"
        elif risk_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_completeness(self, evaluation_result: Dict) -> float:
        """
        Calculate profile completeness percentage
        
        Args:
            evaluation_result: Evaluation result
            
        Returns:
            Completeness percentage (0-100)
        """
        total_fields = 0
        filled_fields = 0
        
        # Check key fields
        fields_to_check = [
            'dimension_scores',
            'entrepreneurial_fit',
            'strengths',
            'areas_for_development',
            'personality_profile',
            'detailed_insights',
            'recommendations'
        ]
        
        for field in fields_to_check:
            total_fields += 1
            value = evaluation_result.get(field)
            if value:
                if isinstance(value, (list, dict)) and len(value) > 0:
                    filled_fields += 1
                elif isinstance(value, str) and value.strip():
                    filled_fields += 1
        
        completeness = (filled_fields / total_fields) * 100
        return round(completeness, 1)


# Singleton instance storage
_profile_manager_instance: Optional[UserProfileManager] = None


def get_user_profile_manager() -> UserProfileManager:
    """Get singleton instance of user profile manager"""
    global _profile_manager_instance
    if _profile_manager_instance is None:
        _profile_manager_instance = UserProfileManager()
    return _profile_manager_instance

