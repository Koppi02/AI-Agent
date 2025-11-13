"""
Validates predictions against ABS_torveny hierarchy
"""

import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HierarchyValidator:
    """Validates hierarchical relationships in predictions"""
    
    def __init__(self, abs_torveny_df: pd.DataFrame):
        """
        Args:
            abs_torveny_df: DataFrame containing ABS hierarchy rules
        """
        self.abs_torveny_df = abs_torveny_df
        
        # Gyors lookup cache-k
        self._create_lookup_caches()
    
    def _create_lookup_caches(self):
        """Create fast lookup dictionaries"""
        df = self.abs_torveny_df
        
        # (BrandName, BaseBrandName, SegmentName) -> valid combinations
        self.valid_combinations = set()
        for _, row in df.iterrows():
            combo = (
                row['BrandName'].lower(),
                row['BaseBrandName'].lower(),
                row['SegmentName'].lower()
            )
            self.valid_combinations.add(combo)
        
        # BrandName -> possible BaseBrandNames
        self.brand_to_basebrands = df.groupby('BrandName')['BaseBrandName'].apply(set).to_dict()
        
        # BrandName -> possible SegmentNames
        self.brand_to_segments = df.groupby('BrandName')['SegmentName'].apply(set).to_dict()
        
        logger.info(f"? Hierarchy cache created: {len(self.valid_combinations)} valid combinations")
    
    def validate_prediction(self, prediction: Dict[str, str]) -> Dict:
        """
        Validate a prediction against hierarchy rules
        
        Args:
            prediction: Dict with keys 'brand', 'basebrand', 'segment', 'advertiser'
            
        Returns:
            Dict with validation results
        """
        brand = prediction['brand']
        basebrand = prediction['basebrand']
        segment = prediction['segment']
        advertiser = prediction.get('advertiser', '')
        
        # 1. Ellenõrizzük a brand-basebrand-segment hármast
        combo = (brand.lower(), basebrand.lower(), segment.lower())
        is_valid_hierarchy = combo in self.valid_combinations
        
        if not is_valid_hierarchy:
            # Részletes hibaüzenet
            error_msg = self._generate_error_message(brand, basebrand, segment)
            
            return {
                'is_valid': False,
                'error': error_msg,
                'suggested_correction': self._suggest_correction(brand, basebrand, segment)
            }
        
        # 2. Ellenõrizzük az advertisert
        advertiser_exists = self._check_advertiser_exists(brand, advertiser)
        
        return {
            'is_valid': True,
            'hierarchy_valid': True,
            'advertiser_is_new': not advertiser_exists,
            'suggested_entry': self._create_suggested_entry(
                basebrand, brand, advertiser, segment
            ) if not advertiser_exists else None
        }
    
    def _check_advertiser_exists(self, brand: str, advertiser: str) -> bool:
        """Check if advertiser exists for given brand"""
        matches = self.abs_torveny_df[
            (self.abs_torveny_df['BrandName'].str.lower() == brand.lower()) &
            (self.abs_torveny_df['AdvertiserName'].str.lower() == advertiser.lower())
        ]
        return len(matches) > 0
    
    def _generate_error_message(self, brand: str, basebrand: str, segment: str) -> str:
        """Generate detailed error message for invalid hierarchy"""
        # Ellenõrizzük, melyik kapcsolat rossz
        possible_basebrands = self.brand_to_basebrands.get(brand, set())
        possible_segments = self.brand_to_segments.get(brand, set())
        
        if brand not in self.brand_to_basebrands:
            return f"Brand '{brand}' nem található az ABS_torveny-ben"
        
        if basebrand not in [bb.lower() for bb in possible_basebrands]:
            return f"BaseBrand '{basebrand}' nem tartozik '{brand}' alá. Lehetséges: {possible_basebrands}"
        
        if segment not in [s.lower() for s in possible_segments]:
            return f"Segment '{segment}' nem tartozik '{brand}' alá. Lehetséges: {possible_segments}"
        
        return f"Érvénytelen hierarchia: {segment} ? {brand} ? {basebrand}"
    
    def _suggest_correction(self, brand: str, basebrand: str, segment: str) -> Optional[Dict]:
        """Suggest a corrected hierarchy if possible"""
        # Ha a brand valid, de a többi nem, javasoljunk helyes értékeket
        if brand in self.brand_to_basebrands:
            correct_basebrands = list(self.brand_to_basebrands[brand])
            correct_segments = list(self.brand_to_segments[brand])
            
            return {
                'brand': brand,
                'suggested_basebrand': correct_basebrands[0] if correct_basebrands else None,
                'suggested_segment': correct_segments[0] if correct_segments else None
            }
        
        return None
    
    def _create_suggested_entry(
        self,
        basebrand: str,
        brand: str,
        advertiser: str,
        segment: str
    ) -> Dict:
        """Create a suggested CSV entry for new advertiser"""
        return {
            'BaseBrandName': basebrand,
            'BrandName': brand,
            'AdvertiserName': advertiser,
            'SegmentName': segment
        }
