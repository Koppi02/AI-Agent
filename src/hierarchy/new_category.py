"""
Handles new advertiser/category proposals and CSV updates
"""

import pandas as pd
import os
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class NewCategoryHandler:
    """Manages new category proposals and CSV updates"""
    
    def __init__(self, abs_torveny_path: str):
        """
        Args:
            abs_torveny_path: Path to ABS_torveny.csv file
        """
        self.abs_torveny_path = abs_torveny_path
        self.pending_proposals: List[Dict] = []
    
    def propose_new_advertiser(
        self,
        suggested_entry: Dict,
        confidence: float,
        image_path: str,
        ocr_evidence: str
    ) -> Dict:
        """
        Create a proposal for a new advertiser
        
        Args:
            suggested_entry: Dict with BaseBrandName, BrandName, AdvertiserName, SegmentName
            confidence: Confidence score of the prediction
            image_path: Path to the source image
            ocr_evidence: OCR text that led to this proposal
            
        Returns:
            Proposal dictionary
        """
        proposal = {
            **suggested_entry,
            'confidence': confidence,
            'image_path': image_path,
            'ocr_evidence': ocr_evidence[:200],  # First 200 chars
            'proposed_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        self.pending_proposals.append(proposal)
        
        logger.info(f"?? New advertiser proposal: {suggested_entry['AdvertiserName']}")
        logger.info(f"   Brand: {suggested_entry['BrandName']}")
        logger.info(f"   Confidence: {confidence:.2f}")
        
        return proposal
    
    def approve_and_save(
        self,
        proposal: Dict,
        user_approved: bool = True
    ) -> bool:
        """
        Approve a proposal and save to CSV
        
        Args:
            proposal: The proposal dictionary
            user_approved: Whether user manually approved it
            
        Returns:
            Success status
        """
        if not user_approved:
            logger.info(f"? Proposal rejected: {proposal['AdvertiserName']}")
            proposal['status'] = 'rejected'
            return False
        
        try:
            # Load current CSV
            abs_df = pd.read_csv(self.abs_torveny_path, encoding='utf-8', sep=';')
            
            # Create new row
            new_row = pd.DataFrame([{
                'BaseBrandName': proposal['BaseBrandName'],
                'BrandName': proposal['BrandName'],
                'AdvertiserName': proposal['AdvertiserName'],
                'SegmentName': proposal['SegmentName']
            }])
            
            # Append and save
            abs_df = pd.concat([abs_df, new_row], ignore_index=True)
            abs_df.to_csv(self.abs_torveny_path, index=False, encoding='utf-8', sep=';')
            
            proposal['status'] = 'approved'
            logger.info(f"? New advertiser saved: {proposal['AdvertiserName']}")
            
            return True
            
        except Exception as e:
            logger.error(f"? Failed to save new advertiser: {e}")
            proposal['status'] = 'error'
            return False
    
    def get_pending_proposals(self) -> List[Dict]:
        """Get all pending proposals"""
        return [p for p in self.pending_proposals if p['status'] == 'pending']
    
    def export_proposals_to_csv(self, output_path: str):
        """Export all proposals to a CSV file for review"""
        if not self.pending_proposals:
            logger.warning("No proposals to export")
            return
        
        df = pd.DataFrame(self.pending_proposals)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"?? Exported {len(self.pending_proposals)} proposals to {output_path}")
