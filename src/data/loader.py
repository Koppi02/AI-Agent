"""
Data loading utilities for ABS_torveny and labels CSV files
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of CSV data files"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary containing data paths
        """
        self.config = config
        self.abs_torveny_df: Optional[pd.DataFrame] = None
        self.labels_df: Optional[pd.DataFrame] = None
        
        # Hierarchy mappings
        self.basebrand_to_advertiser: Dict = {}
        self.brand_to_basebrand: Dict = {}
        self.segment_to_brand: Dict = {}
        
        # Category mappings
        self.advertiser_to_index: Dict = {}
        self.index_to_advertiser: Dict = {}
        self.basebrand_to_index: Dict = {}
        self.index_to_basebrand: Dict = {}
        self.brand_to_index: Dict = {}
        self.index_to_brand: Dict = {}
        self.segment_to_index: Dict = {}
        self.index_to_segment: Dict = {}
    
    def load_abs_torveny(self) -> pd.DataFrame:
        """Load and process ABS_torveny.csv"""
        abs_path = os.path.join(
            self.config['data']['drive_path'],
            self.config['data']['abs_torveny_csv']
        )
        
        try:
            self.abs_torveny_df = pd.read_csv(abs_path, encoding='utf-8', sep=';')
            logger.info(f"? ABS_torveny loaded: {len(self.abs_torveny_df)} rows")
            
            # Clean column names
            self.abs_torveny_df.columns = self.abs_torveny_df.columns.str.strip()
            self.abs_torveny_df.columns = self.abs_torveny_df.columns.str.replace('?»?', '', regex=False)
            
            # Create hierarchy mappings
            self._create_hierarchy_mappings()
            
            return self.abs_torveny_df
            
        except FileNotFoundError:
            logger.error(f"? ABS_torveny not found: {abs_path}")
            raise
        except Exception as e:
            logger.error(f"? Error loading ABS_torveny: {e}")
            raise
    
    def _create_hierarchy_mappings(self):
        """Create dictionary mappings from ABS_torveny"""
        if self.abs_torveny_df is None:
            return
        
        df = self.abs_torveny_df
        
        if 'BaseBrandName' in df.columns and 'AdvertiserName' in df.columns:
            self.basebrand_to_advertiser = df.groupby('BaseBrandName')['AdvertiserName'].unique().to_dict()
            logger.info("? basebrand_to_advertiser created")
        
        if 'BrandName' in df.columns and 'BaseBrandName' in df.columns:
            self.brand_to_basebrand = df.groupby('BrandName')['BaseBrandName'].unique().to_dict()
            logger.info("? brand_to_basebrand created")
        
        if 'SegmentName' in df.columns and 'BrandName' in df.columns:
            self.segment_to_brand = df.groupby('SegmentName')['BrandName'].unique().to_dict()
            logger.info("? segment_to_brand created")
    
    def load_labels(self, use_processed: bool = True) -> pd.DataFrame:
        """
        Load labels CSV (either processed or raw)
        
        Args:
            use_processed: If True, try to load processed_labels.csv first
        """
        drive_path = self.config['data']['drive_path']
        processed_path = os.path.join(drive_path, self.config['data']['processed_labels_csv'])
        raw_path = os.path.join(drive_path, self.config['data']['labels_csv'])
        
        # Try processed first
        if use_processed and os.path.exists(processed_path):
            logger.info("?? Loading processed labels...")
            self.labels_df = pd.read_csv(processed_path, encoding='utf-8', sep=';')
            self._create_category_mappings()
            logger.info(f"? Processed labels loaded: {len(self.labels_df)} rows")
            return self.labels_df
        
        # Fall back to raw
        logger.info("?? Loading raw labels...")
        self.labels_df = pd.read_csv(raw_path, encoding='utf-8', sep=';')
        logger.info(f"? Raw labels loaded: {len(self.labels_df)} rows")
        
        return self.labels_df
    
    def _create_category_mappings(self):
        """Create index-to-category mappings from labels_df"""
        if self.labels_df is None:
            return
        
        # Get unique categories
        all_advertisers = sorted(self.labels_df['a_adver'].unique())
        all_basebrands = sorted(self.labels_df['a_basebrand'].unique())
        all_brands = sorted(self.labels_df['a_brand'].unique())
        all_segments = sorted(self.labels_df['a_segment'].unique())
        
        # Create bidirectional mappings
        self.advertiser_to_index = {name: idx for idx, name in enumerate(all_advertisers)}
        self.index_to_advertiser = {idx: name for idx, name in enumerate(all_advertisers)}
        
        self.basebrand_to_index = {name: idx for idx, name in enumerate(all_basebrands)}
        self.index_to_basebrand = {idx: name for idx, name in enumerate(all_basebrands)}
        
        self.brand_to_index = {name: idx for idx, name in enumerate(all_brands)}
        self.index_to_brand = {idx: name for idx, name in enumerate(all_brands)}
        
        self.segment_to_index = {name: idx for idx, name in enumerate(all_segments)}
        self.index_to_segment = {idx: name for idx, name in enumerate(all_segments)}
        
        logger.info(f"?? Categories: {len(all_segments)} segments, {len(all_brands)} brands, "
                   f"{len(all_basebrands)} basebrands, {len(all_advertisers)} advertisers")
    
    def get_num_categories(self) -> Tuple[int, int, int, int]:
        """Returns (num_segments, num_brands, num_basebrands, num_advertisers)"""
        return (
            len(self.segment_to_index),
            len(self.brand_to_index),
            len(self.basebrand_to_index),
            len(self.advertiser_to_index)
        )
