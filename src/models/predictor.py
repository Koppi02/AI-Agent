"""
Main prediction engine combining CNN, OCR, NER and hierarchy validation
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Optional, List
from pathlib import Path

from src.ocr.text_extractor import TextExtractor
from src.ocr.ner_processor import NERProcessor
from src.hierarchy.validator import HierarchyValidator
from src.hierarchy.new_category import NewCategoryHandler

logger = logging.getLogger(__name__)


class MaverickPredictor:
    """Main predictor class for hierarchical classification with new category detection"""
    
    def __init__(
        self,
        config: Dict,
        model: tf.keras.Model,
        index_to_segment: Dict,
        index_to_brand: Dict,
        index_to_basebrand: Dict,
        index_to_advertiser: Dict,
        abs_torveny_df,
        abs_torveny_path: str
    ):
        """
        Args:
            config: Configuration dictionary
            model: Trained Keras model
            index_to_*: Category index mappings
            abs_torveny_df: Hierarchy DataFrame
            abs_torveny_path: Path to ABS_torveny.csv
        """
        self.config = config
        self.model = model
        
        # Category mappings
        self.index_to_segment = index_to_segment
        self.index_to_brand = index_to_brand
        self.index_to_basebrand = index_to_basebrand
        self.index_to_advertiser = index_to_advertiser
        
        # Create reverse mappings for known advertisers
        self.known_advertisers_lower = set(
            name.lower() for name in index_to_advertiser.values()
        )
        
        # Initialize components
        self.text_extractor = TextExtractor(config)
        self.ner_processor = NERProcessor(config)
        self.hierarchy_validator = HierarchyValidator(abs_torveny_df)
        self.new_category_handler = NewCategoryHandler(abs_torveny_path)
        
        # Image preprocessing params
        self.img_height = config['model']['image_height']
        self.img_width = config['model']['image_width']
        self.confidence_threshold = config['prediction']['confidence_threshold']
        
        logger.info("? MaverickPredictor initialized")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: Dict, **kwargs):
        """
        Load predictor from saved model checkpoint
        
        Args:
            checkpoint_path: Path to .keras model file
            config: Configuration dictionary
            **kwargs: Additional required objects (mappings, dataframes)
        """
        model = tf.keras.models.load_model(checkpoint_path)
        logger.info(f"? Model loaded from {checkpoint_path}")
        
        return cls(config=config, model=model, **kwargs)
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for CNN"""
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array
    
    def _get_cnn_predictions(self, image_path: str) -> Dict:
        """Get raw CNN predictions"""
        img_array = self._preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        # Extract predictions for each level
        segment_probs = predictions[0][0]
        brand_probs = predictions[1][0]
        basebrand_probs = predictions[2][0]
        advertiser_probs = predictions[3][0]
        
        return {
            'segment': {
                'index': np.argmax(segment_probs),
                'confidence': float(segment_probs[np.argmax(segment_probs)]),
                'name': self.index_to_segment.get(np.argmax(segment_probs), 'Unknown')
            },
            'brand': {
                'index': np.argmax(brand_probs),
                'confidence': float(brand_probs[np.argmax(brand_probs)]),
                'name': self.index_to_brand.get(np.argmax(brand_probs), 'Unknown')
            },
            'basebrand': {
                'index': np.argmax(basebrand_probs),
                'confidence': float(basebrand_probs[np.argmax(basebrand_probs)]),
                'name': self.index_to_basebrand.get(np.argmax(basebrand_probs), 'Unknown')
            },
            'advertiser': {
                'index': np.argmax(advertiser_probs),
                'confidence': float(advertiser_probs[np.argmax(advertiser_probs)]),
                'name': self.index_to_advertiser.get(np.argmax(advertiser_probs), 'Unknown')
            }
        }
    
    def predict_with_new_category_support(
        self,
        image_path: str,
        enable_ocr: bool = True
    ) -> Dict:
        """
        Full prediction pipeline with new category detection
        
        Args:
            image_path: Path to image file
            enable_ocr: Whether to use OCR for new variant detection
            
        Returns:
            Complete prediction result dictionary
        """
        logger.info(f"?? Predicting: {image_path}")
        
        # 1. CNN Predictions
        cnn_preds = self._get_cnn_predictions(image_path)
        
        result = {
            'image_path': image_path,
            'segment': cnn_preds['segment']['name'],
            'segment_confidence': cnn_preds['segment']['confidence'],
            'brand': cnn_preds['brand']['name'],
            'brand_confidence': cnn_preds['brand']['confidence'],
            'basebrand': cnn_preds['basebrand']['name'],
            'basebrand_confidence': cnn_preds['basebrand']['confidence'],
            'advertiser': cnn_preds['advertiser']['name'],
            'advertiser_confidence': cnn_preds['advertiser']['confidence'],
            'is_new_advertiser': False,
            'requires_approval': False,
            'ocr_text': '',
            'ner_entities': [],
            'suggested_new_variants': []
        }
        
        # 2. OCR + NER if advertiser confidence is low
        if enable_ocr and cnn_preds['advertiser']['confidence'] < self.confidence_threshold:
            logger.info(f"?? Low advertiser confidence ({cnn_preds['advertiser']['confidence']:.2f}), running OCR...")
            
            # Extract text
            ocr_text = self.text_extractor.extract_text(image_path)
            result['ocr_text'] = ocr_text
            
            if ocr_text:
                # NER processing
                ner_entities = self.ner_processor.extract_entities(ocr_text)
                result['ner_entities'] = ner_entities
                
                # Identify new variants
                suggested_variants = self.ner_processor.identify_new_variant(
                    ner_entities=ner_entities,
                    predicted_brand=result['brand'],
                    ocr_text=ocr_text,
                    known_advertisers_lower=self.known_advertisers_lower
                )
                result['suggested_new_variants'] = suggested_variants
                
                # If new variants found, use the first one as advertiser suggestion
                if suggested_variants:
                    result['advertiser'] = suggested_variants[0]
                    result['is_new_advertiser'] = True
                    logger.info(f"?? Suggested new advertiser: {suggested_variants[0]}")
        
        # 3. Hierarchy Validation
        validation = self.hierarchy_validator.validate_prediction({
            'segment': result['segment'],
            'brand': result['brand'],
            'basebrand': result['basebrand'],
            'advertiser': result['advertiser']
        })
        
        result['hierarchy_valid'] = validation['is_valid']
        
        if not validation['is_valid']:
            result['hierarchy_error'] = validation.get('error', '')
            result['suggested_correction'] = validation.get('suggested_correction')
            logger.warning(f"?? Invalid hierarchy: {validation.get('error')}")
        
        # 4. Handle new advertiser proposal
        if result['is_new_advertiser'] and validation['is_valid']:
            result['requires_approval'] = True
            result['suggested_csv_entry'] = validation.get('suggested_entry')
            
            # Create proposal
            proposal = self.new_category_handler.propose_new_advertiser(
                suggested_entry=validation['suggested_entry'],
                confidence=result['brand_confidence'],
                image_path=image_path,
                ocr_evidence=result['ocr_text']
            )
            result['proposal_id'] = proposal['proposed_at']
        
        # 5. Log result
        self._log_prediction_result(result)
        
        return result
    
    def _log_prediction_result(self, result: Dict):
        """Pretty print prediction results"""
        logger.info("\n" + "="*60)
        logger.info("?? PREDICTION RESULTS")
        logger.info("="*60)
        logger.info(f"  Segment:    {result['segment']} (conf: {result['segment_confidence']:.2f})")
        logger.info(f"  Brand:      {result['brand']} (conf: {result['brand_confidence']:.2f})")
        logger.info(f"  BaseBrand:  {result['basebrand']} (conf: {result['basebrand_confidence']:.2f})")
        logger.info(f"  Advertiser: {result['advertiser']} (conf: {result['advertiser_confidence']:.2f})")
        
        if result['is_new_advertiser']:
            logger.info("\n?? NEW ADVERTISER DETECTED!")
            logger.info(f"   Suggested: {result['advertiser']}")
            logger.info(f"   Requires manual approval: {result['requires_approval']}")
        
        if not result['hierarchy_valid']:
            logger.error(f"\n? Hierarchy Error: {result.get('hierarchy_error')}")
        
        logger.info("="*60 + "\n")
    
    def approve_new_advertiser(self, proposal_id: str, user_approved: bool = True) -> bool:
        """
        Approve and save a new advertiser proposal
        
        Args:
            proposal_id: Timestamp ID of the proposal
            user_approved: User approval status
            
        Returns:
            Success status
        """
        # Find proposal by ID
        for proposal in self.new_category_handler.pending_proposals:
            if proposal['proposed_at'] == proposal_id:
                return self.new_category_handler.approve_and_save(proposal, user_approved)
        
        logger.error(f"? Proposal not found: {proposal_id}")
        return False
