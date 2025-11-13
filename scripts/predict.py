"""
CLI script for making predictions on new images
"""

import argparse
import sys
from pathlib import Path
import logging
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from src.data.loader import DataLoader
from src.models.predictor import MaverickPredictor
from src.utils.io_utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict with MAVERICK model')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save prediction results (JSON)'
    )
    parser.add_argument(
        '--approve-new',
        action='store_true',
        help='Automatically approve new advertisers'
    )
    
    return parser.parse_args()


def main():
    """Main prediction function"""
    args = parse_args()
    
    # Load config
    logger.info("?? Loading configuration...")
    config = load_config(args.config)
    
    # Load data mappings
    logger.info("?? Loading data mappings...")
    data_loader = DataLoader(config)
    abs_torveny_df = data_loader.load_abs_torveny()
    labels_df = data_loader.load_labels(use_processed=True)
    
    # Load model
    logger.info(f"?? Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    # Create predictor
    predictor = MaverickPredictor(
        config=config,
        model=model,
        index_to_segment=data_loader.index_to_segment,
        index_to_brand=data_loader.index_to_brand,
        index_to_basebrand=data_loader.index_to_basebrand,
        index_to_advertiser=data_loader.index_to_advertiser,
        abs_torveny_df=abs_torveny_df,
        abs_torveny_path=config['data']['drive_path'] + config['data']['abs_torveny_csv']
    )
    
    # Make prediction
    logger.info(f"?? Predicting on {args.image}...")
    result = predictor.predict_with_new_category_support(
        image_path=args.image,
        enable_ocr=not args.no_ocr
    )
    
    # Display results
    print("\n" + "="*70)
    print("?? PREDICTION RESULTS")
    print("="*70)
    print(f"  Image:      {result['image_path']}")
    print(f"  Segment:    {result['segment']} (confidence: {result['segment_confidence']:.2%})")
    print(f"  Brand:      {result['brand']} (confidence: {result['brand_confidence']:.2%})")
    print(f"  BaseBrand:  {result['basebrand']} (confidence: {result['basebrand_confidence']:.2%})")
    print(f"  Advertiser: {result['advertiser']} (confidence: {result['advertiser_confidence']:.2%})")
    
    if result['is_new_advertiser']:
        print("\n?? NEW ADVERTISER DETECTED!")
        print(f"   Suggested: {result['advertiser']}")
        print(f"   OCR Evidence: {result['ocr_text'][:100]}...")
        
        if result['requires_approval']:
            print(f"   ?? Requires manual approval")
            
            if args.approve_new:
                print("   ? Auto-approving new advertiser...")
                success = predictor.approve_new_advertiser(
                    result['proposal_id'],
                    user_approved=True
                )
                if success:
                    print("   ? New advertiser saved to ABS_torveny.csv")
                else:
                    print("   ? Failed to save new advertiser")
    
    if not result['hierarchy_valid']:
        print(f"\n? Hierarchy Error: {result.get('hierarchy_error')}")
    
    print("="*70 + "\n")
    
    # Save results if output path specified
    if args.output:
        # Convert numpy types to native Python for JSON serialization
        json_result = {
            k: float(v) if isinstance(v, (float,)) else v
            for k, v in result.items()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"?? Results saved to {args.output}")


if __name__ == '__main__':
    main()
