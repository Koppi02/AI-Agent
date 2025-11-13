"""
CLI script for training the hierarchical CNN model
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.models.cnn_model import build_hierarchical_cnn, compile_model
from src.utils.io_utils import load_config, ensure_dir
from src.utils.visualization import plot_training_history

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MAVERICK hierarchical CNN')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--initial-epoch',
        type=int,
        default=0,
        help='Initial epoch for resuming training'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Load config
    logger.info("?? Loading configuration...")
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # 1. Load data
    logger.info("?? Loading data...")
    data_loader = DataLoader(config)
    data_loader.load_abs_torveny()
    data_loader.load_labels(use_processed=True)
    
    # 2. Preprocess images
    logger.info("??? Preprocessing images...")
    preprocessor = ImagePreprocessor(config, data_loader.labels_df)
    preprocessor.preprocess_labels()
    train_dataset, val_dataset = preprocessor.create_datasets()
    
    # 3. Build model
    logger.info("??? Building model...")
    num_segments, num_brands, num_basebrands, num_advertisers = data_loader.get_num_categories()
    
    model = build_hierarchical_cnn(
        image_shape=(
            config['model']['image_height'],
            config['model']['image_width'],
            config['model']['channels']
        ),
        num_segments=num_segments,
        num_brands=num_brands,
        num_basebrands=num_basebrands,
        num_advertisers=num_advertisers,
        dropout_rate=config['model']['dropout_rate'],
        conv_dropout_rate=config['model']['conv_dropout_rate']
    )
    
    compile_model(model, learning_rate=config['training']['learning_rate'])
    
    # 4. Load checkpoint if specified
    initial_epoch = args.initial_epoch
    if args.checkpoint:
        logger.info(f"?? Loading checkpoint: {args.checkpoint}")
        model.load_weights(args.checkpoint)
    
    # 5. Setup callbacks
    checkpoint_dir = config['checkpoints']['directory']
    ensure_dir(checkpoint_dir)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "cp-{epoch:04d}.weights.h5"),
            save_weights_only=True,
            verbose=1,
            save_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 6. Train
    logger.info("?? Starting training...")
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        initial_epoch=initial_epoch,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save final model
    model_save_path = config['checkpoints']['model_save_path']
    ensure_dir(os.path.dirname(model_save_path))
    
    logger.info(f"?? Saving final model to {model_save_path}")
    model.save(model_save_path)
    
    # 8. Plot training history
    logger.info("?? Generating training plots...")
    plot_save_path = model_save_path.replace('.keras', '_history.png')
    plot_training_history(history.history, save_path=plot_save_path)
    
    logger.info("? Training completed successfully!")


if __name__ == '__main__':
    main()
