"""
Hierarchical CNN model architecture
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)


def build_hierarchical_cnn(
    image_shape: tuple,
    num_segments: int,
    num_brands: int,
    num_basebrands: int,
    num_advertisers: int,
    dropout_rate: float = 0.5,
    conv_dropout_rate: float = 0.25
) -> Model:
    """
    Build hierarchical CNN model with cascading outputs
    
    Args:
        image_shape: (height, width, channels)
        num_segments: Number of segment categories
        num_brands: Number of brand categories
        num_basebrands: Number of basebrand categories
        num_advertisers: Number of advertiser categories
        dropout_rate: Dropout rate for dense layers
        conv_dropout_rate: Dropout rate for convolutional layers
        
    Returns:
        Compiled Keras Model
    """
    
    # Input layer
    input_layer = Input(shape=image_shape, name='input_image')
    
    # Convolutional base
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(conv_dropout_rate)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(conv_dropout_rate)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(conv_dropout_rate)(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(conv_dropout_rate)(x)
    
    x = Flatten()(x)
    
    # Common dense layers
    common_dense = Dense(512, activation='relu')(x)
    common_dense = BatchNormalization()(common_dense)
    common_dense = Dropout(dropout_rate)(common_dense)
    
    common_dense = Dense(256, activation='relu')(common_dense)
    common_dense = BatchNormalization()(common_dense)
    common_dense = Dropout(dropout_rate)(common_dense)
    
    # Hierarchical output branches
    # Level 1: Segment
    segment_branch = Dense(128, activation='relu')(common_dense)
    segment_branch = BatchNormalization()(segment_branch)
    segment_branch = Dropout(dropout_rate)(segment_branch)
    segment_output = Dense(num_segments, activation='softmax', name='segment_output')(segment_branch)
    
    # Level 2: Brand (depends on segment)
    brand_input = concatenate([common_dense, segment_branch])
    brand_branch = Dense(128, activation='relu')(brand_input)
    brand_branch = BatchNormalization()(brand_branch)
    brand_branch = Dropout(dropout_rate)(brand_branch)
    brand_output = Dense(num_brands, activation='softmax', name='brand_output')(brand_branch)
    
    # Level 3: BaseBrand (depends on brand)
    basebrand_input = concatenate([common_dense, brand_branch])
    basebrand_branch = Dense(128, activation='relu')(basebrand_input)
    basebrand_branch = BatchNormalization()(basebrand_branch)
    basebrand_branch = Dropout(dropout_rate)(basebrand_branch)
    basebrand_output = Dense(num_basebrands, activation='softmax', name='basebrand_output')(basebrand_branch)
    
    # Level 4: Advertiser (depends on basebrand)
    advertiser_input = concatenate([common_dense, basebrand_branch])
    advertiser_branch = Dense(128, activation='relu')(advertiser_input)
    advertiser_branch = BatchNormalization()(advertiser_branch)
    advertiser_branch = Dropout(dropout_rate)(advertiser_branch)
    advertiser_output = Dense(num_advertisers, activation='softmax', name='advertiser_output')(advertiser_branch)
    
    # Create model
    model = Model(
        inputs=input_layer,
        outputs=[segment_output, brand_output, basebrand_output, advertiser_output]
    )
    
    logger.info("? Hierarchical CNN model built")
    logger.info(f"   Segments: {num_segments}, Brands: {num_brands}, "
                f"BaseBrands: {num_basebrands}, Advertisers: {num_advertisers}")
    
    return model


def compile_model(model: Model, learning_rate: float = 0.0001):
    """
    Compile the model with optimizer and loss functions
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'segment_output': 'sparse_categorical_crossentropy',
            'brand_output': 'sparse_categorical_crossentropy',
            'basebrand_output': 'sparse_categorical_crossentropy',
            'advertiser_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'segment_output': 'accuracy',
            'brand_output': 'accuracy',
            'basebrand_output': 'accuracy',
            'advertiser_output': 'accuracy'
        }
    )
    
    logger.info(f"? Model compiled with learning rate: {learning_rate}")
