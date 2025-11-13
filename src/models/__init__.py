"""
CNN model architecture and prediction logic
"""

from src.models.cnn_model import build_hierarchical_cnn
from src.models.predictor import MaverickPredictor

__all__ = ['build_hierarchical_cnn', 'MaverickPredictor']
