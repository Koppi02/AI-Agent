"""
Text extraction from images using EasyOCR
"""

import easyocr
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts text from images using EasyOCR"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing OCR settings
        """
        self.config = config
        self.reader: Optional[easyocr.Reader] = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        try:
            gpu_enabled = self.config['ocr']['gpu_enabled']
            language = self.config['ocr']['language']
            
            self.reader = easyocr.Reader([language], gpu=gpu_enabled)
            logger.info(f"? EasyOCR initialized (GPU: {gpu_enabled})")
        except Exception as e:
            logger.error(f"? Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a single string
        """
        if self.reader is None:
            logger.error("EasyOCR reader not initialized")
            return ""
        
        try:
            # detail=0 returns only text without bounding boxes
            result = self.reader.readtext(image_path, detail=0)
            extracted_text = " ".join(result)
            
            logger.debug(f"?? Extracted from {image_path}: {extracted_text[:100]}...")
            return extracted_text
            
        except Exception as e:
            logger.error(f"? Error extracting text from {image_path}: {e}")
            return ""
    
    def extract_text_with_confidence(self, image_path: str) -> list:
        """
        Extract text with confidence scores and bounding boxes
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of tuples: (bbox, text, confidence)
        """
        if self.reader is None:
            logger.error("EasyOCR reader not initialized")
            return []
        
        try:
            result = self.reader.readtext(image_path)
            return result
        except Exception as e:
            logger.error(f"? Error extracting text with confidence: {e}")
            return []
