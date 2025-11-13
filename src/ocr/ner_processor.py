"""
Named Entity Recognition using spaCy (Hungarian language support)
"""

import spacy
import re
import logging
from typing import List, Tuple, Set, Optional

logger = logging.getLogger(__name__)


class NERProcessor:
    """Hungarian NER processor with custom pattern matching"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing NER settings
        """
        self.config = config
        self.nlp: Optional[spacy.language.Language] = None
        self._load_model()
        
        # Variant indicators és stopwords betöltése
        self.variant_indicators = set(
            indicator.lower() 
            for indicator in config['ner']['variant_indicators']
        )
        self.stopwords = set(
            word.lower() 
            for word in config['ner']['stopwords_extended']
        )
    
    def _load_model(self):
        """Load spaCy Hungarian model"""
        model_name = self.config['ner']['model_name']
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"? spaCy model '{model_name}' loaded")
        except Exception as e:
            logger.error(f"? Failed to load spaCy model: {e}")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_label) tuples
        """
        if self.nlp is None:
            logger.error("spaCy model not loaded")
            return []
        
        entities = []
        doc = self.nlp(text)
        
        # spaCy NER entities
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        # Egyedi alfanumerikus minták (pl. "XC90", "Model 3", "iPhone 14 Pro")
        alphanumeric_patterns = self._extract_alphanumeric_patterns(text)
        entities.extend(alphanumeric_patterns)
        
        logger.debug(f"?? Extracted {len(entities)} entities")
        return entities
    
    def _extract_alphanumeric_patterns(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract alphanumeric patterns that could be product models
        
        Examples: "XC90", "iPhone 14", "Galaxy S23", "Golf GTI"
        """
        patterns = []
        
        # Regex: betûk és számok kombinációja, esetleg szóközökkel/kötõjelekkel
        alphanumeric_pattern = r'\b([A-Za-z0-9]+(?:[ -]?[A-Za-z0-9]+)*)\b'
        matches = re.findall(alphanumeric_pattern, text)
        
        for match in matches:
            match_lower = match.lower()
            
            # Szûrés
            if (len(match) > 1 and 
                not match.isdigit() and 
                match_lower not in self.stopwords):
                
                # Ha tartalmaz számot ÉS betût, valószínûleg modell név
                if re.search(r'[A-Za-z]', match) and re.search(r'[0-9]', match):
                    patterns.append((match, 'PRODUCT_MODEL'))
                
                # Vagy ha variant indicator-t tartalmaz
                elif any(indicator in match_lower for indicator in self.variant_indicators):
                    patterns.append((match, 'VARIANT'))
        
        return patterns
    
    def identify_new_variant(
        self,
        ner_entities: List[Tuple[str, str]],
        predicted_brand: Optional[str] = None,
        ocr_text: str = "",
        known_advertisers_lower: Set[str] = None
    ) -> List[str]:
        """
        Azonosítja az új változatokat/advertiser-eket az entitások alapján
        
        Args:
            ner_entities: NER entitások listája
            predicted_brand: A CNN által felismert brand
            ocr_text: Teljes OCR szöveg
            known_advertisers_lower: Már ismert advertiser-ek (lowercase)
            
        Returns:
            Lehetséges új advertiser nevek listája
        """
        if known_advertisers_lower is None:
            known_advertisers_lower = set()
        
        potential_new_variants = set()
        
        for entity_text, entity_label in ner_entities:
            entity_lower = entity_text.lower()
            
            # Kizárjuk az ismert advertiser-eket
            if entity_lower in known_advertisers_lower:
                continue
            
            # PRODUCT_MODEL vagy VARIANT típusúak
            if entity_label in ['PRODUCT_MODEL', 'VARIANT', 'MISC']:
                # Ha van predicted brand, kombináljuk
                if predicted_brand:
                    # Ha az entitás NEM tartalmazza a brand nevet, hozzáadjuk
                    if predicted_brand.lower() not in entity_lower:
                        combined = f"{predicted_brand} {entity_text}"
                        potential_new_variants.add(combined)
                    else:
                        potential_new_variants.add(entity_text)
                else:
                    potential_new_variants.add(entity_text)
        
        # OCR szövegbõl is keresünk
        if ocr_text and predicted_brand:
            # Keresünk brand név utáni szavakat
            brand_pattern = rf'\b{re.escape(predicted_brand)}\s+([A-Za-z0-9\-\s]+)'
            matches = re.findall(brand_pattern, ocr_text, re.IGNORECASE)
            
            for match in matches:
                match_clean = match.strip()
                if (len(match_clean) > 2 and 
                    match_clean.lower() not in known_advertisers_lower):
                    potential_new_variants.add(f"{predicted_brand} {match_clean}")
        
        result = list(potential_new_variants)
        logger.info(f"?? Identified {len(result)} potential new variants: {result}")
        return result
