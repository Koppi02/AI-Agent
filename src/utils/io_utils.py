"""
I/O utilities for config management and file operations
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"? Configuration loaded from {config_path}")
    return config


def save_config(config: Dict, output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"? Configuration saved to {output_path}")


def save_json(data: Any, output_path: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation level
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"? JSON saved to {output_path}")


def load_json(json_path: str) -> Any:
    """
    Load JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"? JSON loaded from {json_path}")
    return data


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
