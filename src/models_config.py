"""Utility to load and access model configuration from models_config.yaml"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional


# Path to the config file (relative to project root)
CONFIG_PATH = Path(__file__).parent.parent / "models_config.yaml"


def load_config() -> Dict:
    """Load the model configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_all_models() -> Dict:
    """Get the models enabled for this standalone benchmark."""
    config = load_config()
    return {
        key: info for key, info in config.get('models', {}).items()
        if info.get('enabled', True)
    }


def get_model_directory_names() -> List[str]:
    """Get list of all model directory names (for makefile MODELS variable)."""
    models = get_all_models()
    return [info['directory_name'] for info in models.values()]


def get_model_display_names() -> Dict[str, str]:
    """Get mapping from directory_name to display_name."""
    models = get_all_models()
    return {info['directory_name']: info['display_name'] for info in models.values()}


def get_model_huggingface_ids() -> Dict[str, str]:
    """Get mapping from directory_name to huggingface_id."""
    models = get_all_models()
    return {info['directory_name']: info['huggingface_id'] for info in models.values()}


def get_model_order() -> List[str]:
    """Get the order of models for benchmark tables (by directory_name).

    The order is derived directly from the order of entries in
    models_config.yaml, keeping the YAML the single source of truth.
    """
    return get_model_directory_names()


def get_model_q_text(directory_name: str) -> Optional[str]:
    """Get query text prefix for a given directory_name."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('q_text')
    return None


def get_model_doc_text(directory_name: str) -> Optional[str]:
    """Get document text prefix for a given directory_name (used for both positive and negative samples)."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('doc_text')
    return None


def get_model_normalize(directory_name: str) -> bool:
    """Get normalize setting for a given directory_name."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('normalize', False)
    return False
