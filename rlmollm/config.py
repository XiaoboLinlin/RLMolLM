"""
Configuration paths for RLMolLM package.

This module provides centralized path management for models, configs, and data files.
Users can override these paths or use the defaults.
"""
import os
from pathlib import Path

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# Default paths (relative to package root)
DEFAULT_PATHS = {
    # Model weights (in assets/models/)
    "model_moses": "assets/models/moses_pretrained.pt",
    "model_zinc": "assets/models/zinc_pretrained.pt",
    "model_guacamol": "assets/models/guacamol_pretrained.pt",
    "model_gdb": "assets/models/gdb_pretrained.pt",
    
    # Config files (in assets/configs/)
    "config_moses": "assets/configs/moses.json",
    "config_zinc": "assets/configs/zinc.json",
    "config_guacamol": "assets/configs/guacamol.json",
    "config_gdb": "assets/configs/gdb.json",
    
    # Initial populations (in assets/initial_populations/)
    "population_moses": "assets/initial_populations/moses_2000.csv",
    "population_zinc": "assets/initial_populations/zinc_2000.csv",
    "population_guacamol": "assets/initial_populations/guacamol_2000.csv",
    "population_gdb": "assets/initial_populations/gdb_2000.csv",
}


def get_path(key: str, dataset: str = "moses") -> str:
    """
    Get a standard path for models, configs, or populations.
    
    Args:
        key: Type of path ('model', 'config', or 'population')
        dataset: Dataset name ('moses', 'zinc', 'guacamol', 'gdb')
    
    Returns:
        Absolute path to the requested resource
    
    Example:
        >>> from rlmollm.config import get_path
        >>> model_path = get_path('model', 'moses')
        >>> config_path = get_path('config', 'moses')
        >>> population_path = get_path('population', 'moses')
    """
    full_key = f"{key}_{dataset}"
    if full_key not in DEFAULT_PATHS:
        raise ValueError(
            f"Unknown path key: {full_key}. "
            f"Valid keys: {', '.join(DEFAULT_PATHS.keys())}"
        )
    
    rel_path = DEFAULT_PATHS[full_key]
    abs_path = PACKAGE_ROOT / rel_path
    
    return str(abs_path)


def check_paths_exist(dataset: str = "moses") -> dict:
    """
    Check if required paths exist for a given dataset.
    
    Args:
        dataset: Dataset name to check
    
    Returns:
        Dictionary with path status (True if exists, False otherwise)
    """
    status = {}
    for key in ['model', 'config', 'population']:
        try:
            path = get_path(key, dataset)
            status[f"{key}_{dataset}"] = os.path.exists(path)
        except ValueError:
            status[f"{key}_{dataset}"] = False
    
    return status


# Convenience functions
def get_model_path(dataset: str = "moses") -> str:
    """Get path to pre-trained model for dataset."""
    return get_path('model', dataset)


def get_config_path(dataset: str = "moses") -> str:
    """Get path to config file for dataset."""
    return get_path('config', dataset)


def get_population_path(dataset: str = "moses") -> str:
    """Get path to initial population for dataset."""
    return get_path('population', dataset)


if __name__ == "__main__":
    # Print all available paths
    print("Available RLMolLM paths:")
    print("=" * 60)
    for key, path in DEFAULT_PATHS.items():
        full_path = PACKAGE_ROOT / path
        exists = "✓" if os.path.exists(full_path) else "✗"
        print(f"{exists} {key}: {path}")

