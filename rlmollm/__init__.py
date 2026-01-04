"""
RLMolLM: Reinforcement Learning-Enhanced Language Model for Molecular Design

A Python package for molecular generation and optimization using genetic algorithms
with optional reinforcement learning (PPO).
"""

__version__ = "0.1.0"

from .generator import RLMolLMGenerator
from .config import (
    get_path,
    get_model_path,
    get_config_path,
    get_population_path,
    check_paths_exist,
)

__all__ = [
    'RLMolLMGenerator',
    'get_path',
    'get_model_path',
    'get_config_path',
    'get_population_path',
    'check_paths_exist',
]


