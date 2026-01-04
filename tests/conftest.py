import pytest
from rlmollm.config import get_model_path, get_config_path

@pytest.fixture
def generator():
    """Initialized RLMolLMGenerator for testing."""
    from rlmollm import RLMolLMGenerator
    return RLMolLMGenerator(
        checkpoint_path=get_model_path("moses"),
        config_path=get_config_path("moses"),
        verbose=False
    )

