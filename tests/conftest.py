"""Shared test fixtures.

The torchvision guard that used to live here now runs on `import src`
(`src/_env.py`), so every entrypoint gets it, not just pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: F401,E402  (applies the environment guards)


@pytest.fixture(scope="session")
def tiny_model():
    """A 2-layer randomly-initialised Qwen2 on CPU (see src.models)."""
    from src.models import build_tiny_model

    return build_tiny_model()


@pytest.fixture(scope="session")
def tiny_model_gqa():
    """Wider tiny model with a 4:1 GQA ratio, to exercise head grouping."""
    from src.models import build_tiny_model

    return build_tiny_model(
        num_hidden_layers=3, hidden_size=128, num_attention_heads=8, num_key_value_heads=2
    )


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from src.models import build_tiny_tokenizer

    return build_tiny_tokenizer()
