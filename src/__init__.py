"""SR-KV: summarise-and-retain KV cache eviction.

Importing this package applies the environment guards in `src._env` before
anything reaches `transformers`; see that module for why.
"""

from ._env import ensure_importable_transformers

TORCHVISION_STUBBED = ensure_importable_transformers()

__all__ = ["TORCHVISION_STUBBED"]
