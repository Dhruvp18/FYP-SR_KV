"""Version-tolerant accessors for the transformers Cache internals.

transformers 5.x stores per-layer tensors as ``cache.layers[i].keys`` /
``.values``; 4.x stored them as ``cache.key_cache[i]`` / ``cache.value_cache[i]``.
Everything in this repo goes through the helpers here so that a Kaggle image
shipping a different minor version does not require touching cache logic.
"""

from __future__ import annotations

import torch

MIN_TRANSFORMERS = (5, 0)


def check_transformers_version() -> str:
    import transformers

    v = transformers.__version__
    parts = []
    for chunk in v.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"SR-KV requires transformers>={'.'.join(map(str, MIN_TRANSFORMERS))}, "
            f"found {v}. Run: pip install -U 'transformers>=5.0'"
        )
    return v


def n_layers(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(getattr(cache, "key_cache", []))


def layer_initialized(cache, layer_idx: int) -> bool:
    if hasattr(cache, "layers"):
        if layer_idx >= len(cache.layers):
            return False
        layer = cache.layers[layer_idx]
        return bool(getattr(layer, "is_initialized", False)) and layer.keys.numel() > 0
    kc = getattr(cache, "key_cache", [])
    return layer_idx < len(kc) and kc[layer_idx].numel() > 0


def get_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (keys, values) tensors stored for ``layer_idx``."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def set_kv(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Replace the stored (keys, values) for ``layer_idx`` in place."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys = keys
        layer.values = values
        return
    cache.key_cache[layer_idx] = keys
    cache.value_cache[layer_idx] = values


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[b, kv_heads, s, d] -> [b, kv_heads * n_rep, s, d] (GQA expansion)."""
    if n_rep == 1:
        return hidden
    b, kv_heads, s, d = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(b, kv_heads, n_rep, s, d)
    return hidden.reshape(b, kv_heads * n_rep, s, d)
