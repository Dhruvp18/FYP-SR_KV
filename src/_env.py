"""Environment guards applied on `import src`.

`ensure_importable_transformers()` works around a broken *local* torchvision:
importing almost anything from `transformers` pulls in `image_utils`, which
imports torchvision whenever the package is merely *installed* - even if that
install is ABI-incompatible with the local torch and raises on import. SR-KV
touches no vision code, so a stub is enough to let the text models load.

On a healthy environment (Kaggle, Colab, CI) this is a no-op: it only fires
when `import torchvision` actually raises.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types


def _torchvision_is_broken() -> bool:
    if importlib.util.find_spec("torchvision") is None:
        return False  # not installed at all: transformers won't import it
    if "torchvision" in sys.modules:
        return False
    try:
        import torchvision  # noqa: F401
    except Exception:
        return True
    return False


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def ensure_importable_transformers() -> bool:
    """Returns True if a torchvision stub had to be installed."""
    if not _torchvision_is_broken():
        return False

    class InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"
        BOX = "box"
        HAMMING = "hamming"
        NEAREST_EXACT = "nearest_exact"

    tv = _stub_module("torchvision", __version__="0.0.0-srkv-stub")
    tv.transforms = _stub_module("torchvision.transforms", InterpolationMode=InterpolationMode)
    _stub_module("torchvision.transforms.v2")
    _stub_module("torchvision.transforms.v2.functional")
    tv.io = _stub_module("torchvision.io")
    return True
