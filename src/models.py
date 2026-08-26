"""Model loading: bf16/fp16 by default, 4-bit bitsandbytes fallback.

Two rules from the contract are enforced here:

* the model is never loaded with `attn_implementation="eager"` - it loads as
  `sdpa`, and `src.attn_patch` swaps in the `srkv` wrapper around SDPA
  (CLAUDE.md A3);
* there is no silent CPU fallback. If CUDA is unavailable the loader raises
  unless `allow_cpu=True` is passed explicitly (which only the CPU tests do).

The precision decision is logged loudly whenever it is not what was asked for,
so a 4-bit run can never be mistaken for a bf16 run when reading results.
"""

from __future__ import annotations

import logging
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .compat import check_transformers_version

logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

#: rough parameter counts (billions), used to decide bf16 vs 4-bit up front
_PARAMS_B = {
    "qwen2.5-0.5b": 0.5,
    "qwen2.5-1.5b": 1.5,
    "qwen2.5-3b": 3.1,
    "qwen2.5-7b": 7.6,
    "llama3.2-1b": 1.2,
    "llama3.2-3b": 3.2,
}


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def choose_precision(name: str, context_len: int = 8192, headroom_gb: float = 2.5) -> str:
    """Pick "bf16"/"fp16"/"4bit" from the visible GPU and the model size.

    The KV cache is sized for the *uncompressed* baseline on purpose: the
    baseline has to fit too, or the comparison has nothing to compare against.
    """
    total = gpu_memory_gb()
    if total == 0.0:
        return "bf16"

    params_b = _PARAMS_B.get(name.lower(), 3.0)
    weights_gb = params_b * 2.0  # 2 bytes/param at 16-bit
    kv_gb = _estimate_kv_gb(name, context_len)
    if weights_gb + kv_gb + headroom_gb <= total:
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    logger.warning(
        "Falling back to 4-bit for %s: %.1f GB weights + %.1f GB KV + %.1f GB headroom "
        "exceeds the %.1f GB visible on this GPU.",
        name, weights_gb, kv_gb, headroom_gb, total,
    )
    return "4bit"


def _estimate_kv_gb(name: str, context_len: int) -> float:
    """Uncompressed fp16 KV cache size, from the published config."""
    try:
        cfg = AutoConfig.from_pretrained(resolve_model_id(name))
    except Exception:  # offline / gated - fall back to a rough guess
        return 1.0
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    bytes_total = 2 * cfg.num_hidden_layers * kv_heads * head_dim * context_len * 2
    return bytes_total / 1024**3


def load_model(
    name: str,
    *,
    precision: str = "auto",
    context_len: int = 8192,
    allow_cpu: bool = False,
    trust_remote_code: bool = False,
):
    """Load (model, tokenizer). `precision` is "auto" | "bf16" | "fp16" | "4bit"."""
    check_transformers_version()
    model_id = resolve_model_id(name)

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "No CUDA device visible. SR-KV does not silently fall back to CPU "
            "(CLAUDE.md 'Forbidden'). Pass allow_cpu=True only for CPU unit tests, "
            "or start a GPU runtime."
        )

    if precision == "auto":
        precision = choose_precision(name, context_len=context_len)
    logger.info("Loading %s at precision=%s", model_id, precision)

    kwargs: dict = {
        "attn_implementation": "sdpa",  # never "eager" - see CLAUDE.md A3
        "trust_remote_code": trust_remote_code,
    }

    if precision == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32  # CPU tests: bf16 matmuls are painfully slow
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.srkv_precision = precision
    return model, tokenizer


def model_kv_bytes_per_token(model) -> int:
    """Bytes of KV cache one token costs, for the memory report."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dtype_size = 2
    for p in model.parameters():
        dtype_size = p.dtype.itemsize if p.dtype.is_floating_point else 2
        break
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * dtype_size


def build_tiny_model(
    num_hidden_layers: int = 2,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    vocab_size: int = 2048,
    seed: int = 0,
):
    """A randomly-initialised 2-layer Qwen2 for CPU tests.

    Small enough to run a full prefill + generate loop in a second on CPU, but
    a real `Qwen2ForCausalLM` - so it exercises the actual HF cache plumbing,
    attention dispatch and mask construction that the GPU runs depend on.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=4096,
        use_cache=True,
        attn_implementation="sdpa",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


class TinyWordTokenizer:
    """Reversible word-level tokenizer for the CPU harness self-test.

    `eval/run.py --tiny` has to run with no network and no downloaded weights,
    but NIAH scoring needs `decode(encode(x)) == x` for the magic number to
    survive the round trip - so this keeps a real growing vocabulary rather
    than hashing tokens into buckets.
    """

    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.pad_token = self.eos_token = "<eos>"
        self.chat_template = None
        self._to_id: dict[str, int] = {"<eos>": 0, "<unk>": 1}
        self._to_word: dict[int, str] = {0: "<eos>", 1: "<unk>"}
        self.padding_side = "left"

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0

    def _encode_word(self, word: str) -> int:
        if word in self._to_id:
            return self._to_id[word]
        if len(self._to_id) >= self.vocab_size:
            return 1
        idx = len(self._to_id)
        self._to_id[word] = idx
        self._to_word[idx] = word
        return idx

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **kwargs):
        ids = [self._encode_word(w) for w in str(text).split()]
        if return_tensors == "pt":
            import torch as _torch

            return {"input_ids": _torch.tensor([ids], dtype=_torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        words = [self._to_word.get(int(i), "<unk>") for i in ids]
        if skip_special_tokens:
            words = [w for w in words if w not in ("<eos>", "<unk>")]
        return " ".join(words)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


def build_tiny_tokenizer(vocab_size: int = 2048) -> TinyWordTokenizer:
    return TinyWordTokenizer(vocab_size=vocab_size)
