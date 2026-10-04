"""A labelled synthetic incident stream and incremental, query-isolated eval.

No eviction logic lives here. All policies remain in src/caches.
"""
from __future__ import annotations

import copy
import hashlib
import random
import re
import time
from dataclasses import asdict, dataclass

import torch

from eval.memory import peak_memory
from src.attn_patch import attach_cache
from src.models import model_kv_bytes_per_token

VERSION = "incident-stream-v1"
SERVICES = ("database", "payments", "auth", "queue")
CLASSES = SERVICES + ("NONE",)
SCENARIOS = ("handoff", "recovery", "recurrence", "persistent")
METHODS = ("full", "streaming_llm", "snapkv_unified", "recency_hard_evict", "centroid_merge", "sr_kv")
CAPACITY = 512
CHUNK = 256
INTRO = (
    "Monitor these chronological service logs. An ERROR opens a service failure. "
    "A RECOVERY explicitly closes that service failure. INFO messages and successful "
    "individual requests do not close a failure. All services start healthy.\n"
)
QUESTION = (
    "\nBased only on the logs so far, which service currently has an unresolved failure? "
    "Only an explicit RECOVERY closes a failure; INFO requests do not. "
    "Answer with one letter only.\n"
    "A) database: investigate database connectivity\n"
    "B) payments: investigate the payment gateway\n"
    "C) auth: investigate authentication\n"
    "D) queue: investigate queue workers\n"
    "E) NONE: no unresolved failure\nAnswer:"
)
FAILURES = {
    "database": ("connection refused by database endpoint", "query failed: database connection timeout"),
    "payments": ("payment gateway rejected transaction", "payment authorisation request failed"),
    "auth": ("authentication token verification failed", "identity endpoint returned unavailable"),
    "queue": ("worker failed to acknowledge job", "queue consumer stopped processing jobs"),
}
NORMAL = ("request completed successfully", "metrics snapshot exported", "health probe scheduled",
          "configuration snapshot unchanged", "routine trace batch written", "maintenance check scheduled")


@dataclass
class IncidentStream:
    stream_id: int
    seed: int
    scenario: str
    segments: list[str]
    expected: list[str]
    event_trace: list[list[dict]]
    version: str = VERSION

    @property
    def content_hash(self):
        return hashlib.sha256("".join(self.segments).encode()).hexdigest()

    def to_dict(self):
        return {**asdict(self), "content_hash": self.content_hash}


def build_streams(n=24, seed=41003):
    streams = []
    for idx in range(n):
        rng = random.Random(f"{VERSION}:{seed}:{idx}")
        roles = list(SERVICES)
        rng.shuffle(roles)
        a, b, c, _ = roles
        scenario = SCENARIOS[idx % len(SCENARIOS)]
        targets = {
            "handoff": [a, b, c], "recovery": [a, b, "NONE"],
            "recurrence": [a, "NONE", a], "persistent": [a, a, b],
        }[scenario]
        active = "NONE"
        segments, trace = [], []
        for checkpoint, target in enumerate(targets):
            events, lines = [], []
            transition = rng.choice((25, 38, 51))
            for j in range(72):
                minute = checkpoint * 72 + j
                stamp = f"{minute // 60:02d}:{minute % 60:02d}:00"
                # Recovery, then an optional new failure, always chronological.
                if j == transition and active != target:
                    if active != "NONE":
                        lines.append(f"[{stamp}] RECOVERY service={active} incident closed after verified restoration.\n")
                        events.append({"event": "recovery", "service": active})
                    active = target
                    if active != "NONE":
                        lines.append(f"[{stamp}] ERROR service={active} {rng.choice(FAILURES[active])}.\n")
                        events.append({"event": "error", "service": active})
                # Some old failures are noisy; the final handoff is a rare new
                # incident. Repetition never changes the ground-truth state.
                repeated = active != "NONE" and j < 60 and rng.random() < 0.48
                if scenario == "handoff" and checkpoint == 2 and j >= transition:
                    repeated = False
                if repeated:
                    lines.append(f"[{stamp}] ERROR service={active} {rng.choice(FAILURES[active])}; attempt={j+1}.\n")
                    events.append({"event": "error", "service": active})
                else:
                    service = rng.choice(SERVICES)
                    lines.append(f"[{stamp}] INFO service={service} {rng.choice(NORMAL)}; trace={idx:03d}-{minute:03d}.\n")
            assert active == target
            segments.append("".join(lines))
            trace.append(events)
        streams.append(IncidentStream(idx, seed, scenario, segments, list(targets), trace))
    return streams


def predicted_class(text):
    match = re.match(r"\s*(?:Answer\s*:\s*)?([ABCDE])(?:\b|\))", text, re.I)
    return CLASSES["ABCDE".index(match.group(1).upper())] if match else "INVALID"


def prompt_parts(tokenizer):
    marker = "LOG_STREAM_START_8361"
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": INTRO + marker + QUESTION}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix, suffix = wrapped.split(marker)
    return prefix, suffix


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def ingest(model, cache, ids, *, chunk_size=CHUNK):
    """Advance a cache with new tokens at true absolute positions."""
    device = next(model.parameters()).device
    with attach_cache(model, cache):
        for start in range(0, len(ids), chunk_size):
            batch = torch.tensor([ids[start:start + chunk_size]], device=device)
            pos = cache.n_tokens_seen
            positions = torch.arange(pos, pos + batch.shape[1], device=device).unsqueeze(0)
            mask = torch.ones((1, cache.get_seq_length() + batch.shape[1]), dtype=torch.long, device=device)
            model(batch, past_key_values=cache, use_cache=True, attention_mask=mask,
                  position_ids=positions, logits_to_keep=1)
            if not cache.check_conservation():
                raise RuntimeError("Cache accounting failed during ingest")
            if cache.max_capacity is not None and cache.get_seq_length() > cache.max_capacity:
                raise RuntimeError("Retained cache exceeded fixed capacity")


@torch.no_grad()
def probe(model, tokenizer, cache, suffix):
    """Generate from a branch; never append the probe to the continuing stream."""
    device = next(model.parameters()).device
    sync(device)
    started = time.perf_counter()
    original_seen = cache.n_tokens_seen
    branch = copy.deepcopy(cache)
    query = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    positions = torch.arange(original_seen, original_seen + query.shape[1], device=device).unsqueeze(0)
    mask = torch.ones((1, branch.get_seq_length() + query.shape[1]), dtype=torch.long, device=device)
    with attach_cache(model, branch):
        out = model.generate(query, past_key_values=branch, attention_mask=mask,
                             position_ids=positions, do_sample=False, max_new_tokens=6,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    sync(device)
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(out[0, query.shape[1]:], skip_special_tokens=True)
    assert cache.n_tokens_seen == original_seen
    if not branch.check_conservation():
        raise RuntimeError("Probe cache accounting failed")
    del branch
    return text, elapsed


def measure_stream(model, tokenizer, stream, cache, *, chunk_size=CHUNK):
    device = next(model.parameters()).device
    prefix, suffix = prompt_parts(tokenizer)
    cumulative_ingest = 0.0
    for checkpoint, segment in enumerate(stream.segments):
        text = (prefix if checkpoint == 0 else "") + segment
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        with peak_memory() as mem:
            sync(device)
            start = time.perf_counter()
            ingest(model, cache, ids, chunk_size=chunk_size)
            sync(device)
            ingest_seconds = time.perf_counter() - start
            cumulative_ingest += ingest_seconds
            allocated = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
            stats = dict(cache.get_stats())
            generated, probe_seconds = probe(model, tokenizer, cache, suffix)
        prediction = predicted_class(generated)
        yield dict(
            checkpoint=checkpoint, expected=stream.expected[checkpoint], predicted=prediction,
            accuracy=float(prediction == stream.expected[checkpoint]), generated_text=generated,
            stream_tokens=cache.n_tokens_seen, new_tokens=len(ids), cache_stats=stats,
            capacity=cache.max_capacity, conservation_ok=bool(cache.check_conservation()),
            ingest_seconds=ingest_seconds, cumulative_ingest_seconds=cumulative_ingest,
            probe_seconds=probe_seconds, compression_seconds=sum(cache.compress_times),
            retained_kv_bytes=stats["n_tokens_cached"] * model_kv_bytes_per_token(model),
            allocated_before_probe=allocated, peak_gpu_bytes=mem["max_memory_allocated"],
            cache_config=cache.config_dict(),
        )
