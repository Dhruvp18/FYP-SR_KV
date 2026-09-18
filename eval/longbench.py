"""LongBench subset: NarrativeQA, Qasper, GovReport, TriviaQA.

Four tasks, not the full suite - two single-doc QA, one summarisation, one
few-shot QA - chosen to cover the behaviours KV eviction is most likely to
break (locating one fact in a long document vs. needing the whole document).

Metrics follow the official LongBench implementation: F1 over normalised
tokens for the QA tasks, ROUGE-L F1 for GovReport. The normalisation and F1
below are transcriptions of LongBench's `metrics.py`; scores are comparable to
published LongBench numbers only for the same subset and the same prompt
template, so the report compares against our own uncompressed baseline rather
than against the leaderboard.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

#: LongBench task -> (max generated tokens, metric)
TASKS: dict[str, dict] = {
    "narrativeqa": {"max_new_tokens": 128, "metric": "qa_f1"},
    "qasper": {"max_new_tokens": 128, "metric": "qa_f1"},
    "gov_report": {"max_new_tokens": 512, "metric": "rouge_l"},
    "triviaqa": {"max_new_tokens": 32, "metric": "qa_f1"},
}

PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you "
        "can, using a single phrase or sentence if possible. If the question cannot be answered based "
        'on the information in the article, write "unanswerable".\n\n'
        "Article: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words.\n\nThe following are some examples.\n\n{context}\n\n{input}"
    ),
}


@dataclass
class LongBenchSample:
    task: str
    sample_idx: int
    context: str = field(repr=False, default="")
    question: str = ""
    answers: list[str] = field(default_factory=list)
    max_new_tokens: int = 128

    @property
    def task_id(self) -> str:
        return f"longbench/{self.task}/s{self.sample_idx}"

    def prompt(self) -> str:
        return PROMPTS[self.task].format(context=self.context, input=self.question)


def _longbench_data_dir() -> "Path":
    """Download+extract THUDM/LongBench's data.zip, cached by huggingface_hub.

    Not loaded via `datasets.load_dataset`: that repo ships a loading script
    (`LongBench.py`), and current `datasets` versions (5.x, what Kaggle images
    actually have installed) hard-refuse those - confirmed against a real run
    ("RuntimeError: Dataset scripts are no longer supported, but found
    LongBench.py"), not a permissions/trust_remote_code issue that can be
    flagged past. The script itself does nothing but download this same zip
    and read `data/<task>.jsonl` line by line (verified by reading it), so
    that's what this does directly - same data, no dependency on a loading
    mechanism the library has removed.
    """
    import zipfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench", repo_type="dataset", filename="data.zip"
    )
    extract_dir = Path(zip_path).parent / "extracted"
    data_dir = extract_dir / "data"
    if not data_dir.is_dir():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return data_dir


def _truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Keep the first and last halves of the token budget, drop the middle.

    LongBench's own evaluation harness truncates this way for models with a
    limited context window: relevant information can be anywhere in the
    document, so keeping both ends loses less than truncating from one side.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    kept = ids[:half] + ids[-(max_tokens - half):]
    return tokenizer.decode(kept)


def build_samples(
    tasks=("narrativeqa", "qasper", "gov_report", "triviaqa"),
    *,
    n_samples: int = 25,
    max_context_chars: int | None = None,
    tokenizer=None,
    max_context_tokens: int | None = None,
) -> list[LongBenchSample]:
    """Load the LongBench subset directly from its data.zip (needs network on first run).

    LongBench's native documents run far longer than this project's stated
    scope (8k-16k tokens) - narrativeqa alone averages well past that
    untruncated - and prefill computes full self-attention over the whole
    document before any cache eviction/merging ever runs (CLAUDE.md A2), so
    an oversized document OOMs identically for every method, not just the
    uncompressed baseline (confirmed: `full` tried to allocate 43.89 GiB on a
    14.56 GiB T4 on the real, untruncated narrativeqa context). Pass
    `tokenizer` + `max_context_tokens` to truncate precisely, the same way
    `niah.build_samples` controls context length by tokens rather than
    characters. `max_context_chars` is a cruder fallback for when no
    tokenizer is available.
    """
    import json

    data_dir = _longbench_data_dir()

    out: list[LongBenchSample] = []
    for task in tasks:
        if task not in TASKS:
            raise KeyError(f"unknown LongBench task {task!r}; known: {sorted(TASKS)}")
        path = data_dir / f"{task}.jsonl"
        with path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= n_samples:
                    break
                row = json.loads(line)
                context = row["context"]
                if max_context_tokens and tokenizer is not None:
                    context = _truncate_by_tokens(tokenizer, context, max_context_tokens)
                elif max_context_chars:
                    context = context[:max_context_chars]
                out.append(
                    LongBenchSample(
                        task=task,
                        sample_idx=idx,
                        context=context,
                        question=row["input"],
                        answers=list(row["answers"]),
                        max_new_tokens=TASKS[task]["max_new_tokens"],
                    )
                )
    return out


# ---------------------------------------------------------------------------
# metrics (transcribed from LongBench's metrics.py)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b):
            cur.append(prev[j] + 1 if token_a == token_b else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    if not pred or not gt:
        return 0.0
    lcs = _lcs_length(pred, gt)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(gt)
    return 2 * precision * recall / (precision + recall)


def score(sample: LongBenchSample, generated_text: str) -> float:
    """Best score over the reference answers, as LongBench does."""
    metric = TASKS[sample.task]["metric"]
    fn = qa_f1_score if metric == "qa_f1" else rouge_l_score
    prediction = generated_text.strip().split("\n")[0] if metric == "qa_f1" else generated_text
    return max((fn(prediction, gt) for gt in sample.answers), default=0.0)
