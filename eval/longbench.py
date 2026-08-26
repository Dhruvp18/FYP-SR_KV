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


def build_samples(
    tasks=("narrativeqa", "qasper", "gov_report", "triviaqa"),
    *,
    n_samples: int = 25,
    max_context_chars: int | None = None,
) -> list[LongBenchSample]:
    """Load the LongBench subset via `datasets` (needs network on first run)."""
    from datasets import load_dataset

    out: list[LongBenchSample] = []
    for task in tasks:
        if task not in TASKS:
            raise KeyError(f"unknown LongBench task {task!r}; known: {sorted(TASKS)}")
        ds = load_dataset("THUDM/LongBench", task, split="test")
        for idx, row in enumerate(ds):
            if idx >= n_samples:
                break
            context = row["context"]
            if max_context_chars:
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
