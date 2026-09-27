"""Sourcing one live-demo question from the project's own eval task.

Thin wrapper around `eval.gist_mcq.build_samples` - no eviction, scoring, or
splicing logic of its own. `eval/gist_mcq.py` is the project's purpose-built
task (PREREGISTRATION.md addendum H3): a real-prose passage with a fact
spliced in at several points, then a 4-option MCQ question with one correct
letter, scored by `eval.gist_mcq.score()`.
"""

from __future__ import annotations

import random

from eval.gist_mcq import GistSample, build_samples

from .config import GIST_CONTEXT_LEN_DEFAULT

VARIANTS = ("attribution", "aggregation")


def build_demo_sample(
    tokenizer,
    *,
    variant: str | None = None,
    context_len: int = GIST_CONTEXT_LEN_DEFAULT,
    corpus: str = "pg",
) -> GistSample:
    """One fresh (context, question, options, answer) sample.

    `corpus="pg"` (real Paul Graham prose) is the live-demo default, matching
    `build_samples`'s own default and its documented reason: synthetic filler
    is too uniformly boring for any scoring rule to have to work for its
    answer, which defeats the point of the task (see gist_mcq.py's
    docstring). Pass `corpus="synthetic"` for the CLI/test path, where no
    network access should be required.
    """
    chosen = variant if variant in VARIANTS else random.choice(VARIANTS)
    seed = random.randint(0, 1_000_000)  # a fresh sample every call
    samples = build_samples(
        tokenizer,
        context_lengths=[context_len],
        variants=(chosen,),
        n_samples=1,
        corpus=corpus,
        seed=seed,
    )
    return samples[0]
