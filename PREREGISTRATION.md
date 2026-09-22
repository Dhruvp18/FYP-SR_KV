# Pre-registration — Phase 8: does centroid-merging help anywhere?

Written **before** any Phase 8 data exists. Committed first, deliberately, so
the success criterion cannot move once numbers are on screen. Git history is
the proof of ordering.

## Why this phase exists

Phases 5 and 6 produced a clean null. `centroid_merge` against
`snapkv_unified` — one boolean apart, the comparison the whole design exists
to make:

    NIAH 1.5B @ budget 0.2     1.000  vs  1.000     (delta  0.000)
    NIAH 3B    (both budgets)  0.983  vs  0.983     (delta  0.000)
    LongBench 1.5B @ 0.3       0.248  vs  0.249     (delta -0.001)

Two structural reasons that null may be an artefact of *what was measured*
rather than a property of the mechanism:

1. **Every metric so far rewards exact tokens.** NIAH is exact match;
   LongBench uses `qa_f1` and `rouge_l`, both n-gram overlap. A centroid is an
   attention-weighted average and cannot reproduce an exact token, so these
   metrics penalise it by construction. Consistent with this, `gov_report`
   (ROUGE) is the one task where merging is *significantly worse*
   (-0.0094, 95% CI [-0.0179, -0.0022], paired bootstrap n=25).
2. **Compression has been nearly free.** At budget 0.3 hard eviction costs
   only 2.3% against uncompressed. There is almost nothing for merging to
   recover.

## Hypotheses

- **H1 (metric).** Under a metric that rewards *distributional* fidelity
  rather than exact tokens, centroid-merging beats hard eviction. Perplexity
  is that metric, and is the standard KV-compression metric (StreamingLLM
  reports it).
- **H2 (pressure).** Under aggressive compression, where hard eviction
  discards information rather than redundancy, centroid-merging beats hard
  eviction. Partial signal should beat none.

## Success criterion — fixed now

For each experiment, the comparison is **`centroid_merge` vs
`snapkv_unified`**, paired on `sample_idx`, tested by paired bootstrap
(10,000 resamples, 95% CI).

A hypothesis is **SUPPORTED** only if:
- the 95% CI on the paired mean difference excludes zero **in the predicted
  direction**, and
- n >= 50 pairs per comparison, and
- it holds at **more than one** setting (>=2 budgets for H2, >=2 context
  lengths for H1).

Anything else is **NOT SUPPORTED**. A single significant cell among many is
noise, and is to be reported as such.

`sr_kv` vs `snapkv_unified` is reported alongside but is **secondary**: the
recency term is already established as harmful on depth-randomised retrieval,
and H1/H2 are about clustering.

## What gets reported either way

Every result, including negatives, and including any experiment started and
abandoned. The existing Phase 5/6 nulls stay in the writeup regardless of what
Phase 8 shows. If H1 and H2 both fail, the conclusion is that centroid-merging
does not help, stated plainly.

Explicitly ruled out:
- reporting a subgroup that was not named above,
- re-running a comparison with more samples because the first was not
  significant,
- switching the metric or the pairing after seeing results.

## Experiments

**E1 (H1) — perplexity.** qwen2.5-1.5b, contexts 2048/4096/8192, budgets
0.2/0.3, methods full/streaming_llm/snapkv_unified/centroid_merge/sr_kv,
n>=50 documents. Metric: token-level perplexity on a held-out continuation,
conditioned on the compressed cache of the prefix. Lower is better, so the
predicted direction is `centroid_merge` perplexity **below**
`snapkv_unified`.

**E2 (H2) — tight budgets.** LongBench, budgets 0.05/0.1/0.15, same four
tasks and methods as Phase 5, n=25/task (100 pairs/method).

## Known threat to validity

Perplexity and downstream task accuracy are not the same thing, and a win on
E1 alone does not establish a useful method. If E1 succeeds and E2 fails, the
honest claim is "centroid-merging preserves distributional information that
n-gram metrics do not credit", not "SR-KV is better".

---

## Addendum (2026-09-20) — H3: a task built for the mechanism, not borrowed

Written before any `phase8_gist_*` file exists. E1/E2 ask whether the null is
an artefact of *what was measured* on tasks NIAH and LongBench already
happened to include. H3 instead asks: is there *any* application where
clustering's actual mechanism - averaging cosine-similar candidate keys into
one attention-weighted centroid, per CLAUDE.md A5 / `src/clustering.py` - is
a plausible win, and can a task be built to isolate exactly that?

Two structural properties make clustering plausible over hard eviction, and
neither NIAH nor LongBench has both:

1. **The evicted content must be genuinely redundant** - similar key vectors
   for cosine-clustering to average into a faithful summary, rather than
   forcing dissimilar content together into a vector that resembles nothing.
   NIAH's needle is unique by construction; LongBench's `gov_report` is dense,
   non-redundant prose, and is already the one task where clustering measured
   *significantly worse* (-0.0094, 95% CI [-0.018, -0.002], n=25 - see above).
   This is cited as the reasoning that motivated H3, not as evidence for it:
   re-testing `gov_report` with more samples is exactly the move the base
   pre-registration rules out, so H3 is tested on a new task instead.
2. **The metric must not need exact tokens back.** Same reasoning as H1, but
   applied to a downstream judgment instead of perplexity: if the answer is
   "which of these four options", a blurry-but-present attention target can
   move the model toward the right choice without reproducing any wording.

`eval/gist_mcq.py` builds a task with both properties: a fact is restated
several times, in different phrasing, scattered non-locally through a long
document (so no single restatement is "recently attended" when compression
scores it), and the question is multiple-choice, scored on which letter comes
out - not on string overlap with any span in the text.

Two independent variants, each its own hypothesis:

- **H3a (`attribution`).** An entity is linked to an attribute (a "team
  colour") via four restatements at scattered depths, plus two single-mention
  decoy entities. Question: which team was the entity assigned to (MCQ).
  Tests whether the entity/attribute link survives when every individual
  mention is, alone, unremarkable to windowed-attention scoring.
- **H3b (`aggregation`).** A majority topic is mentioned five times and two
  minority topics once each, scattered non-locally. Question: which issue
  came up most often (MCQ). Tests whether *relative frequency* survives, not
  just existence - a centroid pulled from five similar mentions should point
  more strongly toward that topic than one pulled from a single mention,
  something hard eviction has no equivalent of (a token either survives whole
  or is gone; there is no "how many almost-evicted tokens agreed").

### Success criterion - identical structure to H1/H2, fixed now

For each variant independently: `centroid_merge` vs `snapkv_unified`, paired
on `sample_idx`, paired bootstrap (10,000 resamples, 95% CI). SUPPORTED
requires the CI to exclude zero in the predicted direction (higher accuracy
for `centroid_merge`), n>=50 pairs, at >=2 budget settings. H3a and H3b are
scored and reported independently - a win on one is not evidence for the
other, since they test different claims about what a centroid preserves.
`sr_kv` vs `snapkv_unified` is secondary, exactly as in H1/H2.

### Experiment

**E3 (H3a + H3b).** qwen2.5-1.5b, context 8192 (compression must actually
bite for either mechanism to show anything - see the "fair last shot" note
above on why 0.3 was nearly free), budgets 0.1/0.15, methods
full/snapkv_unified/centroid_merge/sr_kv, n=50/variant/budget.
`streaming_llm` is dropped for this experiment only: it is architecturally
different, not part of the clustering comparison, and this is a second
speculative experiment on top of an already-committed E1/E2, not a full
phase - `make phase8-gist`.

### Known threat to validity

This task is synthetic and designed specifically to give clustering the
conditions it should need. A win here establishes "the mechanism can win
somewhere", which is weaker than "the mechanism helps on realistic text" -
the same gap E1 already has between perplexity and downstream usefulness. A
loss here, on a task built to give the mechanism every advantage this project
could construct, would be considerably stronger evidence against it than
another null on tasks that were never designed to favour either policy.
---

## Addendum (2026-09-22) — H4: robustness to noise at the eviction boundary

Written before any `rank_swap_frac>0` data exists. H1-H3 all asked whether
centroid-merging preserves more *content*. H4 asks a different question:
does centroid-merging make the policy more *robust* - specifically, more
tolerant of getting individual keep/evict decisions wrong?

The motivating intuition: a hard-eviction method that makes a wrong call on a
borderline token loses that token completely, with no recourse. A
clustering method that makes the identical wrong call still folds that token
into a centroid alongside whatever else landed near the cutoff - lossy, but
not a total loss. If that intuition is right, injecting artificial noise into
*which* borderline tokens get evicted should hurt `snapkv_unified` more than
it hurts `centroid_merge`, in proportion to how much noise is injected.

This is a stress test, not a claim about how real eviction errors arise in
practice. The scoring function is not being modified - the noise is injected
mechanically, after scoring, only among tokens whose keep/evict status was
already a close call (see `src/scoring.py::apply_rank_swap`). Tokens the
policy is confident about (clearly the most/least important) are never
touched, so this cannot manufacture a difference out of decisions that were
never in question.

### Mechanism

`ScoringConfig.rank_swap_frac` (`src/scoring.py`): before the keep/evict
top-k split in `SRKVCache._compress()`, a band of
`round(rank_swap_frac * n_candidates)` slots straddling the cutoff (in
importance-sorted order) has its keep/evict assignment randomly reshuffled.
`rank_swap_frac=0` is exactly the original top-k (verified: full test suite
green with the change in place, default unchanged). Applied identically
regardless of `use_clustering`, so it is a clean one-flag addition to both
`snapkv_unified` and `centroid_merge` - same class, same mechanism, per
CLAUDE.md's binding design constraint.

### Hypothesis

**H4.** As `rank_swap_frac` increases from 0, `snapkv_unified`'s perplexity
degrades more than `centroid_merge`'s.

Primary statistic, per non-zero `rank_swap_frac` value X and per (context,
budget) setting: the **differential degradation**

    D = [nll(snapkv_unified, X) - nll(snapkv_unified, 0)]
      - [nll(centroid_merge, X) - nll(centroid_merge, 0)]

paired on `sample_idx` (the same prefix/continuation samples are reused
across every `rank_swap_frac` value at a given context length, from the same
seed), by paired bootstrap (10,000 resamples, 95% CI) - identical machinery
to H1-H3. D > 0 means `snapkv_unified` degraded more, supporting H4.

Also reported, not part of the verdict but requested for the writeup: each
method's *own* slope - `centroid_merge(X)` vs `centroid_merge(0)`, and
`snapkv_unified(X)` vs `snapkv_unified(0)` - independently, by the same
paired bootstrap, so the raw degradation curve for each method is visible on
its own, not just the difference between them.

### Success criterion - identical bar to H1-H3

For each `rank_swap_frac` value tested independently: SUPPORTED requires the
95% CI on D to exclude zero in the predicted direction (D > 0), n >= 50
pairs, holding at >= 2 of the 6 (context, budget) settings. Anything else is
NOT SUPPORTED for that noise level.

### Experiment

**E4.** qwen2.5-1.5b, contexts 2048/4096/8192, budgets 0.2/0.3 - the same six
settings as H1/E1, for continuity - methods `centroid_merge` and
`snapkv_unified` only (this tests the clustering mechanism specifically, not
the recency term), `rank_swap_frac` in {0.0, 0.10, 0.25}, n=50 documents per
cell. `rank_swap_frac=0.0` is re-measured fresh under this addendum's code
rather than reusing E1's numbers, so the comparison has no cross-run
confound from the code change itself.

### Known threat to validity

`rank_swap_frac` is a synthetic intervention with no claimed correspondence
to any real source of eviction error in this project's actual scoring
function - a null result here says centroid-merging is not more robust to
*this kind* of boundary noise, not that it is not robust to anything. A
positive result establishes a robustness property under a controlled stress
test, which is weaker than (but consistent with, and suggestive for) a claim
about real deployment conditions.
