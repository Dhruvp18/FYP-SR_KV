"""Phase 8 analysis: exactly the test named in PREREGISTRATION.md, nothing else.

    python scripts/analyse_phase8.py

Written before any Phase 8 data existed, so the comparison, the pairing, the
statistic and the bar are all fixed in code rather than chosen once the
numbers are visible. If you are tempted to add a subgroup here after seeing a
result, do not - the pre-registration names what counts, and everything else
is exploratory and must be labelled as such.

The comparison is `centroid_merge` vs `snapkv_unified`, one boolean apart, on
`sample_idx`, by paired bootstrap. SUPPORTED requires the 95% CI to exclude
zero in the predicted direction, n>=50 pairs, at >=2 settings.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TREATMENT = "centroid_merge"
CONTROL = "snapkv_unified"
#: Reported next to the primary test, never in place of it. The
#: pre-registration calls `sr_kv` vs `snapkv_unified` secondary because the
#: recency term is already established as harmful on depth-randomised
#: retrieval, and H1/H2 are about clustering. It is printed so the writeup
#: has the number, and it is excluded from the verdict so a win here cannot
#: be promoted into "H1 supported".
SECONDARY = "sr_kv"
#: Uncompressed reference. Not a comparison - it is what says whether
#: compression cost anything at all at a given setting, which is the context
#: that makes a null between two compressors meaningful rather than vacuous.
REFERENCE = "full"
MIN_PAIRS = 50
MIN_SETTINGS = 2
BOOTSTRAP = 10_000
SEED = 0


def paired_bootstrap(diffs, *, n=BOOTSTRAP, seed=SEED):
    """Mean paired difference with a 95% percentile CI."""
    rng = random.Random(seed)
    mean = statistics.mean(diffs)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return mean, boots[int(0.025 * n)], boots[int(0.975 * n)]


def load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((REPO_ROOT / "results").glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if "error" not in r]


def _pairs(records, *, setting_key, value_key, treatment, control):
    """{setting: {method: {sample_idx: value}}} for one pair of arms."""
    paired = defaultdict(dict)
    for r in records:
        method = r.get("method")
        if method not in (treatment, control):
            continue
        setting = setting_key(r)
        paired[setting].setdefault(method, {})[r["sample_idx"]] = r[value_key]
    return paired


def _row(setting, arms, *, treatment, control, lower_is_better):
    """One printed line; returns (n, significant_win) or None if unpairable."""
    if treatment not in arms or control not in arms:
        return None
    ids = sorted(set(arms[treatment]) & set(arms[control]))
    if not ids:
        return None
    t = [arms[treatment][i] for i in ids]
    c = [arms[control][i] for i in ids]
    # a "win" is treatment below control when lower is better
    diffs = [(ci - ti) if lower_is_better else (ti - ci) for ti, ci in zip(t, c)]
    mean, lo, hi = paired_bootstrap(diffs)
    better, worse = lo > 0, hi < 0
    mark = "WIN" if better else ("WORSE" if worse else "")
    print(f"{str(setting):<28}{len(ids):>5}{statistics.mean(t):>10.4f}"
          f"{statistics.mean(c):>10.4f}{mean:>+10.4f}"
          f"   [{lo:+.4f}, {hi:+.4f}]{mark:>6}")
    return len(ids), better


def _header(treatment, control, lower_is_better):
    print(f"{treatment} vs {control}, paired on sample_idx, "
          f"{'lower' if lower_is_better else 'higher'} is better")
    print(f"{'setting':<28}{'n':>5}{'treat':>10}{'ctrl':>10}{'diff':>10}{'95% CI':>24}{'':>6}")


def secondary(records, *, setting_key, value_key, lower_is_better, ref_key=None):
    """sr_kv vs snapkv_unified, and the cost of compressing at all.

    Printed, never scored. Nothing here can change the verdict - see the
    comment on SECONDARY.
    """
    print(f"\n--- secondary (reported, not part of the verdict) ---")
    paired = _pairs(records, setting_key=setting_key, value_key=value_key,
                    treatment=SECONDARY, control=CONTROL)
    if any(SECONDARY in arms and CONTROL in arms for arms in paired.values()):
        _header(SECONDARY, CONTROL, lower_is_better)
        for setting in sorted(paired, key=str):
            _row(setting, paired[setting], treatment=SECONDARY, control=CONTROL,
                 lower_is_better=lower_is_better)
    else:
        print(f"  no {SECONDARY} records paired with {CONTROL}")

    if ref_key is None:
        print(f"\n  no {REFERENCE} baseline in this experiment - the size of any "
              f"null between compressors is uninterpretable without one")
        return

    # The uncompressed arm is recorded at budget=1.0, so it shares no setting
    # with a compressed arm whose key includes the budget: pairing on the full
    # key finds nothing and prints "no baseline" even when the baseline is
    # right there in the file. `ref_key` is the part of the setting the
    # reference does share (the context, or the LongBench task), and each
    # compressed setting is paired against its own context's reference rather
    # than against a budget-collapsed average of several.
    refs = defaultdict(dict)
    arms_by_setting = defaultdict(lambda: defaultdict(dict))
    setting_to_ref = {}
    for r in records:
        if r.get("method") == REFERENCE:
            refs[ref_key(r)][r["sample_idx"]] = r[value_key]
        elif r.get("method") == CONTROL:
            setting = setting_key(r)
            arms_by_setting[setting][CONTROL][r["sample_idx"]] = r[value_key]
            setting_to_ref[setting] = ref_key(r)

    pairable = [s for s in arms_by_setting if setting_to_ref.get(s) in refs]
    if not pairable:
        print(f"\n  no {REFERENCE} baseline to compare against - the size of any "
              f"null between compressors is uninterpretable without it")
        return

    print(f"\n  cost of compression: {CONTROL} vs uncompressed {REFERENCE}")
    _header(CONTROL, REFERENCE, lower_is_better)
    for setting in sorted(pairable, key=str):
        arms = dict(arms_by_setting[setting])
        arms[REFERENCE] = refs[setting_to_ref[setting]]
        _row(setting, arms, treatment=CONTROL, control=REFERENCE,
             lower_is_better=lower_is_better)


def compare(records, *, setting_key, value_key, lower_is_better, label):
    """One row per setting, plus the pre-registered verdict."""
    paired = _pairs(records, setting_key=setting_key, value_key=value_key,
                    treatment=TREATMENT, control=CONTROL)

    print(f"\n=== {label} ===")
    _header(TREATMENT, CONTROL, lower_is_better)

    wins = 0
    rows_seen = 0
    for setting in sorted(paired, key=str):
        row = _row(setting, paired[setting], treatment=TREATMENT, control=CONTROL,
                   lower_is_better=lower_is_better)
        if row is None:
            continue
        rows_seen += 1
        n_pairs, better = row
        if better and n_pairs >= MIN_PAIRS:
            wins += 1

    if not rows_seen:
        print("  no paired records found")
        return None

    supported = wins >= MIN_SETTINGS
    print(f"\n  settings with a significant win at n>={MIN_PAIRS}: {wins} "
          f"(need >={MIN_SETTINGS})")
    print(f"  VERDICT: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    return supported


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    print("Phase 8 - pre-registered analysis (PREREGISTRATION.md)")
    print(f"criterion: 95% CI excludes zero in the predicted direction, "
          f"n>={MIN_PAIRS} pairs, at >={MIN_SETTINGS} settings")

    verdicts = {}

    ppl = load("phase8_perplexity_*.jsonl")
    if ppl:
        ppl_setting = lambda r: f"ctx={r['context_len']} b={r['budget']}"
        verdicts["H1 (perplexity)"] = compare(
            ppl,
            setting_key=ppl_setting,
            value_key="nll",
            lower_is_better=True,
            label="H1 - perplexity (metric rewards distribution, not n-grams)",
        )
        secondary(ppl, setting_key=ppl_setting, value_key="nll", lower_is_better=True,
                  ref_key=lambda r: f"ctx={r['context_len']}")
    else:
        print("\n=== H1 - perplexity === no data yet")

    tight = load("phase8_tight_longbench_*.jsonl")
    if tight:
        tight_setting = lambda r: f"{r['lb_task']} b={r['budget']}"
        verdicts["H2 (tight budgets)"] = compare(
            tight,
            setting_key=tight_setting,
            value_key="accuracy",
            lower_is_better=False,
            label="H2 - LongBench at aggressive compression",
        )
        # No uncompressed arm here on purpose: `full` is budget-independent
        # and Phase 5 already measured it on these same four tasks. Re-running
        # it would cost an hour to reproduce a number we have, and pulling it
        # in from the Phase 5 file would be the cross-experiment blending that
        # has already caused six bugs in this project.
        secondary(tight, setting_key=tight_setting, value_key="accuracy",
                  lower_is_better=False)
    else:
        print("\n=== H2 - tight budgets === no data yet")

    gist = load("phase8_gist_*.jsonl")
    if gist:
        gist_setting = lambda r: f"ctx={r['context_len']} b={r['budget']}"
        for variant, tag, label_name in (
            ("attribution", "H3a", "H3a - entity attribution (redundant mentions, MCQ-scored)"),
            ("aggregation", "H3b", "H3b - topic aggregation (relative frequency, MCQ-scored)"),
        ):
            rows = [r for r in gist if r.get("variant") == variant]
            if not rows:
                continue
            verdicts[f"{tag} ({variant})"] = compare(
                rows, setting_key=gist_setting, value_key="accuracy",
                lower_is_better=False, label=label_name,
            )
            secondary(rows, setting_key=gist_setting, value_key="accuracy",
                      lower_is_better=False, ref_key=lambda r: f"ctx={r['context_len']}")
    else:
        print("\n=== H3 - gist recall (redundant, MCQ-scored) === no data yet")

    print("\n" + "=" * 62)
    for name, supported in verdicts.items():
        print(f"  {name:<24}{'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    if verdicts and not any(verdicts.values()):
        print("\n  Neither hypothesis supported. The Phase 5/6 null stands, and it")
        print("  now survives a change of metric and a change of compression")
        print("  pressure - which makes it a stronger result, not a failed one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
