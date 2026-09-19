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


def compare(records, *, setting_key, value_key, lower_is_better, label):
    """One row per setting, plus the pre-registered verdict."""
    paired = defaultdict(dict)
    for r in records:
        method = r.get("method")
        if method not in (TREATMENT, CONTROL):
            continue
        setting = setting_key(r)
        paired[setting].setdefault(method, {})[r["sample_idx"]] = r[value_key]

    print(f"\n=== {label} ===")
    print(f"{TREATMENT} vs {CONTROL}, paired on sample_idx, "
          f"{'lower' if lower_is_better else 'higher'} is better")
    print(f"{'setting':<28}{'n':>5}{'treat':>10}{'ctrl':>10}{'diff':>10}{'95% CI':>24}{'':>6}")

    wins = 0
    rows_seen = 0
    for setting in sorted(paired, key=str):
        arms = paired[setting]
        if TREATMENT not in arms or CONTROL not in arms:
            continue
        ids = sorted(set(arms[TREATMENT]) & set(arms[CONTROL]))
        if not ids:
            continue
        rows_seen += 1
        t = [arms[TREATMENT][i] for i in ids]
        c = [arms[CONTROL][i] for i in ids]
        # a "win" is treatment below control when lower is better
        diffs = [(ci - ti) if lower_is_better else (ti - ci) for ti, ci in zip(t, c)]
        mean, lo, hi = paired_bootstrap(diffs)
        better = lo > 0
        worse = hi < 0
        mark = "WIN" if better else ("WORSE" if worse else "")
        if better and len(ids) >= MIN_PAIRS:
            wins += 1
        print(f"{str(setting):<28}{len(ids):>5}{statistics.mean(t):>10.4f}"
              f"{statistics.mean(c):>10.4f}{mean:>+10.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{mark:>6}")

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
        verdicts["H1 (perplexity)"] = compare(
            ppl,
            setting_key=lambda r: f"ctx={r['context_len']} b={r['budget']}",
            value_key="nll",
            lower_is_better=True,
            label="H1 - perplexity (metric rewards distribution, not n-grams)",
        )
    else:
        print("\n=== H1 - perplexity === no data yet")

    tight = load("phase8_tight_longbench_*.jsonl")
    if tight:
        verdicts["H2 (tight budgets)"] = compare(
            tight,
            setting_key=lambda r: f"{r['lb_task']} b={r['budget']}",
            value_key="accuracy",
            lower_is_better=False,
            label="H2 - LongBench at aggressive compression",
        )
    else:
        print("\n=== H2 - tight budgets === no data yet")

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
