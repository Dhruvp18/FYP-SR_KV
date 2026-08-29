"""Phase 4: pick the centroid RoPE position convention from measured data.

    python eval/run.py --method sr_kv --rope_position_mode latest        ... \
        --output results/rope_latest.json
    python eval/run.py --method sr_kv --rope_position_mode earliest      ... \
        --output results/rope_earliest.json
    python eval/run.py --method sr_kv --rope_position_mode attn_weighted ... \
        --output results/rope_attn_weighted.json
    python scripts/freeze_rope_mode.py --apply

Reads the three sweeps, prints mean NIAH accuracy per mode, and (with
`--apply`) rewrites `configs/defaults.yaml` with the winner and flips
`rope_position_mode_frozen` to true. Every later phase inherits it from that
file, so no script has to hardcode a mode - and `tests/test_configs.py` checks
that none does.

Refuses to freeze if a mode is missing, or if all three modes are at chance,
because in that case the bug is upstream in clustering rather than in position
assignment and freezing a "winner" would just be freezing noise.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rope_positions import POSITION_MODES  # noqa: E402
from scripts.make_plots import load_records  # noqa: E402

DEFAULTS_PATH = REPO_ROOT / "configs" / "defaults.yaml"
#: below this, "the best mode" is indistinguishable from noise
CHANCE_CEILING = 0.05


def mode_scores(records, *, model=None) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("method") not in ("sr_kv", "centroid_merge"):
            continue
        if model and r.get("model") != model:
            continue
        mode = r.get("rope_position_mode")
        if mode in POSITION_MODES:
            scores[mode].append(r["accuracy"])
    return scores


def apply_winner(winner: str, evidence: str, path: Path = DEFAULTS_PATH) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("rope_position_mode:"):
            out.append(f"rope_position_mode: {winner}")
        elif line.startswith("rope_position_mode_frozen:"):
            out.append("rope_position_mode_frozen: true")
        elif line.startswith("rope_position_mode_evidence:"):
            out.append(f'rope_position_mode_evidence: "{evidence}"')
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--apply", action="store_true", help="write the winner into defaults.yaml")
    args = parser.parse_args(argv)

    scores = mode_scores(load_records(Path(args.results_dir)), model=args.model)
    missing = [m for m in POSITION_MODES if not scores.get(m)]
    if missing:
        print(f"missing results for mode(s): {missing}")
        print("run all three Phase 4 sweeps before freezing a default")
        return 1

    means = {m: sum(v) / len(v) for m, v in scores.items()}
    print(f"\n{'mode':<16}{'accuracy':>10}{'n':>6}")
    for mode in POSITION_MODES:
        print(f"{mode:<16}{means[mode]:>10.3f}{len(scores[mode]):>6}")

    winner = max(means, key=means.get)
    if means[winner] <= CHANCE_CEILING:
        print(
            f"\nREFUSING to freeze: the best mode scores {means[winner]:.3f}, which is at or\n"
            "below chance. All three modes collapsing together points at a bug upstream in\n"
            "clustering or centroid construction, not at the position convention. Fix that\n"
            "first - freezing now would freeze noise."
        )
        return 2

    spread = means[winner] - min(means.values())
    print(f"\nwinner: {winner} (accuracy {means[winner]:.3f}, spread across modes {spread:.3f})")
    if spread < 0.02:
        print("note: the three modes are within 2 points - report this as 'no strong effect'")

    if args.apply:
        evidence = ", ".join(f"{m}={means[m]:.3f}" for m in POSITION_MODES)
        apply_winner(winner, evidence)
        print(f"wrote {DEFAULTS_PATH} (rope_position_mode: {winner}, frozen)")
    else:
        print("(dry run - pass --apply to write configs/defaults.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
