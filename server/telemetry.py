"""Observing a cache's own bookkeeping, without reimplementing its decisions.

`SRKVCacheBase` (src/caches/base.py) tracks, per slot, a current RoPE
`position`, a `weight` (how many original tokens it represents), and an
`is_centroid` flag - but it does not keep a history of *which* original token
positions fed a given centroid, only the count. The demo wants to draw that
span, so this module reconstructs it by diffing consecutive snapshots of the
cache's own public state: `cache.positions[0]`, `cache.slot_weights[0]`,
`cache.is_centroid[0]` (layer 0, batch 0, head 0 - the class's own documented
representative slot, see base.py's "Stats are read off (batch 0, kv-head 0)").

This never guesses at scoring or eviction logic. It only asks: which slots
that existed a moment ago are gone now, and which slot that's new now has a
weight that adds up to some of them - and treats that arithmetic match as "the
gone ones were folded into this one." A slot that vanishes with no matching
weight increase anywhere was hard-evicted, not merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotSnapshot:
    positions: list[float]
    weights: list[int]
    is_centroid: list[bool]


def snapshot(cache) -> SlotSnapshot:
    """Read a cache's layer-0 slot state as plain Python lists."""
    if 0 not in cache.positions or cache.positions[0].numel() == 0:
        return SlotSnapshot([], [], [])
    pos = cache.positions[0][0, 0].tolist()
    weights = cache.slot_weights[0][0, 0].tolist()
    centroid = cache.is_centroid[0][0, 0].tolist()
    return SlotSnapshot(pos, [int(w) for w in weights], [bool(c) for c in centroid])


@dataclass
class MergeEvent:
    position: float
    weight: int
    member_positions: list[int]


@dataclass
class Diff:
    merged: list[MergeEvent] = field(default_factory=list)
    evicted_positions: list[int] = field(default_factory=list)
    #: every original position currently folded into *some* centroid, and
    #: every original position ever hard-evicted (cumulative, whole run) -
    #: computed here so the frontend never has to reconstruct per-position
    #: status itself from a stream of deltas (fragile: centroid positions
    #: drift a little every step even with no new merge, so "match by
    #: position" breaks across steps - matching is done once, here, using the
    #: cache's own weight/position pairing).
    folded_positions: list[int] = field(default_factory=list)
    evicted_positions_cumulative: list[int] = field(default_factory=list)


class LaneTelemetryState:
    """Per-lane membership tracker, seeded fresh for every new run."""

    def __init__(self) -> None:
        #: rounded-position identity -> the set of *original* token positions
        #: it currently stands for. A brand-new real token is seeded as {p: {p}}
        #: the first time it's observed.
        self.membership: dict[int, set[int]] = {}
        self.all_evicted: set[int] = set()

    def diff(self, prev: SlotSnapshot, new: SlotSnapshot) -> Diff:
        prev_by_key = {round(p): (w, c) for p, w, c in zip(prev.positions, prev.weights, prev.is_centroid)}
        new_by_key = {round(p): (w, c) for p, w, c in zip(new.positions, new.weights, new.is_centroid)}
        prev_keys, new_keys = set(prev_by_key), set(new_by_key)

        appeared = new_keys - prev_keys
        disappeared = new_keys.symmetric_difference(prev_keys) & prev_keys
        remaining_disappeared = set(disappeared)

        out = Diff()
        for a_pos in sorted(appeared):
            target_weight, a_is_centroid = new_by_key[a_pos]
            self.membership.setdefault(a_pos, {a_pos})  # seed, may be overwritten below
            if not a_is_centroid:
                # a real new token is never "the same slot" as anything that
                # just vanished - only a centroid can be the result of a
                # compress() step (a merge, or the same centroid re-positioned
                # with unchanged weight). Matching a brand-new plain token
                # against a same-weight disappeared plain token would wrongly
                # read every hard-eviction (streaming_llm/snapkv_unified) as a
                # silent no-op instead of a real, permanent eviction.
                continue
            candidates = sorted(remaining_disappeared, key=lambda p: abs(p - a_pos))
            chosen: list[int] = []
            acc = 0
            for c_pos in candidates:
                c_w, _ = prev_by_key[c_pos]
                if acc + c_w <= target_weight:
                    chosen.append(c_pos)
                    acc += c_w
                if acc == target_weight:
                    break
            if acc == target_weight and chosen:
                members: set[int] = set()
                for c_pos in chosen:
                    members |= self.membership.pop(c_pos, {c_pos})
                    remaining_disappeared.discard(c_pos)
                self.membership[a_pos] = members
                if len(chosen) > 1:
                    # more than one prior slot collapsed into this one - a real
                    # merge, not just a centroid's blended position drifting.
                    out.merged.append(
                        MergeEvent(position=a_pos, weight=target_weight, member_positions=sorted(members))
                    )

        for d_pos in remaining_disappeared:
            members = self.membership.pop(d_pos, {d_pos})
            out.evicted_positions.extend(sorted(members))

        self.all_evicted.update(out.evicted_positions)
        out.evicted_positions_cumulative = sorted(self.all_evicted)

        folded: set[int] = set()
        for pos, is_c in zip(new.positions, new.is_centroid):
            if is_c:
                folded |= self.membership.get(round(pos), set())
        out.folded_positions = sorted(folded)

        return out
