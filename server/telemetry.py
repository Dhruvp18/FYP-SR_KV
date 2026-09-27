"""Observing a cache's own bookkeeping, without reimplementing its decisions.

The obvious design - track each centroid's identity across steps and its
exact set of original members - turns out not to be reconstructable in
general. `SRKVCache._compress()` (`src/caches/sr_kv.py`) reclusters by
default (`recluster_centroids=True`, the class's own default): a compress
call can dissolve *every* existing centroid and re-run k-means over the
combined pool of old centroid members and newly-eligible plain tokens,
producing a fresh set of `n_centroids` slots with no stable correspondence to
the previous ones. A slot's blended position isn't "the same centroid,
repositioned" from one step to the next - it can be a genuinely different
grouping. Trying to match old centroids to new ones by weight/position
(an earlier version of this file did exactly that) works until two or more
centroids coexist and reclustering reshuffles them, at which point the
matching silently misattributes members - some real folds get counted as
hard evictions instead, breaking the `seen == alive + folded + evicted`
identity real runs otherwise satisfy exactly.

What *is* exactly knowable, every step, needs no matching at all: which
original positions are currently individual plain slots (their position is
always their own exact original position - a plain token is never
repositioned) - call this "alive." A position that stops being alive stops
being alive forever (no algorithm in this repo ever un-merges or restores a
token), and CLAUDE.md's own contract fixes what "not alive" means, per
method: `use_clustering=True` methods (`centroid_merge`, `sr_kv`) never hard
drop anything - every non-alive token is inside *some* centroid, forever
folded, never evicted. `use_clustering=False` methods never form a centroid
at all - every non-alive token is hard-evicted. So "not alive" plus "does
this method cluster" is the whole answer, with no per-slot matching to get
wrong.
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
class Diff:
    #: positions that stopped being alive on *this* call (only meaningful for
    #: hard-eviction methods; empty for clustering methods, since nothing
    #: they drop is a hard eviction).
    evicted_positions: list[int] = field(default_factory=list)
    #: positions that stopped being alive on *this* call for a clustering
    #: method - i.e. were just folded into the centroid pool.
    newly_folded_positions: list[int] = field(default_factory=list)
    #: every original position currently folded into *some* centroid
    #: (cumulative, whole run so far).
    folded_positions: list[int] = field(default_factory=list)
    #: every original position ever hard-evicted (cumulative, whole run).
    evicted_positions_cumulative: list[int] = field(default_factory=list)


def _alive_positions(snap: SlotSnapshot) -> set[int]:
    return {int(round(p)) for p, is_c in zip(snap.positions, snap.is_centroid) if not is_c}


class LaneTelemetryState:
    """Per-lane tracker, seeded fresh for every new run.

    `uses_clustering` decides where a position that stops being alive gets
    attributed - it must match the real flag the lane's cache was actually
    built with (`src/caches/__init__.py`'s `METHODS` dict), not be guessed.
    """

    def __init__(self, uses_clustering: bool) -> None:
        self.uses_clustering = uses_clustering
        self.alive: set[int] = set()
        self.folded: set[int] = set()
        self.evicted: set[int] = set()

    def observe(self, new_segment_positions: range, new_snapshot: SlotSnapshot) -> Diff:
        """Call once after every forward step (whether it added one decode
        token or a whole multi-token passage/question segment).

        `new_segment_positions` is exactly the positions this call added
        that didn't exist a moment ago - required even though they'll
        usually just show up "alive" in `new_snapshot`, because a segment
        can compress *within its own forward call* (segment longer than
        remaining budget headroom): some of its own brand-new tokens can be
        folded or evicted before this function ever sees them as "alive."
        Without including them here, that would look like they were simply
        never seen, not like they were dropped.
        """
        candidates = self.alive | set(new_segment_positions)
        new_alive = _alive_positions(new_snapshot)
        newly_gone = candidates - new_alive
        self.alive = new_alive

        out = Diff()
        if self.uses_clustering:
            self.folded |= newly_gone
            out.newly_folded_positions = sorted(newly_gone)
        else:
            self.evicted |= newly_gone
            out.evicted_positions = sorted(newly_gone)

        out.folded_positions = sorted(self.folded)
        out.evicted_positions_cumulative = sorted(self.evicted)
        return out
