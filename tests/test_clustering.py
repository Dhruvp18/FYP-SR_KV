"""Phase 3 exit tests for `src/clustering.py` and `src/rope_positions.py`."""

from __future__ import annotations

import pytest
import torch

from src.clustering import CLUSTER_MODES, cluster_and_merge, cosine_threshold_labels
from src.rope_positions import POSITION_MODES, RopeHelper, assign_cluster_positions


def _synthetic_clusters(g=2, k=4, per_cluster=10, d=16, spread=0.05, seed=0):
    """k well-separated blobs per group, in temporal order."""
    torch.manual_seed(seed)
    centres = torch.randn(g, k, d) * 3.0
    members = centres.unsqueeze(2) + torch.randn(g, k, per_cluster, d) * spread
    keys = members.reshape(g, k * per_cluster, d)
    positions = torch.arange(k * per_cluster, dtype=torch.float32).expand(g, k * per_cluster)
    return keys, positions.contiguous(), centres


@pytest.mark.parametrize("mode", CLUSTER_MODES)
def test_centroid_lies_close_to_the_mean_of_its_members(mode):
    """The merged vector must sit inside the cloud it summarises, not off in space."""
    g, k, per_cluster, d = 2, 4, 10, 16
    keys, positions, centres = _synthetic_clusters(g, k, per_cluster, d)
    values = keys.clone()
    weights = torch.ones(g, k * per_cluster)
    slot_weights = torch.ones(g, k * per_cluster)

    ck, cv, cpos, cw = cluster_and_merge(
        keys, values, positions, weights, slot_weights, k, mode=mode,
        rope_position_mode="attn_weighted",
    )

    # every centroid must be near one of the true blob centres
    dists = torch.cdist(ck.float(), centres.float())  # [g, k, k]
    assert dists.min(dim=-1).values.max() < 0.5, "centroid landed far from every blob"

    # and each centroid is a convex combination of its members: it must lie
    # within the members' coordinatewise bounding box
    lo = keys.amin(dim=1, keepdim=True)
    hi = keys.amax(dim=1, keepdim=True)
    assert torch.all(ck >= lo - 1e-4) and torch.all(ck <= hi + 1e-4)


def test_merge_conserves_slot_weight_and_emits_exactly_k_centroids():
    g, k, per_cluster, d = 2, 5, 8, 16
    n = k * per_cluster
    keys, positions, _ = _synthetic_clusters(g, k, per_cluster, d)
    slot_weights = torch.full((g, n), 3.0)  # pretend each slot already stands for 3 tokens

    ck, cv, cpos, cw = cluster_and_merge(
        keys, keys.clone(), positions, torch.rand(g, n), slot_weights, k,
        rope_position_mode="attn_weighted",
    )
    assert ck.shape == (g, k, d) and cpos.shape == (g, k)
    assert torch.allclose(cw.sum(dim=-1), slot_weights.sum(dim=-1))


def test_every_cluster_is_non_empty_even_on_degenerate_input():
    """Identical keys are the worst case for k-means: all mass on one centroid."""
    g, n, d, k = 2, 20, 8, 5
    keys = torch.ones(g, n, d)
    positions = torch.arange(n, dtype=torch.float32).expand(g, n).contiguous()
    ck, cv, cpos, cw = cluster_and_merge(
        keys, keys.clone(), positions, torch.ones(g, n), torch.ones(g, n), k,
        rope_position_mode="attn_weighted",
    )
    assert torch.isfinite(ck).all() and torch.isfinite(cpos).all()
    assert torch.allclose(cw.sum(dim=-1), torch.full((g,), float(n)))
    assert (cw > 0).all(), "an empty cluster would mean a wasted slot and a 0/0 centroid"


@pytest.mark.parametrize("mode", POSITION_MODES)
def test_cluster_positions_stay_within_member_range(mode):
    g, k, per_cluster, d = 2, 4, 10, 16
    n = k * per_cluster
    keys, positions, _ = _synthetic_clusters(g, k, per_cluster, d)
    _, _, cpos, _ = cluster_and_merge(
        keys, keys.clone(), positions, torch.rand(g, n), torch.ones(g, n), k,
        rope_position_mode=mode,
    )
    assert torch.all(cpos >= positions.min()) and torch.all(cpos <= positions.max())
    assert torch.isfinite(cpos).all()


def test_position_modes_pick_the_documented_member():
    pos = torch.tensor([[[10.0, 20.0, 30.0]]])       # 1 group, 1 cluster, 3 members
    w = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, True]]])
    assert assign_cluster_positions("latest", pos, w, mask).item() == 30.0
    assert assign_cluster_positions("earliest", pos, w, mask).item() == 10.0
    assert assign_cluster_positions("attn_weighted", pos, w, mask).item() == 10.0

    w2 = torch.tensor([[[1.0, 1.0, 2.0]]])
    assert assign_cluster_positions("attn_weighted", pos, w2, mask).item() == pytest.approx(22.5)


def test_masked_out_members_are_ignored_by_position_assignment():
    pos = torch.tensor([[[10.0, 20.0, 30.0]]])
    w = torch.ones(1, 1, 3)
    mask = torch.tensor([[[True, True, False]]])
    assert assign_cluster_positions("latest", pos, w, mask).item() == 20.0
    assert assign_cluster_positions("attn_weighted", pos, w, mask).item() == pytest.approx(15.0)


def test_rope_rotation_by_delta_matches_rotating_from_scratch():
    """The identity centroid re-positioning depends on (CLAUDE.md A5)."""
    helper = RopeHelper.from_config(head_dim=32, rope_theta=10000.0)
    x = torch.randn(1, 2, 5, 32)
    p = torch.full((1, 2, 5), 7.0)
    target = torch.full((1, 2, 5), 41.0)

    direct = helper.rotate(x, target)                      # rotate raw -> 41
    via_delta = helper.rotate(helper.rotate(x, p), target - p)  # raw -> 7 -> 41
    assert torch.allclose(direct, via_delta, atol=1e-5)

    roundtrip = helper.rotate_to(helper.unrotate_to_zero(helper.rotate(x, p), p), p)
    assert torch.allclose(roundtrip, helper.rotate(x, p), atol=1e-5)


def test_rope_helper_rejects_unsupported_scaling():
    class FakeCfg:
        rope_scaling = {"rope_type": "dynamic"}

        def get_text_config(self):
            return self

    class FakeRotary:
        inv_freq = torch.ones(8)

    class FakeInner:
        rotary_emb = FakeRotary()

    class FakeModel:
        model = FakeInner()
        config = FakeCfg()

    with pytest.raises(NotImplementedError, match="dynamic"):
        RopeHelper.from_model(FakeModel())


def test_cosine_threshold_is_analysis_only_and_yields_variable_k():
    """Documents why the threshold rule cannot drive the cache (CLAUDE.md A1)."""
    keys, _, _ = _synthetic_clusters(g=1, k=4, per_cluster=10, d=16)
    tight = cosine_threshold_labels(keys, threshold=0.99).max().item() + 1
    loose = cosine_threshold_labels(keys, threshold=0.10).max().item() + 1
    assert tight != loose, "cluster count varies with the data, hence not usable as a fixed budget"
