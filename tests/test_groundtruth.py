"""Tests for the range ground-truth estimators.

Built on synthetic point sets where the right answer is known by construction,
including the adversarial cases the dev-split measurements exposed: an occluder
inside the box, background seen past the silhouette, and a starved sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.groundtruth import (
    GT_INVALID,
    GT_RELAXED,
    GT_STRICT,
    MIN_POINTS_DEFAULT,
    estimate_range,
    gt_tier,
    nearest_cluster,
    points_in_box,
)
from aps.kitti.tracklets import (
    OCC_FULLY,
    OCC_PARTLY,
    OCC_UNSET,
    OCC_VISIBLE,
    TRUNC_IN_IMAGE,
    TRUNC_TRUNCATED,
)

BOX = np.array([100.0, 100.0, 200.0, 200.0])


def scatter(n: int, depth: float, box=BOX, inset: float = 0.05, seed: int = 0):
    """n points spread evenly inside `box`, all at one depth."""
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * inset, (y2 - y1) * inset
    uv = np.column_stack([rng.uniform(x1 + dx, x2 - dx, n), rng.uniform(y1 + dy, y2 - dy, n)])
    return uv, np.full(n, depth)


# ── points_in_box ────────────────────────────────────────────────────────────


def test_points_in_box_selects_only_interior():
    uv = np.array([[150.0, 150.0], [50.0, 150.0], [150.0, 250.0], [100.0, 100.0]])
    depth = np.full(4, 10.0)
    assert points_in_box(uv, depth, BOX).tolist() == [True, False, False, True]


def test_points_in_box_excludes_nonpositive_depth():
    uv = np.array([[150.0, 150.0], [150.0, 150.0]])
    depth = np.array([10.0, -10.0])
    assert points_in_box(uv, depth, BOX).tolist() == [True, False]


def test_shrink_pulls_every_edge_inward():
    """0.25 keeps the central half: 125..175 of a 100..200 box."""
    uv = np.array([[110.0, 150.0], [150.0, 150.0], [190.0, 150.0]])
    depth = np.full(3, 10.0)
    assert points_in_box(uv, depth, BOX, shrink=0.25).tolist() == [False, True, False]


# ── nearest_cluster ──────────────────────────────────────────────────────────


def test_nearest_cluster_returns_the_near_group():
    depths = np.concatenate([np.full(10, 20.0), np.full(10, 40.0)])
    assert nearest_cluster(depths, gap_m=2.0, min_points=4) == pytest.approx(np.full(10, 20.0))


def test_nearest_cluster_skips_unsupported_foreground():
    """A lone stray return in front must not become the answer.

    This is the difference between the cluster estimator and a fancy minimum.
    """
    depths = np.concatenate([[5.0], np.full(10, 30.0)])
    assert nearest_cluster(depths, gap_m=2.0, min_points=4) == pytest.approx(np.full(10, 30.0))


def test_nearest_cluster_keeps_a_supported_occluder():
    """With enough support the occluder IS the nearest cluster -- by design.

    Documents the failure mode measured on real data: a genuinely occluded box
    yields a confident measurement of the occluder. The fix is the validity
    gate, not the estimator.
    """
    depths = np.concatenate([np.full(8, 6.0), np.full(30, 30.0)])
    assert nearest_cluster(depths, gap_m=2.0, min_points=4).mean() == pytest.approx(6.0)


def test_nearest_cluster_handles_empty():
    assert len(nearest_cluster(np.array([]))) == 0


# ── estimate_range ───────────────────────────────────────────────────────────


def test_flat_surface_recovers_its_depth():
    uv, depth = scatter(200, 25.0)
    for est in ("min", "p10", "p20", "median", "shrink_p20", "cluster_p20"):
        gt = estimate_range(uv, depth, BOX, estimator=est)
        assert gt.range_m == pytest.approx(25.0), est
        assert gt.is_valid


def test_abstains_rather_than_guessing_when_starved():
    uv, depth = scatter(MIN_POINTS_DEFAULT - 1, 25.0)
    gt = estimate_range(uv, depth, BOX, estimator="p20")
    assert not gt.is_valid
    assert np.isnan(gt.range_m)


def test_shrink_estimator_abstains_instead_of_widening():
    """The D-011 negative result, locked in as a test.

    Points live only in the box's border ring. A fallback to a smaller shrink
    would 'recover' this box by measuring exactly the contaminated returns that
    made it 12x worse on real data. Abstention is the required behaviour.
    """
    ring = np.array([[105.0, 105.0]] * 40)
    depth = np.full(40, 25.0)
    assert not estimate_range(ring, depth, BOX, estimator="shrink_p20").is_valid
    assert estimate_range(ring, depth, BOX, estimator="p20").is_valid


def test_shrink_rejects_silhouette_background():
    """Object in the middle, background at the border -- the real contamination."""
    obj_uv, obj_d = scatter(60, 30.0, inset=0.35, seed=1)
    bg_uv, bg_d = scatter(60, 80.0, inset=0.0, seed=2)
    # Force the background points onto the border ring.
    bg_uv[:, 0] = np.where(bg_uv[:, 0] < 150, 102.0, 198.0)
    uv = np.vstack([obj_uv, bg_uv])
    depth = np.concatenate([obj_d, bg_d])

    assert estimate_range(uv, depth, BOX, estimator="shrink_p20").range_m == pytest.approx(30.0)
    # The unshrunk median is dragged toward the background it should have excluded.
    assert estimate_range(uv, depth, BOX, estimator="median").range_m > 40.0


def test_median_is_biased_far_on_a_body_with_depth_extent():
    """Why D-003 picks near-face: median reports the object's middle."""
    depth = np.linspace(20.0, 24.0, 100)  # a 4 m long car, near face at 20 m
    uv, _ = scatter(100, 0.0, inset=0.35)
    assert estimate_range(uv, depth, BOX, estimator="median").range_m == pytest.approx(
        22.0, abs=0.1
    )
    assert estimate_range(uv, depth, BOX, estimator="p20").range_m == pytest.approx(20.8, abs=0.2)


def test_spread_reports_internal_disagreement():
    uv, _ = scatter(200, 0.0, inset=0.35)
    flat = estimate_range(uv, np.full(200, 25.0), BOX, estimator="p20")
    deep = estimate_range(uv, np.linspace(20.0, 30.0, 200), BOX, estimator="p20")
    assert flat.spread_m == pytest.approx(0.0, abs=1e-9)
    assert deep.spread_m == pytest.approx(5.0, abs=0.2)


def test_unknown_estimator_raises():
    uv, depth = scatter(50, 25.0)
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_range(uv, depth, BOX, estimator="vibes")


# ── validity gate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("occlusion", "truncation", "expected"),
    [
        (OCC_VISIBLE, TRUNC_IN_IMAGE, GT_STRICT),
        (OCC_UNSET, TRUNC_IN_IMAGE, GT_RELAXED),
        (OCC_PARTLY, TRUNC_IN_IMAGE, GT_INVALID),
        (OCC_FULLY, TRUNC_IN_IMAGE, GT_INVALID),
        # Truncation invalidates regardless of how visible the object is: the
        # label's near face may lie outside the image entirely.
        (OCC_VISIBLE, TRUNC_TRUNCATED, GT_INVALID),
        (OCC_UNSET, TRUNC_TRUNCATED, GT_INVALID),
    ],
)
def test_gt_tier(occlusion, truncation, expected):
    assert gt_tier(occlusion, truncation) == expected
