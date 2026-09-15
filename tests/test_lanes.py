"""Lane geometry and the lane solver, on synthetic roads with known answers.

A road is rendered by painting markings at chosen lateral positions through the
same camera model the solver inverts. The solver must recover those positions in
metres; the ground-truth extractor must recover them from a filled lane mask.
Neither test uses the labelled benchmark, so neither can overfit it.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.kitti.road import RoadGeometry
from aps.lanes import (
    BevGrid,
    Boundary,
    LaneConfig,
    estimate_lane,
    lane_from_mask,
    marking_mask,
    summarise,
    warp_to_bev,
)

P2 = np.array(
    [
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884],
    ]
)
H, W = 375, 1242


@pytest.fixture(scope="module")
def cfg() -> LaneConfig:
    return LaneConfig.from_config()  # the committed, frozen parameters


@pytest.fixture(scope="module")
def geom(cfg) -> RoadGeometry:
    return RoadGeometry.nominal(P2, cfg.camera_height_m)


def render_road(geom: RoadGeometry, lines_x_m, width_m: float = 0.15) -> np.ndarray:
    """Grey asphalt with white markings at the given lateral positions."""
    img = np.full((H, W, 3), 90, dtype=np.uint8)
    zs = np.arange(5.0, 45.0, 0.02)
    for x0 in lines_x_m:
        for dx in np.arange(-width_m / 2, width_m / 2 + 1e-9, 0.01):
            u, v = geom.road_to_pixel(np.full_like(zs, x0 + dx), zs)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            img[v[ok].astype(int), u[ok].astype(int)] = 235
    return img


def render_lane_mask(geom: RoadGeometry, left_m: float, right_m: float) -> np.ndarray:
    """A filled ego-lane label between two lateral positions."""
    ego = np.zeros((H, W), dtype=bool)
    for v in range(H):
        p = geom.pixel_to_road(W / 2, v)
        if p is None:
            continue
        _, z = p
        ul, _ = geom.road_to_pixel(np.array(left_m), np.array(z))
        ur, _ = geom.road_to_pixel(np.array(right_m), np.array(z))
        lo, hi = int(np.clip(ul, 0, W - 1)), int(np.clip(ur, 0, W - 1))
        ego[v, lo : hi + 1] = True
    return ego


# ── geometry ─────────────────────────────────────────────────────────────────


def test_nominal_geometry_puts_the_road_at_the_camera_height(geom, cfg):
    assert geom.camera_height_m == pytest.approx(cfg.camera_height_m)


@pytest.mark.parametrize(("x", "z"), [(0.0, 10.0), (-1.7, 15.0), (2.3, 32.0)])
def test_pixel_to_road_inverts_road_to_pixel(geom, x, z):
    u, v = geom.road_to_pixel(np.array(x), np.array(z))
    back = geom.pixel_to_road(float(u), float(v))
    assert back is not None
    assert back == pytest.approx((x, z), abs=1e-6)


def test_a_road_point_ahead_projects_below_the_horizon(geom):
    """v = cy + f·h/z: nearer road is lower in the image."""
    _, v_near = geom.road_to_pixel(np.array(0.0), np.array(8.0))
    _, v_far = geom.road_to_pixel(np.array(0.0), np.array(30.0))
    assert v_near > v_far > 172.854


def test_a_ray_above_the_horizon_never_meets_the_road(geom):
    assert geom.pixel_to_road(620.0, 100.0) is None


# ── marking extraction ───────────────────────────────────────────────────────


def test_tophat_keeps_a_thin_stripe_and_rejects_a_broad_bright_region(cfg):
    """The reason for a top-hat over a brightness threshold: a sunlit patch or a
    car body is broad, a lane marking is thin."""
    bev = np.full((200, 320, 3), 90, dtype=np.uint8)
    bev[:, 100:103] = 235  # 0.15 m stripe at 0.05 m/px
    bev[:, 200:260] = 235  # 3 m wide bright region
    mask = marking_mask(bev, cfg)
    assert mask[:, 100:103].mean() > 0.9
    assert mask[:, 215:245].mean() < 0.05


# ── the solver ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("left", "right"), [(-1.75, 1.75), (-1.2, 2.3), (-2.1, 1.4)])
def test_solver_recovers_lane_position_and_width_in_metres(geom, cfg, left, right):
    img = render_road(geom, [left, right])
    est, _ = estimate_lane(img, geom, cfg)
    assert est.detected
    assert est.width_m == pytest.approx(right - left, abs=0.10)
    assert est.centre_offset_m == pytest.approx((left + right) / 2, abs=0.08)


def test_solver_reports_no_lane_on_an_unmarked_road(geom, cfg):
    est, _ = estimate_lane(np.full((H, W, 3), 90, dtype=np.uint8), geom, cfg)
    assert not est.detected
    assert np.isnan(est.centre_offset_m)


def test_solver_ignores_the_next_lane_over(geom, cfg):
    """A third marking 3.5 m further out must not replace the ego boundary."""
    img = render_road(geom, [-1.75, 1.75, 5.25])
    est, _ = estimate_lane(img, geom, cfg)
    assert est.width_m == pytest.approx(3.5, abs=0.10)


def test_wrong_camera_height_scales_the_lateral_answer(cfg):
    """Why nominal-vs-oracle is measured: a height error is a scale error on
    every lateral metre. Rendered with the true height, solved with a 5% wrong one."""
    truth = RoadGeometry.nominal(P2, 1.60)
    wrong = RoadGeometry.nominal(P2, 1.60 * 1.05)
    img = render_road(truth, [-1.75, 1.75])
    est, _ = estimate_lane(img, wrong, cfg)
    assert est.width_m == pytest.approx(3.5 * 1.05, rel=0.03)


def test_bev_is_empty_outside_the_image(geom, cfg):
    grid = BevGrid.from_config(cfg)
    bev = warp_to_bev(np.full((H, W, 3), 200, dtype=np.uint8), geom, grid)
    # x = -8 m is 53 deg off-axis at 6 m ahead (outside the +/-40.7 deg half-FOV)
    # but only 11 deg off-axis at 40 m (inside). Row 0 is far, the last row near.
    assert bev[-1, 0].max() == 0
    assert bev[0, 0].max() == 200


# ── ground truth from a mask ─────────────────────────────────────────────────


@pytest.mark.parametrize(("left", "right"), [(-1.75, 1.75), (-1.3, 2.2)])
def test_ground_truth_extraction_recovers_the_labelled_lane(geom, cfg, left, right):
    est = lane_from_mask(render_lane_mask(geom, left, right), geom, cfg)
    assert est.detected
    assert est.width_m == pytest.approx(right - left, abs=0.05)
    assert est.centre_offset_m == pytest.approx((left + right) / 2, abs=0.05)


def test_ground_truth_skips_rows_where_the_lane_leaves_the_image(geom, cfg):
    """A lane clipped by the image border must not be measured AT the border,
    which would make every truncated lane look narrow."""
    ego = render_lane_mask(geom, -1.75, 1.75)
    ego[:, :400] = ego[:, :400] | (ego.any(axis=1)[:, None])  # smear left side to the edge
    est = lane_from_mask(ego, geom, cfg)
    assert est.right is not None
    if est.left is not None:
        assert est.width_m > 3.0


# ── departure ────────────────────────────────────────────────────────────────


def _flat(x: float) -> Boundary:
    return Boundary(np.array([0.0, 0.0, x]), 6.0, 40.0, 100)


def test_departure_warning_fires_when_the_vehicle_edge_nears_a_boundary(cfg):
    # Vehicle half-width 0.91 + margin 0.20 = 1.11 m of required clearance.
    assert summarise(_flat(-1.0), _flat(2.5), cfg).departure_warning
    assert not summarise(_flat(-1.75), _flat(1.75), cfg).departure_warning


def test_departure_is_undefined_without_both_boundaries(cfg):
    est = summarise(_flat(-1.75), None, cfg)
    assert not est.detected
    assert not est.departure_warning
