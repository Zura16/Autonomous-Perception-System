"""Closed-form tests for the monocular range estimators and their sensitivities.

The whole project rests on `D = f*H/h`. Every case here inverts a projection
whose answer is known exactly, so a sign error or a frame mix-up fails loudly
rather than shifting a curve by a few percent.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.geometry import (
    CameraModel,
    CredibleEnvelope,
    box_is_clipped_at_bottom,
    crossover_range_m,
    horizon_row_px,
    min_supportable_range_m,
    pitch_range_error_frac,
    prior_range_error_frac,
    range_contact_point,
    range_size_prior,
)
from tests.test_calibration import CY, FY, make_calib

H_CAM = 1.655


def make_camera(
    height_m: float = H_CAM,
    pitch_deg: float = 0.0,
    max_range_m: float = 50.0,
    min_rows_from_bottom: int = 2,
) -> CameraModel:
    return CameraModel(
        height_m=height_m,
        height_std_m=0.027,
        pitch_deg=pitch_deg,
        priors={
            "vehicle": {"height_m": 1.585, "height_std_m": 0.390},
            "vru": {"height_m": 1.700, "height_std_m": 0.147},
        },
        min_rows_from_bottom=min_rows_from_bottom,
        min_pixels_below_horizon=3.0,
        max_range_m=max_range_m,
    )


def contact_row_for(range_m: float, height_m: float = H_CAM) -> float:
    """Image row where a ground point at `range_m` lands, for a level camera."""
    return CY + FY * height_m / range_m


# ── horizon ──────────────────────────────────────────────────────────────────


def test_level_camera_horizon_is_the_principal_row():
    assert horizon_row_px(make_calib(baseline_m=0.0), 0.0) == pytest.approx(CY)


def test_nose_down_pitch_raises_the_horizon_in_the_image():
    """Rows increase downward, so a higher horizon is a SMALLER row value."""
    calib = make_calib(baseline_m=0.0)
    assert horizon_row_px(calib, 1.0) < horizon_row_px(calib, 0.0)
    assert horizon_row_px(calib, 1.0) == pytest.approx(CY - FY * np.tan(np.radians(1.0)))


# ── contact point ────────────────────────────────────────────────────────────


# Below ~6 m the contact row falls outside a 375-row image, so those cases are
# legitimately clipped rather than measurable -- see the clipped-box tests.
#
# Each case runs against a camera whose declared support comfortably contains
# it. The algebra inverts exactly at any range; whether the estimator is ALLOWED
# to answer there is a separate question, tested on its own below. Folding the
# two together would let a policy change look like a broken projection, and
# putting a case exactly ON the support boundary would test nothing but the
# rounding of a float comparison.
@pytest.mark.parametrize("truth", [7.0, 10.0, 25.0, 50.0, 80.0])
def test_contact_point_inverts_the_projection_exactly(truth):
    calib = make_calib(baseline_m=0.0)
    camera = make_camera(max_range_m=1.1 * truth)
    box = np.array([100.0, 50.0, 200.0, contact_row_for(truth)])
    assert range_contact_point(box, calib, camera) == pytest.approx(truth)


def test_contact_point_abstains_above_the_horizon():
    """A box bottom at or above the horizon makes the relation diverge.

    Returning a very large range instead would be a confident answer to an
    ill-posed question.
    """
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    assert np.isnan(range_contact_point(np.array([0.0, 0.0, 10.0, CY - 5]), calib, camera))
    assert np.isnan(range_contact_point(np.array([0.0, 0.0, 10.0, CY]), calib, camera))


def test_contact_point_abstains_beyond_its_characterised_range():
    """Past `max_range_m` the estimator extrapolates beyond all of its evidence.

    This is the D-025 defect, pinned. A box bottom just above the 50 m contact
    row used to return a number -- up to 369 m on the test split -- because the
    only guard was a 3 px numerical floor. At 50 m the denominator is already
    down to 23.9 px, where one pixel of box-edge error costs 4.2% of the range.
    """
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    inside = np.array([0.0, 0.0, 10.0, contact_row_for(49.0)])
    outside = np.array([0.0, 0.0, 10.0, contact_row_for(51.0)])
    assert range_contact_point(inside, calib, camera) == pytest.approx(49.0)
    assert np.isnan(range_contact_point(outside, calib, camera))


def test_the_support_gate_binds_before_the_numerical_floor():
    """The two guards are not interchangeable, and the wider one must win.

    A 3 px denominator is finite arithmetic and a 398 m answer, which is why the
    numerical floor alone shipped the defect. This pins that a box the numerical
    floor would happily accept is still refused for being out of support.
    """
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    below = 5.0  # comfortably past the 3 px numerical floor
    box = np.array([0.0, 0.0, 10.0, CY + below])
    assert below > camera.min_pixels_below_horizon
    assert FY * H_CAM / below > camera.max_range_m  # ~239 m, the answer it used to give
    assert np.isnan(range_contact_point(box, calib, camera))


def test_contact_point_is_independent_of_box_height():
    """It reads only the bottom edge -- that is what makes it prior-free."""
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    bottom = contact_row_for(20.0)
    tall = np.array([0.0, 10.0, 50.0, bottom])
    short = np.array([0.0, bottom - 5, 50.0, bottom])
    assert range_contact_point(tall, calib, camera) == pytest.approx(
        range_contact_point(short, calib, camera)
    )


def test_contact_point_scales_with_assumed_camera_height():
    """A 10% error in the mounting height is a 10% error in range."""
    calib = make_calib(baseline_m=0.0)
    box = np.array([0.0, 0.0, 10.0, contact_row_for(30.0)])
    a = range_contact_point(box, calib, make_camera(height_m=H_CAM))
    b = range_contact_point(box, calib, make_camera(height_m=H_CAM * 1.1))
    assert b / a == pytest.approx(1.1)


def test_assuming_zero_pitch_when_the_camera_is_pitched_matches_the_budget_formula():
    """The exact relation: D_est = D / (1 - x), x = tan(theta)*D/h_cam.

    The single most load-bearing formula in Phase 3. This test is what caught
    the linearised `dD/D ~ x` being used as if it were exact, which understates
    the pitch cost by a third at 50 m (docs/decisions.md D-016).
    """
    calib = make_calib(baseline_m=0.0)
    truth, theta = 30.0, 0.7

    # The true image row of a ground point at `truth`, with the camera pitched.
    horizon = horizon_row_px(calib, theta)
    bottom = horizon + FY * H_CAM / truth
    box = np.array([0.0, 0.0, 10.0, bottom])

    naive = range_contact_point(box, calib, make_camera(pitch_deg=0.0))
    predicted = truth * (1 + pitch_range_error_frac(theta, truth, H_CAM))
    assert naive == pytest.approx(predicted, rel=1e-6)
    assert naive > truth  # nose-down pitch makes objects look further away

    # The linearised form understates this materially -- the reason the exact
    # relation is used everywhere. At 30 m and 0.7 deg it is off by ~5%.
    linear = truth * (1 + np.tan(np.radians(theta)) * truth / H_CAM)
    assert linear < naive * 0.97


def test_knowing_the_pitch_recovers_the_true_range():
    """Confirms the error above is the assumption's cost, not a coding error."""
    calib = make_calib(baseline_m=0.0)
    truth, theta = 30.0, 0.7
    bottom = horizon_row_px(calib, theta) + FY * H_CAM / truth
    box = np.array([0.0, 0.0, 10.0, bottom])
    assert range_contact_point(box, calib, make_camera(pitch_deg=theta)) == pytest.approx(truth)


# ── size prior ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("truth", [5.0, 20.0, 60.0])
def test_size_prior_inverts_the_projection_when_the_prior_is_right(truth):
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    height_px = FY * camera.prior_height_m("vehicle") / truth
    box = np.array([0.0, 100.0, 50.0, 100.0 + height_px])
    assert range_size_prior(box, calib, camera, "vehicle") == pytest.approx(truth)


def test_size_prior_is_completely_pitch_insensitive():
    """No ground plane appears in D = f*H/h, so pitch cannot enter.

    This orthogonality is the entire reason both estimators are kept.
    """
    calib = make_calib(baseline_m=0.0)
    box = np.array([0.0, 100.0, 50.0, 140.0])
    flat = range_size_prior(box, calib, make_camera(pitch_deg=0.0), "vehicle")
    tilted = range_size_prior(box, calib, make_camera(pitch_deg=2.0), "vehicle")
    assert flat == pytest.approx(tilted)


def test_size_prior_error_transfers_one_for_one_and_does_not_grow_with_range():
    """A 10% prior error is a 10% range error at 10 m and at 60 m alike."""
    calib = make_calib(baseline_m=0.0)
    right, wrong = make_camera(), make_camera()
    wrong.priors["vehicle"]["height_m"] = 1.585 * 1.1
    for truth in (10.0, 60.0):
        height_px = FY * 1.585 / truth
        box = np.array([0.0, 0.0, 50.0, height_px])
        a = range_size_prior(box, calib, right, "vehicle")
        b = range_size_prior(box, calib, wrong, "vehicle")
        assert b / a == pytest.approx(1.1)


def test_size_prior_abstains_on_a_degenerate_box():
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    assert np.isnan(range_size_prior(np.array([0.0, 50.0, 10.0, 50.0]), calib, camera, "vehicle"))


def test_vru_and_vehicle_priors_give_different_ranges_for_the_same_box():
    """Why the coarse group split is kept despite class-agnostic detection."""
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    box = np.array([0.0, 100.0, 50.0, 140.0])
    assert range_size_prior(box, calib, camera, "vru") > range_size_prior(
        box, calib, camera, "vehicle"
    )


# ── clipped boxes ────────────────────────────────────────────────────────────


def test_box_clipped_at_the_image_bottom_is_detected():
    """375-row image, margin 2 -> a bottom edge at 373 or beyond is clipped."""
    calib, camera = make_calib(), make_camera()
    assert box_is_clipped_at_bottom(np.array([0.0, 0.0, 10.0, 374.0]), calib, camera)
    assert box_is_clipped_at_bottom(np.array([0.0, 0.0, 10.0, 373.0]), calib, camera)
    assert not box_is_clipped_at_bottom(np.array([0.0, 0.0, 10.0, 372.0]), calib, camera)


def test_both_estimators_abstain_on_a_clipped_box():
    """The object continues below the image, so neither its contact point nor
    its pixel height is observable.

    Measured on val: clipped boxes are 4.2% of detections and carry 46.4% MAPE
    against 13.6% for clear ones, dominating the near-field error (D-017).
    """
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    clipped = np.array([0.0, 200.0, 50.0, 374.0])
    assert np.isnan(range_contact_point(clipped, calib, camera))
    assert np.isnan(range_size_prior(clipped, calib, camera, "vehicle"))


def test_an_unclipped_box_one_pixel_higher_is_still_measured():
    """The gate must not swallow boxes that merely sit low in the frame."""
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    ok = np.array([0.0, 200.0, 50.0, 372.0])
    assert np.isfinite(range_contact_point(ok, calib, camera))
    assert np.isfinite(range_size_prior(ok, calib, camera, "vehicle"))


# ── sensitivity relations ────────────────────────────────────────────────────


def test_pitch_error_grows_faster_than_linearly_with_range():
    """x/(1-x) is superlinear: doubling the range MORE than doubles the error."""
    a = pitch_range_error_frac(0.5, 20.0, H_CAM)
    b = pitch_range_error_frac(0.5, 40.0, H_CAM)
    assert b > 2 * a
    assert b == pytest.approx(2 * a, rel=0.25)  # still roughly linear at small x


def test_pitch_error_diverges_where_the_contact_point_reaches_the_horizon():
    """At D = h/tan(theta) the relation is undefined, and the estimator abstains."""
    diverge = H_CAM / np.tan(np.radians(0.715))
    assert pitch_range_error_frac(0.715, diverge * 1.01, H_CAM) == float("inf")
    assert np.isfinite(pitch_range_error_frac(0.715, diverge * 0.5, H_CAM))


def test_prior_error_is_flat_in_range():
    assert prior_range_error_frac(0.2, 1.585) == pytest.approx(0.2 / 1.585)


def test_crossover_matches_the_published_prediction():
    """Vehicles: 20.1% pooled spread at typical 0.319 deg -> ~50 m (exact form).

    The linearised form would say 59.7 m, putting the crossover outside the
    0-50 m evidence envelope when it actually sits just inside it.
    """
    assert crossover_range_m(0.201, 0.319, H_CAM) == pytest.approx(49.8, abs=1.0)
    assert crossover_range_m(0.087, 0.319, H_CAM) == pytest.approx(23.8, abs=1.0)


def test_crossover_is_infinite_without_pitch_error():
    """With perfect pitch the contact-point estimator never loses to the prior."""
    assert crossover_range_m(0.201, 0.0, H_CAM) == float("inf")


# ── credible envelope ────────────────────────────────────────────────────────


def test_the_envelope_gates_both_ends_not_just_the_far_one():
    """The near field is OUTSIDE the band, and was being published anyway.

    The HUD checked only the upper bound, so 0-20 m range values -- the worst
    the project measures, 21-22% MAPE on both splits -- were printed as vouched
    while 30-50 m values, among the best, were dimmed away.
    """
    env = CredibleEnvelope(min_range_m=20.0, max_range_m=50.0, max_mape_pct=13.0, gates_ttc=False)
    assert not env.contains(8.0)
    assert not env.contains(19.9)
    assert env.contains(20.0)
    assert env.contains(35.0)
    assert env.contains(50.0)
    assert not env.contains(50.1)


def test_the_envelope_abstains_on_a_missing_range():
    """An abstention is not 'outside the band' -- both must render blank."""
    env = CredibleEnvelope(min_range_m=20.0, max_range_m=50.0, max_mape_pct=13.0, gates_ttc=False)
    assert not env.contains(float("nan"))
    assert not env.contains(float("inf"))


def test_the_shipped_envelope_matches_what_both_splits_support():
    """Pins the configured band to the measurement that justifies it.

    20-50 m is where val AND held-out test both stay within 13% MAPE (Rows 3.1,
    9.7). The previous 10-30 m was declared on val alone and did not survive
    (17.2% at 10-20 m on test), so this guards against a quiet reversion to the
    friendlier number.
    """
    env = CredibleEnvelope.from_config()
    assert (env.min_range_m, env.max_range_m) == (20.0, 50.0)
    assert env.max_mape_pct == 13.0
    assert env.gates_ttc is False, "TTC is measured separately and is not gated on range error"


# ── the near-field limit (D-026) ─────────────────────────────────────────────


def test_min_supportable_range_is_where_the_contact_point_leaves_the_image():
    """Below D_min the ground contact point is not in the picture at all.

    A contact point at range D lands at row `horizon + fy*h/D`, so it exits an
    H-row image at `D_min = fy*h/(H - horizon)`. This is a property of the
    mounting and the sensor, not of the estimator: a camera 1.655 m up cannot
    see the wheels of a car four metres ahead, and no better algorithm changes
    that.
    """
    calib, camera = make_calib(baseline_m=0.0), make_camera()
    d_min = min_supportable_range_m(calib, camera)
    img_h = calib.image_size[2][1]
    assert d_min == pytest.approx(FY * H_CAM / (img_h - CY))
    # the contact row at D_min is exactly the last row of the image
    assert contact_row_for(d_min) == pytest.approx(img_h)
    # and anything closer projects below it
    assert contact_row_for(d_min * 0.8) > img_h


def test_the_clip_margin_catches_boxes_that_stop_short_of_the_edge():
    """The D-026 defect, pinned: a 2-row margin missed the objects that matter.

    Boxes for sub-D_min objects sit around y2 = 371 of 375 -- four rows clear of
    the edge, so a 2-row margin passed them, and the estimator then saturated
    and over-estimated 100% of the time.
    """
    calib = make_calib(baseline_m=0.0)
    img_h = calib.image_size[2][1]
    box = np.array([0.0, 100.0, 50.0, img_h - 4])  # the observed failure geometry

    old = make_camera(min_rows_from_bottom=2)
    assert not box_is_clipped_at_bottom(box, calib, old)
    assert np.isfinite(range_contact_point(box, calib, old)), "the defect: it answered"

    new = make_camera(min_rows_from_bottom=10)
    assert box_is_clipped_at_bottom(box, calib, new)
    assert np.isnan(range_contact_point(box, calib, new))
    assert np.isnan(range_size_prior(box, calib, new, "vehicle"))


def test_the_shipped_clip_margin_is_the_one_val_supports():
    """Pins the configured margin to the sweep that justifies it.

    2 rows shipped the defect; 10 was chosen on val, where it catches 83% of
    unsupportable boxes for 0.34% of valid ones. Guards against a quiet revert.
    """
    assert CameraModel.from_config().min_rows_from_bottom == 10


def test_a_box_well_clear_of_the_edge_is_still_measured():
    """The gate must not eat the valid population it sits next to.

    Valid objects sit a median 140 px from the edge; the gate costs 0.34% of
    them. A margin that also swallowed ordinary near boxes would trade one
    silent error for a silent abstention.
    """
    calib = make_calib(baseline_m=0.0)
    camera = make_camera(min_rows_from_bottom=10)
    box = np.array([0.0, 100.0, 50.0, contact_row_for(12.0)])
    assert not box_is_clipped_at_bottom(box, calib, camera)
    assert range_contact_point(box, calib, camera) == pytest.approx(12.0)
