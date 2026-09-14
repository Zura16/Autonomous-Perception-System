"""Scale-rate closing speed and TTC, on sequences whose answer is known exactly.

The charter requires synthetic constant-velocity sequences for the KF and TTC
precisely because the correct answer is computable: an object at `D0` closing at
`v` has `TTC = D/v` at every frame, falling linearly to zero. Anything that
cannot reproduce that on noiseless input will not be trusted on real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.motion import (
    NOT_CLOSING,
    InverseHeightKF,
    MotionConfig,
    ScaleRateEstimator,
    ttc_from_state,
)

FY = 721.5377
DT = 0.10356  # the measured KITTI frame interval, not a nominal 0.1


def cfg(**over) -> MotionConfig:
    base = dict(
        rel_height_sigma=0.06,
        rel_height_sigma_far=0.13,
        far_range_px=35.0,
        rel_accel_sigma=0.25,
        deadband_inv_ttc=0.05,
        deadband_sigmas=2.0,
        min_updates=3,
    )
    base.update(over)
    return MotionConfig(**base)


def height_px(range_m: float, object_h_m: float = 1.6) -> float:
    """Apparent height of an object of known size at a known range."""
    return FY * object_h_m / range_m


def approach(d0: float, speed_mps: float, n: int, dt: float = DT, object_h_m: float = 1.6):
    """Constant-velocity approach. Yields (range_m, height_px) per frame."""
    for i in range(n):
        d = d0 - speed_mps * dt * i
        yield d, height_px(d, object_h_m)


# ── the algebra, before any filtering ────────────────────────────────────────


def test_ttc_is_positive_only_while_the_object_grows():
    assert ttc_from_state(u=0.01, u_dot=-0.001) == pytest.approx(10.0)
    assert ttc_from_state(u=0.01, u_dot=+0.001) == NOT_CLOSING  # receding
    assert ttc_from_state(u=0.01, u_dot=0.0) == NOT_CLOSING  # static


def test_ttc_is_free_of_focal_length_and_object_size():
    """f and H cancel in −u/u̇. This is why TTC needs no calibration and no prior.

    Two objects of very different sizes at very different ranges, closing so as
    to share a TTC, must yield the same TTC from image measurements alone.
    """
    ttcs = []
    for object_h, d0, v in ((1.6, 20.0, 10.0), (4.0, 50.0, 25.0)):
        u = [1.0 / height_px(d0 - v * DT * i, object_h) for i in range(2)]
        ttcs.append(ttc_from_state(u[1], (u[1] - u[0]) / DT))
    assert ttcs[0] == pytest.approx(ttcs[1], rel=1e-6)
    # TTC is evaluated at the CURRENT frame, so the truth is D(t=DT)/v, not
    # D(0)/v -- the object has already closed one frame's worth.
    assert ttcs[0] == pytest.approx((20.0 - 10.0 * DT) / 10.0, rel=1e-6)


def test_a_multiplicative_box_bias_cancels_exactly_in_ttc():
    """The measured −4.5% detector box-height bias costs TTC nothing.

    Scaling every height by k scales u and u̇ both by 1/k, and TTC is their
    ratio. This is why a bias that distorts Phase 3's range leaves TTC intact.
    """
    seq = [h for _, h in approach(25.0, 12.0, 2)]
    clean = ttc_from_state(1 / seq[1], (1 / seq[1] - 1 / seq[0]) / DT)
    for k in (0.9, 0.955, 1.1):
        biased = [k * h for h in seq]
        got = ttc_from_state(1 / biased[1], (1 / biased[1] - 1 / biased[0]) / DT)
        assert got == pytest.approx(clean, rel=1e-9)


# ── the filter on exact input ────────────────────────────────────────────────


def test_inverse_height_is_linear_in_time_under_constant_velocity():
    """The premise of the whole design: u ∝ D, and D is linear, so u is linear.

    If this failed, a constant-velocity model over u would be mis-specified and
    the filter would be fighting its own dynamics.
    """
    u = np.array([1.0 / h for _, h in approach(40.0, 15.0, 12)])
    second_difference = np.diff(u, 2)
    assert np.allclose(second_difference, 0.0, atol=1e-12)


@pytest.mark.parametrize(("d0", "v"), [(30.0, 10.0), (50.0, 25.0), (15.0, 5.0)])
def test_filter_recovers_the_true_ttc_on_noiseless_approach(d0, v):
    est = ScaleRateEstimator(cfg(), deadband="none")
    last = None
    for i, (d, h) in enumerate(approach(d0, v, 12)):
        last = est.update(1, h, DT, d)
        _ = i
    assert last.is_closing
    assert last.ttc_s == pytest.approx(last.range_m / v, rel=0.05)


def test_filter_recovers_the_true_closing_speed():
    """Closing speed is negative when approaching -- the sign convention matters."""
    est = ScaleRateEstimator(cfg(), deadband="none")
    last = None
    for d, h in approach(30.0, 12.0, 12):
        last = est.update(1, h, DT, d)
    assert last.closing_speed_mps == pytest.approx(-12.0, rel=0.06)


def test_ttc_counts_down_as_the_object_approaches():
    """TTC must fall roughly linearly, by about dt each frame."""
    est = ScaleRateEstimator(cfg(), deadband="none")
    ttcs = [
        s.ttc_s for d, h in approach(40.0, 10.0, 20) if (s := est.update(1, h, DT, d)).is_closing
    ]
    assert len(ttcs) > 10
    assert ttcs[-1] < ttcs[0]
    steps = np.diff(ttcs[3:])
    assert np.allclose(steps, -DT, atol=0.02)


def test_a_receding_object_never_reports_a_ttc():
    est = ScaleRateEstimator(cfg(), deadband="none")
    last = None
    for i in range(12):
        d = 15.0 + 8.0 * DT * i
        last = est.update(1, height_px(d), DT, d)
    assert not last.is_closing
    assert last.ttc_s == NOT_CLOSING


def test_no_ttc_is_emitted_before_the_filter_has_evidence():
    """A rate from one observation is the prior, not a measurement."""
    est = ScaleRateEstimator(cfg(min_updates=3), deadband="none")
    first = est.update(1, height_px(30.0), DT, 30.0)
    assert not first.is_closing
    assert first.n_updates == 1


# ── the case hard rule 4 exists for ──────────────────────────────────────────


def _flag_rate(
    deadband: str,
    sigma_px: float,
    speed_mps: float = 0.0,
    n_tracks: int = 200,
    seed: int = 0,
    config: MotionConfig | None = None,
) -> float:
    """Fraction of noisy tracks flagged as closing after 15 frames.

    `speed_mps=0` is a stationary object, so the rate is the PHANTOM rate; a
    positive speed is a genuine approach, so the rate is the DETECTION rate.
    """
    rng = np.random.default_rng(seed)
    est = ScaleRateEstimator(config or cfg(), deadband=deadband)
    flagged = 0
    for t in range(n_tracks):
        state = None
        for i in range(15):
            d = 30.0 - speed_mps * DT * i
            h = height_px(d) * (1.0 + sigma_px * rng.standard_normal())
            state = est.update(t, h, DT, d)
        flagged += bool(state.is_closing)
    return flagged / n_tracks


def _phantom_rate(deadband: str, sigma_px: float, n_tracks: int = 200, seed: int = 0) -> float:
    return _flag_rate(deadband, sigma_px, 0.0, n_tracks, seed)


def test_a_stationary_object_produces_no_closing_speed_without_noise():
    """Hard rule 4, the easy half."""
    est = ScaleRateEstimator(cfg(), deadband="none")
    last = None
    for _ in range(15):
        last = est.update(1, height_px(25.0), DT, 25.0)
    assert not last.is_closing
    assert last.closing_speed_mps == 0.0


def test_noise_alone_creates_phantom_closing_in_most_tracks_without_a_deadband():
    """The bug hard rule 4 names, quantified.

    With 6% height noise on a STATIONARY object, over half of tracks falsely
    report closing. A deadband is not a refinement; without one the closing-speed
    output is mostly fiction.
    """
    assert _phantom_rate("none", sigma_px=0.06, n_tracks=400) > 0.50


@pytest.mark.parametrize("deadband", ["fixed", "sigma"])
def test_both_deadbands_cut_phantom_closing_by_at_least_5x(deadband):
    assert _phantom_rate(deadband, sigma_px=0.06, n_tracks=400) < 0.11


def test_the_two_deadbands_are_equivalent_at_matched_phantom_rates():
    """A NULL result, kept because it corrects the obvious intuition.

    The sigma deadband reads the filter's own covariance and so 'adapts to track
    quality'. Swept against the fixed threshold at a matched phantom rate, it
    buys nothing: fixed@0.10 gives 1.5% phantom / 68% detection of a 3 m/s
    approach, sigma@1.5 gives 1.5% / 66%. Adaptivity only pays when track noise
    VARIES across tracks -- a property of the data, not of the mechanism
    (docs/decisions.md D-022).
    """
    fixed_cfg, sigma_cfg = cfg(deadband_inv_ttc=0.10), cfg(deadband_sigmas=1.5)
    kw = dict(sigma_px=0.06, n_tracks=400)
    fixed_phantom = _flag_rate("fixed", config=fixed_cfg, **kw)
    sigma_phantom = _flag_rate("sigma", config=sigma_cfg, **kw)
    fixed_detect = _flag_rate("fixed", speed_mps=3.0, seed=1, config=fixed_cfg, **kw)
    sigma_detect = _flag_rate("sigma", speed_mps=3.0, seed=1, config=sigma_cfg, **kw)

    assert fixed_phantom < 0.04 and sigma_phantom < 0.04
    assert abs(fixed_phantom - sigma_phantom) < 0.02
    assert abs(fixed_detect - sigma_detect) < 0.10


def test_the_deadband_costs_slow_approaches_and_not_fast_ones():
    """What the deadband actually trades.

    At the configured thresholds a 12 m/s approach is never suppressed; slow
    approaches are what get traded away. (A strict enough fixed horizon DOES
    start rejecting 6 m/s approaches -- 31% detection at TTC > 3 s -- which is
    why the operating point is a Phase 7 decision, not a constant.)
    """
    for deadband in ("fixed", "sigma"):
        est = ScaleRateEstimator(cfg(), deadband=deadband)
        last = None
        for i in range(15):
            d = 30.0 - 12.0 * DT * i
            last = est.update(1, height_px(d), DT, d)
        assert last.is_closing, f"{deadband} must not suppress a 12 m/s approach"


def test_a_deadband_does_not_suppress_a_genuine_approach():
    """It must reject noise without rejecting the thing the system exists for."""
    for deadband in ("fixed", "sigma"):
        est = ScaleRateEstimator(cfg(), deadband=deadband)
        last = None
        for d, h in approach(30.0, 12.0, 12):
            last = est.update(1, h, DT, d)
        assert last.is_closing, deadband
        assert last.ttc_s == pytest.approx(last.range_m / 12.0, rel=0.10)


# ── bookkeeping that protects the estimate ───────────────────────────────────


def test_reset_clears_a_track_so_an_id_switch_cannot_splice_two_objects():
    """Phase 4 measured 358 ID switches, concentrated at 10-20 m.

    Carrying a filter across a switch joins two objects' height histories and
    manufactures a scale rate from the seam -- a phantom TTC that looks entirely
    plausible.
    """
    est = ScaleRateEstimator(cfg(), deadband="none")
    for d, h in approach(40.0, 10.0, 8):
        est.update(1, h, DT, d)
    est.reset(1)
    fresh = est.update(1, height_px(12.0), DT, 12.0)
    assert fresh.n_updates == 1
    assert not fresh.is_closing


def test_a_degenerate_height_is_rejected_not_filtered():
    est = ScaleRateEstimator(cfg(), deadband="none")
    s = est.update(1, 0.0, DT, 20.0)
    assert not s.is_valid
    assert s.ttc_s == NOT_CLOSING


def test_unknown_deadband_raises():
    with pytest.raises(ValueError, match="unknown deadband"):
        ScaleRateEstimator(cfg(), deadband="vibes")


def test_measurement_noise_is_relative_so_it_scales_with_the_box():
    """sigma_u = sigma_rel * u -- the property that makes u the right state."""
    c = cfg()
    near = InverseHeightKF(1.0 / 160.0, c)
    far = InverseHeightKF(1.0 / 40.0, c)
    assert far.P[0, 0] / near.P[0, 0] == pytest.approx((160.0 / 40.0) ** 2)
