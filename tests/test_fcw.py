"""The FCW decision layer on sequences with known answers.

The decision layer is evaluated as a detector on real data (eval/eval_fcw.py);
these tests pin its logic so that a real-data number cannot come from a
bookkeeping bug -- an off-by-one in persistence, an event split in two, or an
interval computed wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.fcw import (
    FcwConfig,
    FcwDecider,
    corridor_range_m,
    lateral_offset_m,
    poisson_interval,
    threat_events,
)


def cfg(**over) -> FcwConfig:
    base = dict(
        ttc_warning_s=2.0,
        ttc_aeb_s=1.0,
        corridor_half_width_m=1.5,
        persistence_k=2,
        persistence_n=3,
        fallback_range_m=6.0,
    )
    base.update(over)
    return FcwConfig(**base)


def run(decider: FcwDecider, ttcs, lateral=0.0, closing=True, track=1):
    return [decider.update(track, closing, t, lateral) for t in ttcs]


# ── configuration guards ─────────────────────────────────────────────────────


def test_committed_config_loads():
    c = FcwConfig.from_config()
    assert c.ttc_aeb_s <= c.ttc_warning_s


def test_config_rejects_impossible_persistence():
    with pytest.raises(ValueError, match="persistence"):
        FcwConfig.from_config(persistence_k=4, persistence_n=3)


def test_config_rejects_an_aeb_threshold_above_the_warning():
    with pytest.raises(ValueError, match="AEB"):
        FcwConfig.from_config(ttc_aeb_s=2.5, ttc_warning_s=2.0)


# ── geometry helpers ─────────────────────────────────────────────────────────


def test_lateral_offset_is_the_pinhole_relation():
    assert lateral_offset_m(609.56 + 72.15, 721.54, 609.56, 10.0) == pytest.approx(1.0, abs=1e-3)
    assert lateral_offset_m(609.56, 721.54, 609.56, 30.0) == pytest.approx(0.0)


def test_corridor_range_falls_back_rather_than_exempting_a_close_object():
    """Both range estimators abstain on a box clipped at the image bottom -- an
    object very close ahead. That object must still be tested for the corridor."""
    c = cfg()
    assert corridor_range_m(12.0, 14.0, c) == 12.0
    assert corridor_range_m(float("nan"), 14.0, c) == 14.0
    assert corridor_range_m(float("nan"), float("nan"), c) == c.fallback_range_m


# ── the decision rule ────────────────────────────────────────────────────────


def test_a_steady_approach_warns_once_after_persistence_is_met():
    out = run(FcwDecider(cfg()), [3.0, 2.5, 1.9, 1.8, 1.7, 1.6])
    onsets = [i for i, o in enumerate(out) if o.warning_onset]
    assert onsets == [3]  # candidates at frames 2 and 3 -> 2 of the last 3
    assert all(o.warning for o in out[3:])


def test_a_single_frame_dip_does_not_warn_with_persistence():
    """The failure persistence exists to stop: one jittery TTC frame."""
    out = run(FcwDecider(cfg()), [3.0, 3.0, 1.5, 3.0, 3.0, 3.0])
    assert not any(o.warning for o in out)


def test_the_same_dip_warns_without_persistence():
    out = run(FcwDecider(cfg(persistence_k=1, persistence_n=1)), [3.0, 3.0, 1.5, 3.0])
    assert [o.warning_onset for o in out] == [False, False, True, False]


def test_persistence_costs_exactly_k_minus_one_frames_of_latency():
    """Each extra required frame is 0.104 s taken from the driver."""
    seq = [1.9] * 6
    first = {}
    for k in (1, 2, 3):
        out = run(FcwDecider(cfg(persistence_k=k, persistence_n=3)), seq)
        first[k] = next(i for i, o in enumerate(out) if o.warning_onset)
    assert first == {1: 0, 2: 1, 3: 2}


def test_an_object_outside_the_corridor_never_warns():
    out = run(FcwDecider(cfg()), [1.5] * 6, lateral=2.2)
    assert not any(o.warning for o in out)


def test_a_not_closing_object_never_warns_whatever_its_ttc():
    """The Phase 5 deadband gates the decision; a TTC value alone is not enough."""
    out = run(FcwDecider(cfg()), [1.5] * 6, closing=False)
    assert not any(o.warning for o in out)


def test_aeb_request_requires_the_warning_and_the_stricter_threshold():
    out = run(FcwDecider(cfg()), [1.9, 1.8, 1.5, 0.9, 0.8])
    assert [o.aeb_request for o in out] == [False, False, False, True, True]


def test_tracks_are_independent_and_reset_forgets_history():
    d = FcwDecider(cfg())
    d.update(1, True, 1.9, 0.0)
    d.update(2, True, 1.9, 0.0)
    assert d.update(1, True, 1.9, 0.0).warning_onset
    d.reset(2)
    assert not d.update(2, True, 1.9, 0.0).warning


# ── evaluation primitives ────────────────────────────────────────────────────


def test_threat_events_merge_across_a_short_dropout():
    frames = np.array([10, 11, 12, 14, 15, 30, 31])
    threat = np.ones(len(frames), dtype=bool)
    assert threat_events(frames, threat, max_gap=2) == [(10, 15), (30, 31)]


def test_threat_events_ignore_non_threat_frames():
    frames = np.arange(6)
    threat = np.array([False, True, True, False, False, False])
    assert threat_events(frames, threat) == [(1, 2)]
    assert threat_events(frames, np.zeros(6, dtype=bool)) == []


def test_poisson_interval_matches_known_values():
    """Garwood exact interval: 0 events -> [0, 3.689]; 3 events -> [0.619, 8.767]."""
    assert poisson_interval(0) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert poisson_interval(3) == pytest.approx((0.619, 8.767), abs=1e-3)
