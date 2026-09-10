"""Tests for the trackers and the MOT metrics.

Every metric case here is hand-computable. MOT metrics are notoriously easy to
get subtly wrong -- an ID switch counted on the wrong side, a preserved
correspondence dropped, an IDF1 denominator off -- and a wrong metric produces
numbers that look plausible and cannot be defended.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.motmetrics import MOTAccumulator
from aps.tracking import (
    BoxKalmanFilter,
    IoUTracker,
    SortTracker,
    _box_to_z,
    _z_to_box,
    make_tracker,
)


def box(x1, y1, x2, y2) -> np.ndarray:
    return np.array([float(x1), float(y1), float(x2), float(y2)])


# ── box <-> state ────────────────────────────────────────────────────────────


def test_box_state_roundtrip():
    b = box(100, 50, 140, 130)  # 40 x 80
    assert _z_to_box(_box_to_z(b)) == pytest.approx(b)


def test_box_to_z_is_centre_area_aspect():
    z = _box_to_z(box(100, 50, 140, 130))
    assert z == pytest.approx([120.0, 90.0, 40 * 80, 40 / 80])


# ── Kalman filter ────────────────────────────────────────────────────────────


def test_kalman_reproduces_a_static_box():
    kf = BoxKalmanFilter(box(100, 100, 140, 180))
    for _ in range(5):
        kf.predict()
        kf.update(box(100, 100, 140, 180))
    assert kf.box == pytest.approx(box(100, 100, 140, 180), abs=1.0)


def test_kalman_extrapolates_constant_motion_into_a_gap():
    """The whole point of SORT over the IoU baseline: predict where it is going.

    A box moving 10 px/frame is tracked for several frames, then the detection
    is withheld. The prediction must continue moving, not freeze.
    """
    kf = BoxKalmanFilter(box(100, 100, 140, 180))
    for i in range(1, 12):
        kf.predict()
        kf.update(box(100 + 10 * i, 100, 140 + 10 * i, 180))
    last_centre = kf.box[0]
    predicted = kf.predict()  # no update -- a missed detection
    assert predicted[0] > last_centre + 5.0


def test_kalman_never_produces_a_negative_area():
    """A shrinking track extrapolated far enough would invert the box."""
    kf = BoxKalmanFilter(box(0, 0, 200, 200))
    for i in range(6):
        kf.predict()
        kf.update(box(0, 0, 200 - 30 * i, 200 - 30 * i))
    for _ in range(30):
        b = kf.predict()
        assert b[2] >= b[0]
        assert b[3] >= b[1]


# ── trackers, shared behaviour ───────────────────────────────────────────────


@pytest.mark.parametrize("name", ["iou", "sort"])
def test_tracker_assigns_and_keeps_one_id_for_a_static_object(name):
    tr = make_tracker(name, max_age=1)
    ids = set()
    for _ in range(6):
        out = tr.update(np.array([box(100, 100, 140, 180)]), np.array([0.9]))
        assert len(out) == 1
        ids.add(out[0].track_id)
    assert len(ids) == 1


@pytest.mark.parametrize("name", ["iou", "sort"])
def test_tracker_keeps_ids_distinct_for_two_separated_objects(name):
    tr = make_tracker(name)
    seen = []
    for _ in range(5):
        out = tr.update(np.array([box(0, 0, 40, 80), box(500, 0, 540, 80)]), np.array([0.9, 0.8]))
        seen.append({t.track_id for t in out})
    assert all(len(s) == 2 for s in seen)
    assert seen[0] == seen[-1]


@pytest.mark.parametrize("name", ["iou", "sort"])
def test_tracker_starts_a_new_id_for_a_new_object(name):
    tr = make_tracker(name)
    tr.update(np.array([box(0, 0, 40, 80)]), np.array([0.9]))
    out = tr.update(np.array([box(0, 0, 40, 80), box(300, 0, 340, 80)]), np.array([0.9, 0.7]))
    assert len({t.track_id for t in out}) == 2


@pytest.mark.parametrize("name", ["iou", "sort"])
def test_tracker_drops_a_track_after_max_age(name):
    tr = make_tracker(name, max_age=2)
    first = tr.update(np.array([box(0, 0, 40, 80)]), np.array([0.9]))[0].track_id
    for _ in range(4):
        tr.update(np.zeros((0, 4)), np.array([]))
    out = tr.update(np.array([box(0, 0, 40, 80)]), np.array([0.9]))
    assert out[0].track_id != first


@pytest.mark.parametrize("name", ["iou", "sort"])
def test_tracker_handles_empty_frames(name):
    tr = make_tracker(name)
    assert tr.update(np.zeros((0, 4)), np.array([])) == []


def test_unknown_tracker_raises():
    with pytest.raises(ValueError, match="unknown tracker"):
        make_tracker("magic")


# ── where the two trackers differ ────────────────────────────────────────────


# 20 px/frame on a 60 px-wide box: consecutive frames overlap at IoU 0.5 (so
# both trackers associate normally), but across a ONE-FRAME GAP the object has
# moved 40 px and overlaps its last observed position at only IoU 0.2 -- below
# the 0.3 threshold. That is the regime where a motion model is decisive.
_STEP = 20
_SEQ = [box(_STEP * i, 0, 60 + _STEP * i, 80) for i in range(9)]


def _run_with_gap(tracker):
    """Four clean frames, one dropped detection, then resume. Returns (first_id, id_after)."""
    first = tracker.update(np.array([_SEQ[0]]), np.array([0.9]))[0].track_id
    for b in _SEQ[1:4]:
        tracker.update(np.array([b]), np.array([0.9]))
    tracker.update(np.zeros((0, 4)), np.array([]))  # dropped frame
    out = tracker.update(np.array([_SEQ[5]]), np.array([0.9]))
    return first, (out[0].track_id if out else None)


def test_sort_keeps_its_id_across_a_dropped_detection():
    """SORT's advantage, isolated: it looks where the object is GOING."""
    first, after = _run_with_gap(SortTracker(max_age=3))
    assert after == first


def test_iou_baseline_loses_its_id_across_the_same_gap():
    """Same sequence, same thresholds -- only the motion model differs.

    This pair is the entire justification for SORT's extra complexity, and it is
    why an ID-switch count is the metric that separates them.
    """
    first, after = _run_with_gap(IoUTracker(max_age=3))
    assert after is not None
    assert after != first


def test_iou_baseline_has_no_motion_model():
    """Its box after a gap is the LAST OBSERVED one, not an extrapolation."""
    tr = IoUTracker(max_age=3)
    tr.update(np.array([box(0, 0, 60, 80)]), np.array([0.9]))
    tr.update(np.array([box(20, 0, 80, 80)]), np.array([0.9]))
    tr.update(np.zeros((0, 4)), np.array([]))
    assert tr._tracks[0].box == pytest.approx(box(20, 0, 80, 80))


def test_neither_tracker_can_bootstrap_on_an_object_that_never_overlaps():
    """A real limitation, found by a test whose premise was wrong.

    A newborn track has ZERO velocity, so SORT's first prediction is its birth
    box. An object moving further than its own width per frame therefore never
    associates even once, and the Kalman filter never gets the two observations
    it needs to estimate motion. SORT cannot rescue what it never caught.

    Consequence for the evaluation: fast-crossing objects fragment under BOTH
    trackers, and SORT's advantage is confined to objects it has already locked
    onto (docs/decisions.md D-021).
    """
    fast = [box(45 * i, 0, 60 + 45 * i, 80) for i in range(6)]
    for tracker in (SortTracker(max_age=3), IoUTracker(max_age=3)):
        ids = set()
        for b in fast:
            out = tracker.update(np.array([b]), np.array([0.9]))
            ids.update(t.track_id for t in out)
        assert len(ids) == len(fast), f"{type(tracker).__name__} should fragment every frame"


# ── MOT metrics ──────────────────────────────────────────────────────────────


def test_perfect_tracking_scores_one():
    acc = MOTAccumulator()
    for _ in range(5):
        acc.update(["a"], np.array([box(0, 0, 40, 80)]), [1], np.array([box(0, 0, 40, 80)]))
    r = acc.result()
    assert r.mota == pytest.approx(1.0)
    assert r.idf1 == pytest.approx(1.0)
    assert r.motp == pytest.approx(1.0)
    assert r.num_switches == 0


def test_every_object_missed_scores_zero_mota():
    acc = MOTAccumulator()
    for _ in range(4):
        acc.update(["a"], np.array([box(0, 0, 40, 80)]), [], np.zeros((0, 4)))
    r = acc.result()
    assert r.mota == pytest.approx(0.0)
    assert r.num_misses == 4
    assert r.idf1 == pytest.approx(0.0)


def test_mota_goes_negative_when_false_positives_exceed_ground_truth():
    """Not an overflow -- MOTA is unbounded below, and that must survive."""
    acc = MOTAccumulator()
    for _ in range(3):
        acc.update(
            ["a"],
            np.array([box(0, 0, 40, 80)]),
            [1, 2, 3],
            np.array([box(0, 0, 40, 80), box(200, 0, 240, 80), box(400, 0, 440, 80)]),
        )
    r = acc.result()
    assert r.num_fp == 6
    assert r.mota == pytest.approx(1.0 - 6 / 3)


def test_one_id_switch_is_counted_once():
    """Same object, hypothesis id changes halfway. Exactly one switch."""
    acc = MOTAccumulator()
    b = np.array([box(0, 0, 40, 80)])
    for _ in range(3):
        acc.update(["a"], b, [1], b)
    for _ in range(3):
        acc.update(["a"], b, [2], b)
    r = acc.result()
    assert r.num_switches == 1
    assert r.num_matches == 6
    assert r.mota == pytest.approx(1.0 - 1 / 6)


def test_correspondence_is_preserved_when_a_rival_overlaps_more():
    """The CLEAR MOT rule that stops the metric inventing switches.

    Hypothesis 1 holds the object. A second hypothesis appears overlapping
    slightly better. The existing correspondence must survive.
    """
    acc = MOTAccumulator()
    gt = np.array([box(0, 0, 40, 80)])
    acc.update(["a"], gt, [1], np.array([box(2, 0, 42, 80)]))
    acc.update(["a"], gt, [1, 2], np.array([box(2, 0, 42, 80), box(0, 0, 40, 80)]))
    assert acc.result().num_switches == 0


def test_fragmentation_is_counted_when_a_track_resumes_after_a_gap():
    acc = MOTAccumulator()
    b = np.array([box(0, 0, 40, 80)])
    acc.update(["a"], b, [1], b)
    acc.update(["a"], b, [], np.zeros((0, 4)))  # lost
    acc.update(["a"], b, [1], b)  # resumed, same id
    r = acc.result()
    assert r.fragmentations == 1
    assert r.num_switches == 0  # same identity, so not a switch


def test_idf1_punishes_fragmentation_where_mota_barely_notices():
    """The reason both metrics are reported.

    Ten frames of one object, split across ten different hypothesis ids. Every
    frame is a match, so MOTA loses only the nine switches -- but only one
    hypothesis identity can be credited, so IDF1 collapses.
    """
    acc = MOTAccumulator()
    b = np.array([box(0, 0, 40, 80)])
    for i in range(10):
        acc.update(["a"], b, [i], b)
    r = acc.result()
    assert r.num_matches == 10
    assert r.num_switches == 9
    assert r.mota == pytest.approx(1.0 - 9 / 10)
    # IDTP = 1 (one id matched for one frame); IDFN = 9, IDFP = 9.
    assert r.idf1 == pytest.approx(2 * 1 / (2 * 1 + 9 + 9))
    assert r.idf1 < 0.2


def test_idf1_is_perfect_for_a_consistent_identity():
    acc = MOTAccumulator()
    b = np.array([box(0, 0, 40, 80)])
    for _ in range(10):
        acc.update(["a"], b, [7], b)
    assert acc.result().idf1 == pytest.approx(1.0)


def test_identities_do_not_carry_across_a_sequence_boundary():
    """Without new_sequence(), the last track of one drive would be scored as an
    ID switch against the first object of the next."""
    acc = MOTAccumulator()
    b = np.array([box(0, 0, 40, 80)])
    acc.update(["a"], b, [1], b)
    acc.new_sequence()
    acc.update(["a"], b, [2], b)
    assert acc.result().num_switches == 0


def test_below_iou_threshold_is_a_miss_and_a_false_positive():
    acc = MOTAccumulator(iou_threshold=0.5)
    acc.update(["a"], np.array([box(0, 0, 40, 80)]), [1], np.array([box(35, 0, 75, 80)]))
    r = acc.result()
    assert r.num_matches == 0
    assert r.num_misses == 1
    assert r.num_fp == 1


def test_motp_reports_localisation_quality_of_matches_only():
    acc = MOTAccumulator(iou_threshold=0.3)
    gt = np.array([box(0, 0, 100, 100)])
    acc.update(["a"], gt, [1], np.array([box(0, 0, 100, 50)]))  # IoU = 0.5
    assert acc.result().motp == pytest.approx(0.5)


def test_empty_accumulator_is_nan_not_zero():
    """No ground truth means the question is undefined, not that it failed."""
    r = MOTAccumulator().result()
    assert np.isnan(r.mota)
