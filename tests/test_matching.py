"""Closed-form tests for IoU, association, ignore regions, and AP.

Every number here is computable by hand. Detection evaluation code is unusually
easy to get subtly wrong -- a double-counted match, a mishandled ignore region,
or a wrong AP denominator shifts the headline by points and is indistinguishable
from a model difference afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.matching import average_precision, ioa_matrix, iou_matrix, match_frame, nms

A = np.array([[0.0, 0.0, 10.0, 10.0]])  # area 100


def test_iou_of_identical_boxes_is_one():
    assert iou_matrix(A, A)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert iou_matrix(A, b)[0, 0] == pytest.approx(0.0)


def test_iou_half_overlap():
    """Intersection 50, union 150 -> 1/3."""
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    assert iou_matrix(A, b)[0, 0] == pytest.approx(1.0 / 3.0)


def test_iou_contained_box():
    """A 5x5 box inside a 10x10: intersection 25, union 100."""
    b = np.array([[0.0, 0.0, 5.0, 5.0]])
    assert iou_matrix(A, b)[0, 0] == pytest.approx(0.25)


def test_iou_touching_edges_is_zero_not_negative():
    b = np.array([[10.0, 0.0, 20.0, 10.0]])
    assert iou_matrix(A, b)[0, 0] == pytest.approx(0.0)


def test_iou_matrix_shape_and_empty():
    assert iou_matrix(np.zeros((3, 4)), np.zeros((5, 4))).shape == (3, 5)
    assert iou_matrix(np.zeros((0, 4)), A).shape == (0, 1)
    assert iou_matrix(A, np.zeros((0, 4))).shape == (1, 0)


def test_ioa_differs_from_iou_for_a_small_box_in_a_large_one():
    """Why ignore regions use IoA: a small detection fully inside a big region
    has IoA 1.0 but low IoU, and an IoU test would let it through."""
    small = np.array([[0.0, 0.0, 2.0, 2.0]])
    big = np.array([[0.0, 0.0, 20.0, 20.0]])
    assert ioa_matrix(small, big)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(small, big)[0, 0] == pytest.approx(4.0 / 400.0)


# ── NMS ──────────────────────────────────────────────────────────────────────


def test_nms_suppresses_overlapping_and_keeps_best_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    keep = nms(boxes, np.array([0.5, 0.9, 0.7]), iou_threshold=0.5)
    assert keep.tolist() == [1, 2]


def test_nms_is_class_agnostic_collapsing_a_cyclist():
    """The D-014 case: COCO emits person AND bicycle for one KITTI Cyclist.

    Per-class NMS keeps both, making one a true positive and the other a false
    positive for an object the detector actually found.
    """
    person = [100.0, 100.0, 130.0, 180.0]
    bicycle = [100.0, 130.0, 132.0, 185.0]
    boxes = np.array([person, bicycle])
    # They overlap enough to be one object at a permissive threshold.
    assert len(nms(boxes, np.array([0.9, 0.8]), iou_threshold=0.2)) == 1


def test_nms_keeps_everything_when_nothing_overlaps():
    boxes = np.array([[0, 0, 5, 5], [50, 50, 55, 55], [100, 100, 105, 105]], dtype=float)
    assert len(nms(boxes, np.array([0.3, 0.9, 0.6]))) == 3


# ── match_frame ──────────────────────────────────────────────────────────────


def test_perfect_match():
    m = match_frame(A, np.array([0.9]), A, iou_threshold=0.5)
    assert m.tp.tolist() == [True]
    assert m.gt_matched.tolist() == [True]
    assert m.det_to_gt.tolist() == [0]
    assert m.fp.tolist() == [False]


def test_duplicate_detection_is_a_false_positive():
    """A label may be claimed once. The second box on the same car is an error.

    This is the greedy-by-score protocol: it scores duplicate detections as the
    failure they are, where a total-IoU assignment would not.
    """
    dets = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    m = match_frame(dets, np.array([0.9, 0.8]), A)
    assert m.tp.tolist() == [True, False]
    assert m.fp.tolist() == [False, True]


def test_highest_scoring_detection_claims_the_label():
    dets = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    m = match_frame(dets, np.array([0.3, 0.95]), A)
    assert m.tp.tolist() == [False, True]


def test_below_iou_threshold_is_a_false_positive_and_the_label_is_missed():
    det = np.array([[7.0, 0.0, 17.0, 10.0]])  # IoU 3/17 with A
    m = match_frame(det, np.array([0.9]), A, iou_threshold=0.5)
    assert m.tp.tolist() == [False]
    assert m.gt_matched.tolist() == [False]


def test_unmatched_detection_on_an_ignore_region_scores_as_neither():
    """A detected Tram is a real object this evaluation does not grade.

    KITTI raw has no DontCare regions, so without this the precision figure is
    penalised for correct detections of unscored classes.
    """
    det = np.array([[0.0, 0.0, 10.0, 10.0]])
    m = match_frame(det, np.array([0.9]), np.zeros((0, 4)), ignore_boxes=A)
    assert m.tp.tolist() == [False]
    assert m.ignored.tolist() == [True]
    assert m.fp.tolist() == [False]


def test_a_true_positive_is_not_stolen_by_an_overlapping_ignore_region():
    """Matching a real label wins over overlapping something unscored."""
    m = match_frame(A, np.array([0.9]), A, ignore_boxes=A)
    assert m.tp.tolist() == [True]
    assert m.ignored.tolist() == [False]


def test_no_labels_makes_every_unignored_detection_a_false_positive():
    dets = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    m = match_frame(dets, np.array([0.9, 0.8]), np.zeros((0, 4)))
    assert m.fp.tolist() == [True, True]


def test_no_detections_leaves_every_label_missed():
    m = match_frame(np.zeros((0, 4)), np.array([]), A)
    assert m.gt_matched.tolist() == [False]
    assert len(m.tp) == 0


# ── average_precision ────────────────────────────────────────────────────────


def test_ap_is_one_when_every_label_is_found_with_no_false_positives():
    ap, _, _ = average_precision(np.array([0.9, 0.8]), np.array([True, True]), n_gt=2)
    assert ap == pytest.approx(1.0)


def test_ap_is_zero_when_nothing_is_found():
    ap, _, _ = average_precision(np.array([0.9, 0.8]), np.array([False, False]), n_gt=2)
    assert ap == pytest.approx(0.0)


def test_ap_halves_when_half_the_labels_are_missed():
    """One perfect detection, one label never found: recall caps at 0.5."""
    ap, _, _ = average_precision(np.array([0.9]), np.array([True]), n_gt=2)
    assert ap == pytest.approx(0.5)


def test_ap_penalises_a_high_scoring_false_positive():
    """The FP ranks above the TP, so precision at the only recall step is 0.5."""
    ap, _, _ = average_precision(np.array([0.95, 0.5]), np.array([False, True]), n_gt=1)
    assert ap == pytest.approx(0.5)


def test_ap_ignores_false_positives_ranked_below_every_true_positive():
    """Precision is already 1.0 at full recall; trailing junk cannot reduce it."""
    ap, _, _ = average_precision(np.array([0.9, 0.1]), np.array([True, False]), n_gt=1)
    assert ap == pytest.approx(1.0)


def test_ap_uses_score_order_not_array_order():
    a, _, _ = average_precision(np.array([0.1, 0.9]), np.array([False, True]), n_gt=1)
    b, _, _ = average_precision(np.array([0.9, 0.1]), np.array([True, False]), n_gt=1)
    assert a == pytest.approx(b)


def test_ap_with_no_labels_is_nan_not_zero():
    """No labels means the question is undefined, not that the model failed."""
    ap, _, _ = average_precision(np.array([0.9]), np.array([False]), n_gt=0)
    assert np.isnan(ap)


def test_ap_with_no_detections_is_zero():
    ap, _, _ = average_precision(np.array([]), np.array([]), n_gt=5)
    assert ap == pytest.approx(0.0)


def test_precision_recall_curves_are_monotone_in_recall():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    tp = np.array([True, False, True, True])
    _, precision, recall = average_precision(scores, tp, n_gt=3)
    assert np.all(np.diff(recall) >= 0)
    assert recall[-1] == pytest.approx(1.0)
    assert precision[0] == pytest.approx(1.0)
