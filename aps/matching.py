"""Detection/label association and average precision.

Pure functions over arrays -- no dataset, no model -- so every case here has a
closed-form answer and can be asserted rather than eyeballed. AP bookkeeping is
where detection evaluations quietly go wrong: a double-counted match or a
mishandled ignore region moves the headline number by several points and looks
like a model difference.

Reused by Phase 4 tracking, which needs the same IoU and assignment machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) boxes in [x1,y1,x2,y2] -> (N,M) intersection-over-union."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of `a` -- (N,M).

    Used for ignore regions rather than IoU: a small detection sitting entirely
    inside a large ignore region has low IoU with it but should still be
    ignored. IoU would let it through and score it as a false positive.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area_a[:, None] > 0, inter / area_a[:, None], 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.6) -> np.ndarray:
    """Class-agnostic non-maximum suppression. Returns kept indices, best first.

    Class-agnostic is the point, not a simplification. A COCO detector emits a
    `person` box AND a `bicycle` box for one cyclist; KITTI labels that as a
    single `Cyclist`. Per-class NMS keeps both, so one becomes a true positive
    and the other a false positive for an object the detector actually found.
    Suppressing across classes collapses them into the one object that is
    physically there. See docs/decisions.md D-014.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of associating one frame's detections with its labels."""

    tp: np.ndarray  # (N_det,) bool -- detection matched a label
    ignored: np.ndarray  # (N_det,) bool -- fell on an ignore region; scores as neither
    det_to_gt: np.ndarray  # (N_det,) int -- matched label index, or -1
    gt_matched: np.ndarray  # (N_gt,) bool -- label was found

    @property
    def fp(self) -> np.ndarray:
        """False positives: unmatched and not ignored."""
        return ~self.tp & ~self.ignored


def match_frame(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    ignore_boxes: np.ndarray | None = None,
    ignore_ioa_threshold: float = 0.5,
) -> MatchResult:
    """Greedily associate detections to labels, highest score first.

    Greedy-by-score is the standard AP protocol: each label may be claimed once,
    and a second detection on the same object is a false positive (duplicate
    detections are a real failure, not a rounding detail). Hungarian assignment
    would maximise total IoU instead, which scores a different thing.

    `ignore_boxes` are labelled objects outside the evaluated classes -- KITTI
    Tram, Misc, Person (sitting). A detection landing on one is removed from the
    tally entirely rather than counted as a false positive: the detector found a
    real object that this evaluation simply does not score. KITTI raw has no
    `DontCare` regions, so without this the precision figure would be penalised
    for correct detections.
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    n_det, n_gt = len(det_boxes), len(gt_boxes)

    tp = np.zeros(n_det, dtype=bool)
    ignored = np.zeros(n_det, dtype=bool)
    det_to_gt = np.full(n_det, -1, dtype=int)
    gt_matched = np.zeros(n_gt, dtype=bool)

    ious = iou_matrix(det_boxes, gt_boxes)
    for d in np.argsort(-np.asarray(det_scores, dtype=np.float64)):
        if n_gt == 0:
            break
        candidates = np.where(~gt_matched, ious[d], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_threshold:
            tp[d] = True
            det_to_gt[d] = best
            gt_matched[best] = True

    # Ignore regions are applied only to detections that matched nothing --
    # a detection that already found a real labelled object stays a true
    # positive even if it also overlaps something unscored.
    if ignore_boxes is not None and len(ignore_boxes):
        ioa = ioa_matrix(det_boxes, np.asarray(ignore_boxes, dtype=np.float64))
        ignored = ~tp & (ioa.max(axis=1) >= ignore_ioa_threshold)

    return MatchResult(tp, ignored, det_to_gt, gt_matched)


def average_precision(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """VOC-style all-point-interpolated AP over a pooled set of detections.

    Returns (AP, precision curve, recall curve). All-point interpolation rather
    than the 11-point sampling: 11-point is a legacy approximation that both
    over- and under-states depending on where the curve's steps fall.

    `n_gt` must be the number of labels the detector was *required* to find,
    with ignored detections already removed from `scores`/`is_tp`. Getting that
    denominator wrong is the most common way a recall number becomes fiction.
    """
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if len(scores) == 0:
        return 0.0, np.array([]), np.array([])

    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(is_tp, dtype=bool)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Make precision monotonically non-increasing, then integrate over recall.
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    return ap, precision, recall
