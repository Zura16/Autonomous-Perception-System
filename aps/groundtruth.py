"""Per-detection range ground truth, derived from projected LiDAR.

This module is the ruler. Everything the project later claims about monocular
range is measured against what happens here, so its own error is characterised
first -- a ruler with unquantified error cannot bound anything.

The problem it solves. Given a 2D box in the image and a projected point cloud,
recover the range to the object that box contains. Naively this is "the depth of
the points inside the box", but a 2D box is a poor container:

  * it includes background seen past the object's silhouette, especially at the
    corners of a box around anything that is not rectangular;
  * it includes returns that pass THROUGH glass, so a car's box contains points
    from its own interior and from the road behind it;
  * a nearer, occluding object overlapping the box contributes the *closest*
    points in it, which is exactly what a naive minimum would latch onto;
  * the number of returns falls off as roughly 1/range^2, so the estimator gets
    weaker precisely where the monocular estimator being measured gets weaker.

So several estimators are implemented and *compared against the labelled 3D
boxes* rather than one being asserted. `tools/build_range_gt.py` runs that
comparison and reports which wins, with its residual spread, by range bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aps.kitti.calibration import Calibration
from aps.kitti.tracklets import OCC_FULLY, OCC_PARTLY, OCC_UNSET, OCC_VISIBLE, TRUNC_IN_IMAGE

# A box with fewer returns than this is not measured -- it is reported as a gap
# in ground-truth coverage. Filling it with a low-confidence guess would put
# noise into the ruler and silently widen every downstream error bar.
MIN_POINTS_DEFAULT = 8

# Each box edge is pulled inward by this fraction before depths are read.
# Shrinking buys purity and costs sample count.
#
# NEGATIVE RESULT, kept because it was expensive to learn (docs/decisions.md
# D-011). A fallback ladder (0.25 -> 0.125 -> 0.0) was added to lift validity
# from 86% to 100% by retrying starved boxes at a smaller shrink. Measured on
# the dev split, the boxes it recovered were *twelve times worse* than the ones
# it kept:
#
#     shrink 0.25  N=212  MAE 0.130 m  p95 0.37 m
#     shrink 0.125 N= 25  MAE 1.563 m  p95 11.15 m
#     shrink 0.0   N=  9  MAE 0.481 m  p95 1.81 m
#
# Four fallback boxes were enough to move the 20-30 m bin from 0.15 m to 0.69 m.
# A box too sparse to measure at 0.25 is a box whose object is small, distant,
# or thin -- exactly the case where the border returns are background. The
# abstention was the correct behaviour and the "improvement" was manufacturing
# confident wrong answers to fill a coverage column. Reverted.
SHRINK_FRACTION = 0.25


# ── Ground-truth validity ────────────────────────────────────────────────────
#
# Measured on the dev split against the labelled 3D boxes, this gate is the
# single largest effect in the whole ruler -- far larger than the choice of
# estimator (docs/decisions.md D-010):
#
#     occlusion=0 (visible)      MAE 0.10 m     <- the ruler
#     occlusion=-1 (unset)       MAE 0.28 m
#     occlusion=1 (partly)       MAE 1.33 m
#     occlusion=2 (fully)        MAE 11.86 m    <- measuring the occluder
#
# The occ=2 number is not a defect to fix. You cannot range-find an object you
# cannot see; the box contains the occluder's surface, so a LiDAR estimate for
# it is a confident measurement of the wrong object. Those boxes are declared
# ungroundtruthable and excluded, and the exclusion is reported as coverage.

GT_STRICT, GT_RELAXED, GT_INVALID = "strict", "relaxed", "invalid"


def gt_tier(occlusion: int, truncation: int) -> str:
    """Which ground-truth tier a labelled box qualifies for.

    strict   fully visible, untruncated. Ruler error ~0.10 m. Use for any
             headline range number.
    relaxed  occlusion unlabelled but untruncated. Ruler error ~0.28 m --
             still ~20x tighter than any monocular error being measured, and
             it roughly doubles usable coverage.
    invalid  partly/fully occluded, or truncated at the image border. For
             truncated boxes the label's near face is amodal and may sit
             outside the image entirely, so LiDAR sees only the far part of
             the object and reads ~2 m long. No ground truth exists here.
    """
    if truncation != TRUNC_IN_IMAGE:
        return GT_INVALID
    if occlusion == OCC_VISIBLE:
        return GT_STRICT
    if occlusion == OCC_UNSET:
        return GT_RELAXED
    if occlusion in (OCC_PARTLY, OCC_FULLY):
        return GT_INVALID
    return GT_INVALID


@dataclass(frozen=True)
class RangeGT:
    """Ground-truth range for one box, with the evidence behind it."""

    range_m: float  # near-surface longitudinal range, metres
    n_points: int  # returns supporting the estimate
    n_in_box: int  # returns inside the box before robustifying
    spread_m: float  # p75-p25 of the supporting cluster: internal disagreement
    estimator: str
    shrink: float = 0.0  # shrink fraction used; SHRINK_FRACTION for shrink_p20

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.range_m) and self.n_points > 0


def points_in_box(
    uv: np.ndarray,
    depth: np.ndarray,
    box: np.ndarray,
    shrink: float = 0.0,
) -> np.ndarray:
    """Boolean mask of projected points falling inside `box` [x1,y1,x2,y2].

    `shrink` pulls each edge inward by that fraction of the box's size (0.25
    keeps the central half). Shrinking trades sample count for purity: the
    silhouette-edge background lives at the box border, and the object surface
    lives in the middle. It is also how a box around an object with a hole in it
    (a cyclist, a truck's undercarriage) stops being mostly road.
    """
    x1, y1, x2, y2 = box
    if shrink > 0:
        dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
        x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    return (depth > 0) & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def nearest_cluster(depths: np.ndarray, gap_m: float = 2.0, min_points: int = 4) -> np.ndarray:
    """Return the nearest depth cluster with enough support.

    Splits the sorted depths wherever consecutive values differ by more than
    `gap_m` and returns the first run holding at least `min_points`. This is a
    1D stand-in for segmentation: an object's surface returns are contiguous in
    depth, while the road behind it and any occluder in front of it are
    separated by a gap much larger than the object's own depth extent.

    `min_points` is what makes it robust rather than a fancy minimum -- a lone
    stray return in front of the object forms a run of one and is skipped.
    """
    if len(depths) == 0:
        return depths
    order = np.sort(depths)
    splits = np.flatnonzero(np.diff(order) > gap_m) + 1
    for run in np.split(order, splits):
        if len(run) >= min_points:
            return run
    return order  # nothing had support; caller sees the low n_points


def estimate_range(
    uv: np.ndarray,
    depth: np.ndarray,
    box: np.ndarray,
    estimator: str = "cluster_p20",
    min_points: int = MIN_POINTS_DEFAULT,
) -> RangeGT:
    """Ground-truth range for one 2D box from projected LiDAR.

    Estimators, all returning a *near-surface* range per docs/decisions.md D-003:

      min          minimum depth in the box. The obvious choice and the worst
                   one: one return from an occluder or a mis-projected edge
                   point sets it. Included as the cautionary baseline.
      p10 / p20    low percentile in the box. Robust to single outliers, still
                   contaminated by any occluder covering >20% of the box.
      median       the object's middle, not its near face. Biased far by roughly
                   half the object's depth extent (~2 m for a car). Included
                   because it is what most published work reports.
      shrink_p20   p20 over the central half of the box, ABSTAINING rather than
                   widening when that starves the sample. THE DEFAULT -- measured
                   MAE 0.05/0.08/0.15/0.18/0.23 m over the five range bins on
                   usable-tier boxes, at 86% coverage of them.
      cluster_p20  nearest supported depth cluster, then its 20th percentile.
                   Loses to shrink_p20 beyond 30 m (MAE 1.51 vs 0.15 m): a thin
                   foreground object forms its own supported cluster and the
                   estimator confidently reports the sign post.
    """
    value = {
        "min": lambda a: float(a.min()),
        "p10": lambda a: float(np.percentile(a, 10)),
        "p20": lambda a: float(np.percentile(a, 20)),
        "median": lambda a: float(np.median(a)),
        "shrink_p20": lambda a: float(np.percentile(a, 20)),
        "cluster_p20": lambda a: float(np.percentile(a, 20)),
    }
    if estimator not in value:
        raise ValueError(f"unknown estimator {estimator!r}; have {sorted(value)}")

    shrink = SHRINK_FRACTION if estimator == "shrink_p20" else 0.0
    inside = points_in_box(uv, depth, box, shrink=shrink)
    n_in_box = int(inside.sum())
    d = depth[inside]
    if estimator == "cluster_p20":
        d = nearest_cluster(d)

    if len(d) < min_points:
        return RangeGT(np.nan, len(d), n_in_box, np.nan, estimator, shrink)

    spread = float(np.percentile(d, 75) - np.percentile(d, 25))
    return RangeGT(value[estimator](d), len(d), n_in_box, spread, estimator, shrink)


def project_frame(points_velo: np.ndarray, calib: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Project a cloud once and keep only what lands on the sensor.

    Hoisted out of the per-box loop deliberately: projecting ~50k points is the
    expensive part, and a frame has a handful of boxes. Doing it per box turns a
    linear pass into a quadratic one for no benefit.
    """
    uv, depth = calib.project_velo(points_velo)
    keep = calib.in_image(uv, depth)
    return uv[keep], depth[keep]
