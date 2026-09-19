"""Monocular range from a 2D box. The estimators Phase 1's ruler exists to grade.

Two estimators with **orthogonal failure modes**, which is why both are kept:

    contact-point   D = f_y * h_cam / (v_bottom - v_horizon)
                    Uses where the object meets the road. Needs the camera
                    height and the horizon row; needs NO object prior.
                    Pitch error costs  dD/D ~ tan(dtheta) * D / h_cam
                    -- it GROWS LINEARLY WITH RANGE.

    size-prior      D = f_y * H_object / h_box
                    Uses apparent height against an assumed true height. Needs
                    NO ground plane, so it is completely pitch-insensitive.
                    Prior error costs  dD/D = dH/H
                    -- CONSTANT with range.

So contact-point should win near and size-prior far, crossing where the pitch
cost equals the prior spread. With the measured val pitch (|p95| 0.715 deg,
typical 0.319 deg), h_cam 1.655 m, and the POOLED vehicle prior spread of 20.1%
forced by class-agnostic detection (docs/decisions.md D-014), the predicted
crossover for vehicles sits near 60 m -- beyond the project's 0-50 m evidence
envelope (D-015). For VRUs, whose pooled spread is 8.7%, it sits near 26 m.

That is a prediction derived from the error budget. eval/eval_range.py measures
whether it holds; docs/error-budget.md carries the derivation.

Nothing in this module may read LiDAR. The ground plane and pitch it assumes are
fixed nominals from configs/camera.yaml, exactly as a deployed camera-only
system would have.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from aps.kitti.calibration import Calibration

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CameraModel:
    """Fixed mounting assumptions and object priors. No LiDAR, no per-frame state."""

    height_m: float
    height_std_m: float
    pitch_deg: float
    priors: dict[str, dict]
    min_rows_from_bottom: int
    min_pixels_below_horizon: float
    max_range_m: float

    @classmethod
    def from_config(cls, path: Path | None = None) -> CameraModel:
        cfg = yaml.safe_load((path or REPO / "configs" / "camera.yaml").read_text())
        return cls(
            height_m=float(cfg["camera"]["height_m"]),
            height_std_m=float(cfg["camera"]["height_std_m"]),
            pitch_deg=float(cfg["camera"]["pitch_deg"]),
            priors=cfg["priors"],
            min_rows_from_bottom=int(cfg["contact_point"]["min_rows_from_bottom"]),
            min_pixels_below_horizon=float(cfg["contact_point"]["min_pixels_below_horizon"]),
            max_range_m=float(cfg["contact_point"]["max_range_m"]),
        )

    def prior_height_m(self, group: str) -> float:
        return float(self.priors[group]["height_m"])

    def prior_std_m(self, group: str) -> float:
        return float(self.priors[group]["height_std_m"])


@dataclass(frozen=True)
class CredibleEnvelope:
    """The range band the project vouches for, and what it does NOT cover.

    Deliberately separate from `CameraModel`: that class holds mounting
    assumptions the estimator consumes, whereas this is a *measured result*
    about the estimator's error. Mixing them would let a display read an
    evaluation conclusion as though it were a calibration constant.
    """

    min_range_m: float
    max_range_m: float
    max_mape_pct: float
    gates_ttc: bool

    @classmethod
    def from_config(cls, path: Path | None = None) -> CredibleEnvelope:
        cfg = yaml.safe_load((path or REPO / "configs" / "camera.yaml").read_text())
        e = cfg["credible_envelope"]
        return cls(
            min_range_m=float(e["min_range_m"]),
            max_range_m=float(e["max_range_m"]),
            max_mape_pct=float(e["max_mape_pct"]),
            gates_ttc=bool(e["gates_ttc"]),
        )

    def contains(self, range_m: float) -> bool:
        """Whether a range estimate falls inside the vouched band.

        Both bounds, not just the far one. The near field is outside the band
        (21-22% MAPE on both splits), and checking only the upper bound quietly
        published exactly the values the evaluation is least able to support.
        """
        return bool(np.isfinite(range_m) and self.min_range_m <= range_m <= self.max_range_m)


def horizon_row_px(calib: Calibration, pitch_deg: float = 0.0, cam: int = 2) -> float:
    """Image row of the horizon.

    For a level camera this is exactly the principal point row `cy`: parallel
    ground lines meet at infinity, which projects there. Pitching the camera
    nose-down by theta raises the horizon in the image by `fy * tan(theta)`.
    """
    _, fy, _, cy = calib.intrinsics(cam)
    return cy - fy * np.tan(np.radians(pitch_deg))


def min_supportable_range_m(
    calib: Calibration, camera: CameraModel, pitch_deg: float | None = None, cam: int = 2
) -> float:
    """Closest range whose ground contact point still lands inside the image.

    A contact point at range D projects to row `horizon + fy*h/D`, so it leaves
    an `H_img`-row image below

        D_min = fy * h_cam / (H_img - horizon)

    which is **5.91 m** for this camera. Closer than that the contact point is
    not in the picture at all: the box bottom can reach the frame edge and no
    further, and the estimator saturates there. Measured, sub-D_min estimates
    cluster at 6.03 m (val) / 6.02 m (test) and are over-estimates 100% of the
    time on both splits (docs/decisions.md D-026).

    This is a property of the mounting and the sensor, not of the algorithm: a
    camera 1.655 m up with this field of view cannot see the wheels of a car
    four metres in front of it. A production stack covers that band with a
    wider-FOV camera, a lower mount, or radar -- not with a better estimator.

    Not used as a gate, because saturation makes the estimate itself land at or
    above D_min by construction and so carries no signal. The gate is
    `box_is_clipped_at_bottom`. This function exists to state the limit, to test
    it, and to keep the number derived rather than retyped.
    """
    _, fy, _, _ = calib.intrinsics(cam)
    pitch = camera.pitch_deg if pitch_deg is None else pitch_deg
    horizon = horizon_row_px(calib, pitch, cam)
    return float(fy * camera.height_m / (calib.image_size[cam][1] - horizon))


def box_is_clipped_at_bottom(
    box: np.ndarray, calib: Calibration, camera: CameraModel, cam: int = 2
) -> bool:
    """True when the box runs into the bottom image edge.

    Such an object continues below the visible image, so its ground contact
    point is not observable and its pixel height is truncated -- BOTH estimators
    are reading a boundary rather than the object.

    Measured on val: 4.2% of detections are clipped this way, and they carry
    **46.4% MAPE against 13.6%** for unclipped boxes, biased +1.66 m at close
    range because the clipped bottom edge sits higher than the true contact row.
    They dominate the near-field error (docs/decisions.md D-017).

    The margin matters as much as the test. At 2 rows this caught only boxes
    literally touching the edge and missed the objects that actually break the
    near field -- they sit at y2 ~ 371 of 375, four pixels clear, at a true range
    of ~4 m, below the `min_supportable_range_m` of 5.91 m where the contact
    point is no longer in the picture. Raised to 10 on val evidence: unsupportable
    objects sit a median 4 px from the edge against 140 px for valid ones, so the
    gate catches 83% of them for 0.34% of valid boxes, and val 0-10 m MAPE falls
    21.0 -> 12.4% (contact-point) and 29.5 -> 19.9% (size-prior) with every other
    bin untouched (docs/decisions.md D-026).
    """
    height_px = calib.image_size[cam][1]
    return float(box[3]) >= height_px - camera.min_rows_from_bottom


def range_contact_point(
    box: np.ndarray,
    calib: Calibration,
    camera: CameraModel,
    pitch_deg: float | None = None,
    cam: int = 2,
) -> float:
    """Range from where the object's box meets the road. NaN if unusable.

    `box` is [x1, y1, x2, y2]; the bottom edge `y2` is taken as the contact row.

    Abstains in three cases, all ill-posed rather than merely hard:

      * the contact point is at or above the horizon -- the relation diverges,
        and such a box is floating, mis-detected, or on an upslope the
        flat-ground model does not describe;
      * the box is clipped at the bottom image edge -- the contact point is
        outside the picture, so the bottom edge measures the sensor boundary
        rather than the object;
      * the implied range exceeds `camera.max_range_m` -- the box bottom is so
        close to the horizon that the answer lies beyond every range this
        estimator has been characterised at, and one pixel of box-edge error
        moves it by more than the claimed accuracy.

    Returning a number in any of these cases would be a confident answer to a
    question the geometry cannot pose. The third case was found on the held-out
    test split, where 3.7% of one drive's detections landed inside it and
    produced estimates up to 369 m, inflating mean error in every range bin they
    touched (docs/decisions.md D-025).
    """
    if box_is_clipped_at_bottom(box, calib, camera, cam):
        return float("nan")
    _, fy, _, _ = calib.intrinsics(cam)
    pitch = camera.pitch_deg if pitch_deg is None else pitch_deg
    horizon = horizon_row_px(calib, pitch, cam)
    below = float(box[3]) - horizon
    if not np.isfinite(below) or below < camera.min_pixels_below_horizon:
        return float("nan")
    range_m = float(fy * camera.height_m / below)
    # Tested in range space so the code states the rule the config states, with
    # one source of truth rather than an equivalent pixel threshold kept in sync
    # by hand. An object sitting exactly at max_range_m still falls either way on
    # rounding; that boundary is arbitrary by construction and nothing depends on
    # which side it lands.
    if range_m > camera.max_range_m:
        return float("nan")
    return range_m


def range_size_prior(
    box: np.ndarray,
    calib: Calibration,
    camera: CameraModel,
    group: str,
    cam: int = 2,
) -> float:
    """Range from apparent height against an assumed true height. NaN if unusable.

    Pitch-independent by construction: no ground plane appears in `D = f*H/h`.
    Its error is the prior's error, transferred one-for-one and unchanged with
    range -- which is the whole reason for keeping it alongside contact-point.

    Abstains on a box clipped at the bottom image edge for a different reason
    than contact-point does: the object's pixel *height* is truncated, so `h` is
    an underestimate and the range comes out too large.
    """
    if box_is_clipped_at_bottom(box, calib, camera, cam):
        return float("nan")
    _, fy, _, _ = calib.intrinsics(cam)
    height_px = float(box[3]) - float(box[1])
    if not np.isfinite(height_px) or height_px <= 0:
        return float("nan")
    return float(fy * camera.prior_height_m(group) / height_px)


def pitch_range_error_frac(pitch_deg: float, range_m: float, camera_height_m: float) -> float:
    """Fractional range error a contact-point estimator suffers from pitch error.

    EXACT, not linearised. Assuming zero pitch when the truth is `theta`:

        D_est = D / (1 - x)        with  x = tan(theta) * D / h_cam
        dD/D  = x / (1 - x)

    The commonly quoted `dD/D ~ x` is the first-order expansion and it
    **understates the error badly at range**, because `x` is not small there.
    At the measured val p95 pitch (0.715 deg) and h_cam 1.655 m:

        range    x      linear    exact
         10 m   0.075    7.5%      8.2%
         20 m   0.151   15.1%     17.8%
         30 m   0.226   22.6%     29.2%
         50 m   0.377   37.7%     60.5%

    A unit test against a synthetically pitched projection caught the
    linearisation being used as if it were exact (docs/decisions.md D-016).

    The error diverges as `x -> 1`, i.e. at `D = h_cam / tan(theta)` -- 132 m at
    p95 pitch. That is the range at which the contact point reaches the assumed
    horizon, and `range_contact_point` abstains there rather than returning a
    huge number.
    """
    x = np.tan(np.radians(pitch_deg)) * range_m / camera_height_m
    if x >= 1.0:
        return float("inf")
    return float(x / (1.0 - x))


def prior_range_error_frac(prior_error_m: float, true_height_m: float) -> float:
    """Fractional range error a size-prior estimator suffers from prior error.

    `D = f*H/h` is linear in H, so the prior's fractional error passes through
    unchanged and is independent of range.
    """
    return float(prior_error_m / true_height_m)


def crossover_range_m(prior_cv: float, pitch_deg: float, camera_height_m: float) -> float:
    """Range at which size-prior error overtakes contact-point pitch error.

    Solves the EXACT relation `x / (1 - x) = prior_cv` for `x = tan(theta)*D/h`,
    giving `x = cv / (1 + cv)` and `D = h * cv / (tan(theta) * (1 + cv))`.

    Using the linearised form instead pushes the crossover ~20% further out
    (vehicles: 59.7 m linear vs 49.8 m exact at typical pitch), which would have
    put the crossover outside the 0-50 m evidence envelope when in fact it sits
    just inside it. Below the crossover contact-point wins; above it the prior.
    """
    t = np.tan(np.radians(pitch_deg))
    if t <= 0:
        return float("inf")
    return float(camera_height_m * prior_cv / (t * (1.0 + prior_cv)))
