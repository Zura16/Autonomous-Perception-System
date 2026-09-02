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
        )

    def prior_height_m(self, group: str) -> float:
        return float(self.priors[group]["height_m"])

    def prior_std_m(self, group: str) -> float:
        return float(self.priors[group]["height_std_m"])


def horizon_row_px(calib: Calibration, pitch_deg: float = 0.0, cam: int = 2) -> float:
    """Image row of the horizon.

    For a level camera this is exactly the principal point row `cy`: parallel
    ground lines meet at infinity, which projects there. Pitching the camera
    nose-down by theta raises the horizon in the image by `fy * tan(theta)`.
    """
    _, fy, _, cy = calib.intrinsics(cam)
    return cy - fy * np.tan(np.radians(pitch_deg))


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

    Abstains in two cases, both of them ill-posed rather than merely hard:

      * the contact point is at or above the horizon -- the relation diverges,
        and such a box is floating, mis-detected, or on an upslope the
        flat-ground model does not describe;
      * the box is clipped at the bottom image edge -- the contact point is
        outside the picture, so the bottom edge measures the sensor boundary
        rather than the object.

    Returning a number in either case would be a confident answer to a question
    the geometry cannot pose.
    """
    if box_is_clipped_at_bottom(box, calib, camera, cam):
        return float("nan")
    _, fy, _, _ = calib.intrinsics(cam)
    pitch = camera.pitch_deg if pitch_deg is None else pitch_deg
    horizon = horizon_row_px(calib, pitch, cam)
    below = float(box[3]) - horizon
    if not np.isfinite(below) or below < camera.min_pixels_below_horizon:
        return float("nan")
    return float(fy * camera.height_m / below)


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
