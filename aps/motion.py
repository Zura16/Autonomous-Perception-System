"""Closing speed and time-to-collision from the scale relation -- never by
differentiating range.

The state is the **inverse apparent height** `u = 1/h`, and that choice is the
whole design:

    h = f·H/D          so        u = 1/h = D/(f·H)      i.e.  u is PROPORTIONAL to range

Under constant relative velocity `D(t)` is linear in `t`, so `u(t)` is **exactly**
linear too. A constant-velocity Kalman filter over `[u, u̇]` is therefore not an
approximation of the dynamics -- it is the exact model, under the same assumption
TTC itself already makes. Filtering `h` directly would be wrong: `h ∝ 1/D` grows
hyperbolically as an object approaches, so a constant-rate model on `h` is
mis-specified precisely when the object matters most.

Everything downstream falls out of that one state:

    TTC = −D/Ḋ = −u/u̇           calibration-free and prior-free: f and H cancel
    Ḋ   = −D/TTC = D·u̇/u        needs the Phase 3 range estimate for the scale

So range, closing speed and TTC derive from one consistent state and **cannot
contradict each other** (CLAUDE.md hard rule 3). Finite-differencing range would
turn the 0.25 m ground-truth error into ~3.4 m/s of phantom closing speed at
9.657 Hz -- see docs/error-budget.md Stage 2.

**Two consequences of TTC being a ratio, both measured elsewhere in this project:**

  * The detector's **multiplicative box-height bias cancels exactly.** If
    `h_meas = k·h_true` for constant `k`, then `u` and `u̇` both scale by `1/k`
    and `TTC = −u/u̇` is unchanged. The −4.5% bias that distorts Phase 3's range
    (Row 2.7) costs TTC nothing.
  * It does not cancel when `k` VARIES with range, which it does (−7.1% near,
    −2.4% at 20–30 m). A changing `k` during an approach injects a spurious
    scale rate -- a second-order term quantified in docs/decisions.md D-022.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

# Sentinel for "not closing". Infinite rather than a large number so that any
# threshold comparison behaves, and so it can never be mistaken for a measurement.
NOT_CLOSING = float("inf")


@dataclass(frozen=True)
class MotionConfig:
    """Filter and decision parameters. All from configs/motion.yaml."""

    rel_height_sigma: float  # MEASURED: detector box-height noise, fraction of h
    rel_height_sigma_far: float
    far_range_px: float  # box height below which the far sigma applies
    rel_accel_sigma: float  # process noise, units 1/s^2 (prior, not measured)
    deadband_inv_ttc: float  # |1/TTC| below this is declared not-closing
    deadband_sigmas: float  # n-sigma significance required on u̇
    min_updates: int  # filter outputs nothing before this many observations

    @classmethod
    def from_config(cls, path: Path | None = None) -> MotionConfig:
        cfg = yaml.safe_load((path or REPO / "configs" / "motion.yaml").read_text())
        return cls(
            rel_height_sigma=float(cfg["measurement"]["rel_height_sigma"]),
            rel_height_sigma_far=float(cfg["measurement"]["rel_height_sigma_far"]),
            far_range_px=float(cfg["measurement"]["far_range_px"]),
            rel_accel_sigma=float(cfg["process"]["rel_accel_sigma"]),
            deadband_inv_ttc=float(cfg["deadband"]["inv_ttc"]),
            deadband_sigmas=float(cfg["deadband"]["sigmas"]),
            min_updates=int(cfg["filter"]["min_updates"]),
        )


@dataclass(frozen=True)
class MotionState:
    """One track's motion estimate at one frame. Units live in every name."""

    ttc_s: float  # +inf when not closing
    closing_speed_mps: float  # NEGATIVE when approaching (range decreasing)
    range_m: float
    inv_height: float  # u
    inv_height_rate: float  # u̇
    inv_height_rate_sigma: float  # filter's own 1-sigma on u̇
    n_updates: int
    is_closing: bool
    deadband_applied: bool

    @property
    def is_valid(self) -> bool:
        return self.n_updates > 0 and np.isfinite(self.inv_height)


class InverseHeightKF:
    """Constant-velocity Kalman filter over `[u, u̇]`, `u = 1/h`.

    Both noise models are **multiplicative**, because the measurement is:

      * `h` has roughly constant RELATIVE error (measured: 5.5-6.9% inside 30 m,
        13.4% beyond -- Row 2.7 / D-022), so `σ_h = σ_rel·h`. Propagating to
        `u = 1/h` gives `σ_u = σ_h/h² = σ_rel/h = σ_rel·u`: the relative error
        carries through unchanged, which is a good reason to work in `u`.
      * process noise is set by a *relative* acceleration `σ_ar` (units 1/s²),
        i.e. `ü/u = D̈/D`. A car braking at 5 m/s² at 20 m has `D̈/D = 0.25/s²`,
        which is the scale of the configured prior.

    The relative measurement noise is measured; the process noise is a prior and
    is labelled as one (hard rule 6 applies to filter priors too).
    """

    def __init__(self, u0: float, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self.x = np.array([u0, 0.0])
        # No rate information from one observation: start the rate variance wide
        # enough that the first two updates, not the prior, determine it.
        self.P = np.diag([(cfg.rel_height_sigma * u0) ** 2, (u0 * 2.0) ** 2])
        self.n_updates = 1

    def _sigma_rel(self, height_px: float) -> float:
        return (
            self.cfg.rel_height_sigma_far
            if height_px < self.cfg.far_range_px
            else self.cfg.rel_height_sigma
        )

    def predict(self, dt_s: float) -> None:
        F = np.array([[1.0, dt_s], [0.0, 1.0]])
        u = max(abs(self.x[0]), 1e-9)
        q = (self.cfg.rel_accel_sigma * u) ** 2
        Q = q * np.array([[dt_s**4 / 4.0, dt_s**3 / 2.0], [dt_s**3 / 2.0, dt_s**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, height_px: float) -> None:
        if height_px <= 0:
            return
        u = 1.0 / height_px
        r = (self._sigma_rel(height_px) * u) ** 2
        H = np.array([[1.0, 0.0]])
        y = u - (H @ self.x)[0]
        S = (H @ self.P @ H.T)[0, 0] + r
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        self.n_updates += 1

    @property
    def u(self) -> float:
        return float(self.x[0])

    @property
    def u_dot(self) -> float:
        return float(self.x[1])

    @property
    def u_dot_sigma(self) -> float:
        return float(np.sqrt(max(self.P[1, 1], 0.0)))


def ttc_from_state(u: float, u_dot: float) -> float:
    """`TTC = −u/u̇`, +inf unless the object is genuinely growing in the image.

    Returns +inf for a receding or static object rather than a negative or huge
    number: a negative TTC is not a time, and a huge one is a division artefact
    dressed as a measurement.
    """
    if u_dot >= 0 or u <= 0:
        return NOT_CLOSING
    return float(-u / u_dot)


class ScaleRateEstimator:
    """Per-track scale-rate filters producing range, closing speed and TTC.

    `deadband` selects how a not-closing object is rejected, and the two options
    are genuinely different claims:

      'fixed'  reject when `|1/TTC|` is below a constant -- a fixed horizon,
               simple to state and to defend, but blind to how noisy the track is.
      'sigma'  reject unless `u̇` is significantly negative at `n` sigma, using
               the filter's OWN covariance, so a jittery track must show a
               stronger signal than a clean one.

    **Measured result: at matched phantom rates the two are equivalent.** On
    synthetic stationary objects with 6% height noise, fixed@0.10 gives 1.5%
    phantom triggers and detects 68% of 3 m/s approaches; sigma@1.5 gives 1.5%
    and 66%. The adaptive mechanism buys nothing when every track has similar
    noise, and only pays when track quality VARIES -- which is a claim about
    the data, not about the mechanism (docs/decisions.md D-022).

    Whichever is used, the deadband trades **slow-approach sensitivity** for
    phantom suppression. 12 m/s closings are detected 100% of the time at every
    setting swept; 6 m/s at >=92% except the strictest fixed horizon (TTC > 3 s
    rejected), which catches only 31%.

    **On real val tracks a deadband never changes a TTC value, only whether one
    is emitted.** Measured on the frames all variants flag, TTC error is
    identical across none/fixed/sigma. sigma@2.0's lower headline error is
    selection: it withholds the worst estimates -- which are also genuine threats
    (GT TTC median 2.15 s). That trade belongs to Phase 7 (D-022).

    Hard rule 4 requires the threshold to be a logged decision with a measured
    false-trigger rate. The operating point is NOT set here -- Phase 7 sets it
    against FP/hour on real data.
    """

    def __init__(self, cfg: MotionConfig, deadband: str = "sigma") -> None:
        if deadband not in ("fixed", "sigma", "none"):
            raise ValueError(f"unknown deadband {deadband!r}; have 'fixed', 'sigma', 'none'")
        self.cfg = cfg
        self.deadband = deadband
        self._filters: dict[int, InverseHeightKF] = {}

    def _rejects(self, u: float, u_dot: float, sigma: float) -> bool:
        """True when a nominally closing state falls inside the deadband."""
        if self.deadband == "fixed":
            return abs(u_dot / u) < self.cfg.deadband_inv_ttc
        if self.deadband == "sigma":
            # u̇ must be significantly negative, not merely negative.
            return u_dot + self.cfg.deadband_sigmas * sigma >= 0
        return False

    def reset(self, track_id: int) -> None:
        """Drop a track's history. Called on an ID switch or a fragmentation.

        This matters more than it looks: Phase 4 measured 358 ID switches on val,
        concentrated at 10-20 m. Carrying a filter across a switch splices two
        objects' height histories and manufactures a scale rate from the seam,
        which is a phantom closing speed with a plausible-looking TTC.
        """
        self._filters.pop(track_id, None)

    def update(self, track_id: int, height_px: float, dt_s: float, range_m: float) -> MotionState:
        """Advance one track by one frame and report its motion state."""
        if height_px <= 0 or not np.isfinite(height_px):
            return MotionState(NOT_CLOSING, 0.0, range_m, np.nan, np.nan, np.nan, 0, False, False)

        kf = self._filters.get(track_id)
        if kf is None:
            kf = InverseHeightKF(1.0 / height_px, self.cfg)
            self._filters[track_id] = kf
        else:
            kf.predict(dt_s)
            kf.update(height_px)

        u, u_dot, sigma = kf.u, kf.u_dot, kf.u_dot_sigma
        ttc = ttc_from_state(u, u_dot)

        # A track with too little history has a rate dominated by its prior.
        if kf.n_updates < self.cfg.min_updates:
            return MotionState(
                NOT_CLOSING, 0.0, range_m, u, u_dot, sigma, kf.n_updates, False, False
            )

        deadbanded = u_dot < 0 and self._rejects(u, u_dot, sigma)
        closing = u_dot < 0 and not deadbanded

        if not closing:
            return MotionState(
                NOT_CLOSING, 0.0, range_m, u, u_dot, sigma, kf.n_updates, False, deadbanded
            )

        # Closing speed is negative when approaching, and is the ONLY quantity
        # here that needs the range estimate -- TTC never does.
        closing_speed = -range_m / ttc if np.isfinite(ttc) and ttc > 0 else 0.0
        return MotionState(ttc, closing_speed, range_m, u, u_dot, sigma, kf.n_updates, True, False)
