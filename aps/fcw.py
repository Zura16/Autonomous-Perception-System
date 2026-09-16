"""Forward-collision warning and AEB request: the decision layer.

Scope, stated as everywhere in this project: this emits a **warning** and an AEB
**request** in an offline, open-loop evaluation. It commands nothing. Production
AEB is an ISO 26262 / ISO 21448 function fed by radar and camera together; this
is a monocular decision rule whose purpose is to be measured.

The rule, per tracked object, per frame:

    candidate = closing (Phase 5 deadband passed)
              AND in the ego corridor (|lateral| < corridor half-width)
              AND TTC < warning threshold
    warning   = at least k of the last n frames were candidates
    request   = warning AND TTC < AEB threshold

Two parts carry the design:

**Persistence (k of n).** A single-frame TTC dip is how detector jitter becomes a
false alarm. Requiring several agreeing frames trades warning *latency* -- each
extra frame costs 0.104 s, straight out of the driver's reaction time -- for
false-alarm suppression. That trade is swept and measured, never assumed.

**The corridor is straight.** In-path means the object's lateral offset from the
camera centreline is below a half-width. On a curve an object in the ego lane can
sit outside a straight corridor, and an oncoming car can sit inside it. The same
straight corridor defines ground-truth threats, so the definition is consistent,
but it is not the true ego path, and curved-road errors are attributable to it
(docs/decisions.md D-024).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FcwConfig:
    ttc_warning_s: float
    ttc_aeb_s: float
    corridor_half_width_m: float
    persistence_k: int
    persistence_n: int
    fallback_range_m: float

    @classmethod
    def from_config(cls, path: Path | None = None, **override: float) -> FcwConfig:
        c = yaml.safe_load((path or REPO / "configs" / "fcw.yaml").read_text())
        values = dict(
            ttc_warning_s=float(c["thresholds"]["ttc_warning_s"]),
            ttc_aeb_s=float(c["thresholds"]["ttc_aeb_s"]),
            corridor_half_width_m=float(c["corridor"]["half_width_m"]),
            persistence_k=int(c["persistence"]["k"]),
            persistence_n=int(c["persistence"]["n"]),
            fallback_range_m=float(c["corridor"]["fallback_range_m"]),
        )
        values.update(override)
        if not 1 <= values["persistence_k"] <= values["persistence_n"]:
            raise ValueError("persistence requires 1 <= k <= n")
        if values["ttc_aeb_s"] > values["ttc_warning_s"]:
            raise ValueError("an AEB request cannot fire earlier than the warning")
        return cls(**values)


def lateral_offset_m(box_centre_u_px: float, fx_px: float, cx_px: float, range_m: float) -> float:
    """Lateral offset of a box centre from the camera centreline, in metres.

    The pinhole relation `x = (u - cx)·D/f`. Positive is to the right.
    """
    return float((box_centre_u_px - cx_px) * range_m / fx_px)


def corridor_range_m(contact_m: float, size_prior_m: float, cfg: FcwConfig) -> float:
    """Range used ONLY to place an object laterally for the corridor test.

    Contact-point first (best inside 20 m, Phase 3), then the size prior, then a
    fixed fallback. The fallback exists because both estimators abstain on a box
    clipped at the image bottom -- which is exactly an object very close ahead,
    the case a collision warning most needs to consider. Abstaining from the
    corridor test there would silently exempt the nearest threats. TTC never
    uses this value.
    """
    for r in (contact_m, size_prior_m):
        if np.isfinite(r) and r > 0:
            return float(r)
    return cfg.fallback_range_m


@dataclass
class _TrackState:
    history: deque = field(default_factory=deque)
    warning: bool = False


@dataclass(frozen=True)
class FcwOutput:
    candidate: bool
    warning: bool
    warning_onset: bool  # this frame turned the warning ON
    aeb_request: bool


class FcwDecider:
    """Per-track persistence state machine. Pure logic; no I/O, no image."""

    def __init__(self, cfg: FcwConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[object, _TrackState] = {}

    def reset(self, track_id: object) -> None:
        """Forget a track -- on an ID switch its history belongs to another object."""
        self._tracks.pop(track_id, None)

    def update(
        self, track_id: object, is_closing: bool, ttc_s: float, lateral_m: float
    ) -> FcwOutput:
        c = self.cfg
        state = self._tracks.setdefault(track_id, _TrackState(deque(maxlen=c.persistence_n)))
        candidate = bool(
            is_closing
            and np.isfinite(ttc_s)
            and ttc_s < c.ttc_warning_s
            and np.isfinite(lateral_m)
            and abs(lateral_m) < c.corridor_half_width_m
        )
        state.history.append(candidate)
        was = state.warning
        state.warning = sum(state.history) >= c.persistence_k
        return FcwOutput(
            candidate=candidate,
            warning=state.warning,
            warning_onset=state.warning and not was,
            aeb_request=state.warning and candidate and ttc_s < c.ttc_aeb_s,
        )


# ── evaluation primitives ────────────────────────────────────────────────────


def threat_events(
    frames: np.ndarray, is_threat: np.ndarray, max_gap: int = 2
) -> list[tuple[int, int]]:
    """Group one object's threat frames into events: (first_frame, last_frame).

    Frames separated by at most `max_gap` missing frames stay one event, so a
    one-frame detection dropout does not split a single approach into two events
    and double-count it in the denominator.
    """
    order = np.argsort(frames)
    f, t = np.asarray(frames)[order], np.asarray(is_threat, dtype=bool)[order]
    idx = f[t]
    if len(idx) == 0:
        return []
    events, start, prev = [], int(idx[0]), int(idx[0])
    for x in idx[1:]:
        if x - prev > max_gap + 1:
            events.append((start, prev))
            start = int(x)
        prev = int(x)
    events.append((start, prev))
    return events


def poisson_interval(count: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) two-sided interval for a Poisson count."""
    from scipy.stats import chi2

    a = 1.0 - conf
    lo = 0.0 if count == 0 else chi2.ppf(a / 2, 2 * count) / 2
    hi = chi2.ppf(1 - a / 2, 2 * (count + 1)) / 2
    return float(lo), float(hi)
