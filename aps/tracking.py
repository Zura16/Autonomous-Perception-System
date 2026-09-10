"""2D multi-object tracking: an IoU baseline and SORT.

Two trackers, kept together so the comparison is like-for-like -- same birth and
death policy, same association threshold, differing only in the part under test:

    IoUTracker    associates on raw IoU between the LAST OBSERVED box and the
                  current detections. No motion model at all. The honest
                  baseline: if SORT does not beat this, its Kalman filter is
                  not earning its complexity.

    SortTracker   Bewley et al. 2016. A linear Kalman filter over the image-plane
                  box state, predicting where each track SHOULD be before
                  associating, plus optimal (Hungarian) assignment instead of
                  greedy. The two changes are separable and both are measured.

Why the tracker matters here beyond hard rule 8: Phase 5 derives closing speed
from the scale relation `Ḋ = −D·ḣ/h`, which needs a stable per-object box height
history. An ID switch does not merely dent a metric -- it splices two different
objects' height sequences together and manufactures a closing speed from the
discontinuity. Tracking quality is a *safety* input here, not bookkeeping.

Both trackers work in the image plane on pixel boxes. Neither sees range, LiDAR,
or class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from aps.matching import iou_matrix


@dataclass
class Track:
    """One tracked object's current state."""

    track_id: int
    box: np.ndarray  # [x1, y1, x2, y2] pixels
    score: float
    hits: int = 1  # total detections associated
    age: int = 0  # frames since birth
    time_since_update: int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 1


def _box_to_z(box: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [u, v, s, r]: centre, area, aspect ratio.

    Area-and-aspect rather than width-and-height because a box's area changes
    smoothly as an object approaches while its aspect stays roughly constant --
    so a constant-velocity model on `s` is a reasonable motion prior, whereas one
    on `w` and `h` separately would let a box drift out of proportion.
    """
    w = max(box[2] - box[0], 1e-6)
    h = max(box[3] - box[1], 1e-6)
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h])


def _z_to_box(z: np.ndarray) -> np.ndarray:
    """[u, v, s, r] -> [x1,y1,x2,y2]. Inverse of `_box_to_z`."""
    s = max(float(z[2]), 1e-6)
    r = max(float(z[3]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])


class BoxKalmanFilter:
    """Constant-velocity Kalman filter over [u, v, s, r] with velocities on u,v,s.

    State: [u, v, s, r, u̇, v̇, ṡ] -- seven elements. Aspect ratio `r` carries no
    velocity: a box that is steadily changing shape is usually a detector
    artefact or an occlusion, not motion, and modelling it invites the filter to
    extrapolate nonsense during a miss.

    Covariances follow the reference SORT implementation. They are unusual on
    purpose and worth being able to defend:

      * velocities start with **1000x** the position uncertainty, because a
        newborn track has one observation and therefore no velocity information
        at all -- the filter must not trust its own zero-initialised velocity;
      * scale and aspect measurements are trusted **10x less** than centre
        position, because detector boxes jitter in size far more than in centre;
      * the process noise on the velocity block and on `r` is small, encoding
        "motion is nearly constant between frames at 10 Hz".

    These are priors, not fitted values, and they are logged as such
    (docs/decisions.md D-021).
    """

    def __init__(self, box: np.ndarray) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # size measurements are noisier than centre

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # no velocity information from one frame

        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros(7)
        self.x[:4] = _box_to_z(box)

    def predict(self) -> np.ndarray:
        # Guard against a scale that has been driven negative by an
        # extrapolated shrink -- a negative area has no box.
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _z_to_box(self.x[:4])

    def update(self, box: np.ndarray) -> None:
        z = _box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return _z_to_box(self.x[:4])


@dataclass
class _SortTrack:
    track_id: int
    kf: BoxKalmanFilter
    score: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(4))
    last_detection: np.ndarray = field(default_factory=lambda: np.zeros(4))


class IoUTracker:
    """Greedy IoU association against each track's last observed box.

    Deliberately has **no motion model**. Between frames a track simply keeps its
    previous box, so association fails exactly when an object moves more than
    roughly its own size per frame -- which at 9.657 Hz is a fast crossing object
    or a fast approach. That failure mode is the baseline's whole point: it makes
    visible what a motion model buys.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 1, min_hits: int = 1) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1

        matched_det: set[int] = set()
        if self._tracks and len(boxes):
            ious = iou_matrix(np.array([t.box for t in self._tracks]), boxes)
            # Greedy on descending IoU: take the most confident pairing first and
            # remove both from play. Hungarian would maximise total IoU instead;
            # SortTracker uses that, so the difference is measurable.
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            used_t: set[int] = set()
            for ti, di in order:
                if ious[ti, di] < self.iou_threshold:
                    break
                if ti in used_t or di in matched_det:
                    continue
                track = self._tracks[ti]
                track.box = boxes[di]
                track.score = float(scores[di])
                track.hits += 1
                track.time_since_update = 0
                used_t.add(int(ti))
                matched_det.add(int(di))

        for di in range(len(boxes)):
            if di not in matched_det:
                self._tracks.append(Track(self._next_id, boxes[di], float(scores[di])))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return [t for t in self._tracks if t.time_since_update == 0 and t.hits >= self.min_hits]


class SortTracker:
    """SORT: Kalman prediction + Hungarian assignment on IoU.

    Two changes from the IoU baseline, both separable:
      1. associate against the **predicted** box rather than the last observed
         one, so a moving object is looked for where it is going;
      2. assign **optimally** rather than greedily, so one bad early pairing
         cannot cascade.

    `max_age > 1` lets a track survive a missed detection and keep its ID. That
    is the mechanism that should cut ID switches, and it is why the comparison
    holds `max_age` equal between the two trackers rather than tuning each.

    `output_filtered` controls WHICH BOX is emitted, and it is a measured choice
    rather than an incidental one. Canonical SORT reports the Kalman state, which
    smooths detector jitter but also drags the box away from the observation the
    detector actually made. The IoU baseline emits the raw detection. Leaving
    that difference unexamined would confound association quality with smoothing
    quality, so both settings are evaluated (docs/decisions.md D-021).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 1,
        min_hits: int = 1,
        output_filtered: bool = True,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.output_filtered = output_filtered
        self._tracks: list[_SortTrack] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        for t in self._tracks:
            t.predicted = t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        matched_det: set[int] = set()
        matched_trk: set[int] = set()
        if self._tracks and len(boxes):
            ious = iou_matrix(np.array([t.predicted for t in self._tracks]), boxes)
            # linear_sum_assignment minimises, so negate to maximise total IoU.
            rows, cols = linear_sum_assignment(-ious)
            for ti, di in zip(rows, cols, strict=True):
                if ious[ti, di] < self.iou_threshold:
                    continue  # assigned but not plausible; leave both unmatched
                track = self._tracks[ti]
                track.kf.update(boxes[di])
                track.last_detection = boxes[di]
                track.score = float(scores[di])
                track.hits += 1
                track.time_since_update = 0
                matched_trk.add(int(ti))
                matched_det.add(int(di))

        for di in range(len(boxes)):
            if di not in matched_det:
                self._tracks.append(
                    _SortTrack(
                        self._next_id,
                        BoxKalmanFilter(boxes[di]),
                        float(scores[di]),
                        last_detection=boxes[di],
                    )
                )
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return [
            Track(
                t.track_id,
                t.kf.box if self.output_filtered else t.last_detection,
                t.score,
                t.hits,
                t.age,
                t.time_since_update,
            )
            for t in self._tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]


def make_tracker(name: str, **kwargs) -> IoUTracker | SortTracker:
    """Build a tracker by name.

    'sort'     canonical SORT -- emits the Kalman-filtered box.
    'sort_det' identical association, but emits the raw associated detection.
               The pair isolates association quality from smoothing quality.
    """
    if name == "iou":
        return IoUTracker(**kwargs)
    if name == "sort":
        return SortTracker(output_filtered=True, **kwargs)
    if name == "sort_det":
        return SortTracker(output_filtered=False, **kwargs)
    raise ValueError(f"unknown tracker {name!r}; have 'iou', 'sort', 'sort_det'")
