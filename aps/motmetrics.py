"""CLEAR MOT and IDF1, implemented here rather than imported.

**Caveat that travels with every number this produces:** this is a
self-implemented metric validated against hand-computed synthetic cases, not the
official KITTI/MOTChallenge devkit. Combined with the raw-tracklet labels
([D-004](../docs/decisions.md)), tracking figures from this project are valid for
the comparison it actually makes -- IoU baseline vs SORT on identical data -- and
are **not** comparable to any published leaderboard.

Two metric families, measuring different things, which is why both are required:

    MOTA   detection-flavoured. Counts misses, false positives and ID switches
           against the number of ground-truth objects. Dominated by detection
           quality: a tracker fed a poor detector scores badly no matter how
           good its association is.

    IDF1   identity-flavoured. Solves a GLOBAL one-to-one matching between
           ground-truth identities and hypothesis identities over the whole
           sequence, then scores how much of each trajectory was covered by the
           right identity. A tracker that fragments an object into five IDs can
           still post a decent MOTA (each fragment is a match) while IDF1
           collapses -- which is the case Phase 5 cares about.

MOTA can also go negative: with more false positives than ground-truth objects,
`1 - (FN+FP+IDSW)/GT` drops below zero. That is a feature, not an overflow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from aps.matching import iou_matrix

DEFAULT_IOU = 0.5


@dataclass(frozen=True)
class MOTResult:
    """Aggregated tracking metrics with the raw counts that produced them."""

    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    precision: float
    recall: float
    num_switches: int
    num_fp: int
    num_misses: int
    num_matches: int
    num_gt: int
    num_hyp: int
    num_frames: int
    num_gt_ids: int
    num_hyp_ids: int
    fragmentations: int

    def summary(self) -> str:
        return (
            f"MOTA {self.mota:6.3f}  MOTP {self.motp:5.3f}  IDF1 {self.idf1:6.3f}  "
            f"IDSW {self.num_switches:4d}  FP {self.num_fp:5d}  FN {self.num_misses:5d}  "
            f"GT {self.num_gt}"
        )


@dataclass
class MOTAccumulator:
    """Frame-by-frame accumulator for one or more sequences.

    Call `new_sequence()` between sequences: identity continuity must not carry
    across a cut, or the last track of one drive gets credited as an ID switch
    against the first object of the next.
    """

    iou_threshold: float = DEFAULT_IOU

    num_frames: int = 0
    num_matches: int = 0
    num_fp: int = 0
    num_misses: int = 0
    num_switches: int = 0
    num_gt: int = 0
    num_hyp: int = 0
    fragmentations: int = 0
    _iou_sum: float = 0.0
    # (sequence, gt_id) -> last hypothesis id it was matched to
    _last_match: dict[tuple[int, object], object] = field(default_factory=dict)
    # (sequence, gt_id) -> was it matched in the previous frame
    _was_tracked: dict[tuple[int, object], bool] = field(default_factory=dict)
    # co-occurrence counts for IDF1: (seq, gt_id, hyp_id) -> frames matched
    _pairs: dict[tuple[int, object, object], int] = field(default_factory=lambda: defaultdict(int))
    _gt_len: dict[tuple[int, object], int] = field(default_factory=lambda: defaultdict(int))
    # per-ground-truth-object switch counts, so the harness can stratify them by
    # range without duplicating the correspondence bookkeeping
    switches_by_gt: dict[tuple[int, object], int] = field(default_factory=lambda: defaultdict(int))
    _hyp_len: dict[tuple[int, object], int] = field(default_factory=lambda: defaultdict(int))
    _sequence: int = 0

    def new_sequence(self) -> None:
        self._sequence += 1

    def update(
        self,
        gt_ids: list,
        gt_boxes: np.ndarray,
        hyp_ids: list,
        hyp_boxes: np.ndarray,
    ) -> None:
        """Score one frame."""
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        hyp_boxes = np.asarray(hyp_boxes, dtype=np.float64).reshape(-1, 4)
        self.num_frames += 1
        self.num_gt += len(gt_ids)
        self.num_hyp += len(hyp_ids)
        for g in gt_ids:
            self._gt_len[(self._sequence, g)] += 1
        for h in hyp_ids:
            self._hyp_len[(self._sequence, h)] += 1

        ious = iou_matrix(gt_boxes, hyp_boxes)
        valid = ious >= self.iou_threshold

        matched_gt: set[int] = set()
        matched_hyp: set[int] = set()
        matches: list[tuple[int, int]] = []

        # Pass 1 -- PRESERVE EXISTING CORRESPONDENCES. The CLEAR MOT rule: if a
        # ground-truth object was matched to hypothesis h and that pairing is
        # still valid, keep it even when another hypothesis now overlaps more.
        # Without this the metric invents ID switches every time two objects
        # pass close to each other, and reports the metric's own instability as
        # the tracker's.
        for gi, g in enumerate(gt_ids):
            prev = self._last_match.get((self._sequence, g))
            if prev is None:
                continue
            for hj, h in enumerate(hyp_ids):
                if h == prev and valid[gi, hj]:
                    matches.append((gi, hj))
                    matched_gt.add(gi)
                    matched_hyp.add(hj)
                    break

        # Pass 2 -- optimally assign whatever is left.
        free_gt = [i for i in range(len(gt_ids)) if i not in matched_gt]
        free_hyp = [j for j in range(len(hyp_ids)) if j not in matched_hyp]
        if free_gt and free_hyp:
            sub = ious[np.ix_(free_gt, free_hyp)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols, strict=True):
                if sub[r, c] >= self.iou_threshold:
                    matches.append((free_gt[r], free_hyp[c]))
                    matched_gt.add(free_gt[r])
                    matched_hyp.add(free_hyp[c])

        for gi, hj in matches:
            g, h = gt_ids[gi], hyp_ids[hj]
            key = (self._sequence, g)
            prev = self._last_match.get(key)
            if prev is not None and prev != h:
                self.num_switches += 1
                self.switches_by_gt[key] += 1
            # A fragmentation is a track resuming after a gap, whether or not the
            # identity changed -- it is the discontinuity that breaks a scale-rate
            # history, so it is counted separately from an ID switch.
            if prev is not None and not self._was_tracked.get(key, False):
                self.fragmentations += 1
            self._last_match[key] = h
            self._pairs[(self._sequence, g, h)] += 1
            self._iou_sum += float(ious[gi, hj])

        for g in gt_ids:
            self._was_tracked[(self._sequence, g)] = False
        for gi, _ in matches:
            self._was_tracked[(self._sequence, gt_ids[gi])] = True

        self.num_matches += len(matches)
        self.num_misses += len(gt_ids) - len(matches)
        self.num_fp += len(hyp_ids) - len(matches)

    # ── IDF1 ─────────────────────────────────────────────────────────────────

    def _idf1(self) -> tuple[float, float, float]:
        """Global one-to-one identity matching over whole sequences.

        Cost of binding ground-truth identity `g` to hypothesis identity `h` is
        the number of frames the binding gets *wrong*: `(len(g) - m) + (len(h) - m)`
        where `m` is how often they coincide. Minimising that total is the
        standard IDF1 formulation and is what makes IDF1 punish fragmentation --
        only ONE hypothesis id can ever be credited for a given object.
        """
        gt_keys = sorted(self._gt_len, key=str)
        hyp_keys = sorted(self._hyp_len, key=str)
        if not gt_keys and not hyp_keys:
            return float("nan"), float("nan"), float("nan")
        total_gt = sum(self._gt_len.values())
        total_hyp = sum(self._hyp_len.values())
        if not gt_keys or not hyp_keys:
            return 0.0, 0.0, 0.0

        gt_index = {k: i for i, k in enumerate(gt_keys)}
        hyp_index = {k: j for j, k in enumerate(hyp_keys)}

        # Square, padded cost matrix so every identity may go unmatched: the
        # padding block costs exactly the identity's own length, i.e. what is
        # lost by binding it to nothing.
        n, m = len(gt_keys), len(hyp_keys)
        size = n + m
        cost = np.zeros((size, size))
        cost[:n, m:] = 1e9
        cost[n:, :m] = 1e9
        for i, gk in enumerate(gt_keys):
            cost[i, m + i] = self._gt_len[gk]
        for j, hk in enumerate(hyp_keys):
            cost[n + j, j] = self._hyp_len[hk]
        for i, gk in enumerate(gt_keys):
            for j, hk in enumerate(hyp_keys):
                if gk[0] != hk[0]:  # different sequences never share an identity
                    cost[i, j] = 1e9
                    continue
                overlap = self._pairs.get((gk[0], gk[1], hk[1]), 0)
                cost[i, j] = (self._gt_len[gk] - overlap) + (self._hyp_len[hk] - overlap)

        rows, cols = linear_sum_assignment(cost)
        idtp = 0
        for r, c in zip(rows, cols, strict=True):
            if r < n and c < m:
                gk, hk = gt_keys[r], hyp_keys[c]
                if gk[0] == hk[0]:
                    idtp += self._pairs.get((gk[0], gk[1], hk[1]), 0)

        idfn = total_gt - idtp
        idfp = total_hyp - idtp
        idp = idtp / total_hyp if total_hyp else 0.0
        idr = idtp / total_gt if total_gt else 0.0
        denom = 2 * idtp + idfp + idfn
        idf1 = 2 * idtp / denom if denom else 0.0
        _ = gt_index, hyp_index
        return idf1, idp, idr

    def result(self) -> MOTResult:
        gt = self.num_gt
        mota = (
            1.0 - (self.num_misses + self.num_fp + self.num_switches) / gt if gt else float("nan")
        )
        motp = self._iou_sum / self.num_matches if self.num_matches else float("nan")
        idf1, idp, idr = self._idf1()
        return MOTResult(
            mota=mota,
            motp=motp,
            idf1=idf1,
            idp=idp,
            idr=idr,
            precision=self.num_matches / self.num_hyp if self.num_hyp else float("nan"),
            recall=self.num_matches / gt if gt else float("nan"),
            num_switches=self.num_switches,
            num_fp=self.num_fp,
            num_misses=self.num_misses,
            num_matches=self.num_matches,
            num_gt=gt,
            num_hyp=self.num_hyp,
            num_frames=self.num_frames,
            num_gt_ids=len(self._gt_len),
            num_hyp_ids=len(self._hyp_len),
            fragmentations=self.fragmentations,
        )
