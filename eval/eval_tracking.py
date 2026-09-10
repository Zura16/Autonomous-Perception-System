"""Phase 4: IoU baseline vs SORT, measured -- MOTA / MOTP / IDF1 / ID-switches.

    python eval/eval_tracking.py --split val
    python eval/eval_tracking.py --split dev --max-frames 60

The detector runs **once per frame** and both trackers consume identical
detections, so the only difference measured is the association strategy. Running
each tracker over its own detector pass would let run-to-run nondeterminism
leak into the comparison.

Two caveats travel with every number here:

  * labels are KITTI **raw tracklets**, not the official tracking benchmark
    ([D-004](../docs/decisions.md));
  * the metrics are **self-implemented** and validated on hand-computed cases,
    not the official devkit ([D-021](../docs/decisions.md)).

Both mean these figures are valid for the comparison this project makes -- IoU
vs SORT on identical data -- and are NOT comparable to any published leaderboard.

Hard rule 8 requires MOTA/IDF1/ID-switches. The range-stratified switch rate is
this project's addition: Phase 5 derives closing speed from a per-object box
height history, so an ID switch splices two objects' histories together and
manufactures a closing speed out of the discontinuity. Where those switches
happen in range is therefore a safety-relevant number, not bookkeeping.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import IGNORE_CLASSES, Detector  # noqa: E402
from aps.kitti import load_split, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402
from aps.matching import ioa_matrix  # noqa: E402
from aps.motmetrics import MOTAccumulator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRACKERS = ["iou", "sort", "sort_det"]
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]


def drop_ignored(boxes: np.ndarray, ignore: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Mask of hypotheses NOT sitting on an ignore region.

    A track on a labelled Tram is a real object this evaluation does not score;
    counting it as a false positive would penalise the tracker for working.
    """
    if len(boxes) == 0 or len(ignore) == 0:
        return np.ones(len(boxes), dtype=bool)
    return ioa_matrix(boxes, ignore).max(axis=1) < thr


def run(split: str, detector: Detector, conf: float, iou_thr: float, max_age: int,
        match_iou: float, max_frames: int | None) -> dict:  # fmt: skip
    accs = {name: MOTAccumulator(iou_threshold=match_iou) for name in TRACKERS}
    gt_range: dict[tuple[int, object], list[float]] = defaultdict(list)
    n_frames = 0

    for seq_i, drive in enumerate(load_split(split)):
        tracklets = load_tracklets(drive.dir / "tracklet_labels.xml")
        scored = boxes_by_frame(tracklets, classes=FCW_CLASSES)
        ignored = boxes_by_frame(tracklets, classes=frozenset(IGNORE_CLASSES))
        frames = sorted(set(scored) | set(ignored))[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} frames", flush=True)

        if seq_i:
            for a in accs.values():
                a.new_sequence()
        trackers = {n: make_tracker(n, iou_threshold=iou_thr, max_age=max_age) for n in TRACKERS}
        warmed = False

        for fi in frames:
            image = drive.frame(fi).image()
            if not warmed:
                detector.warmup(image)
                warmed = True
            dets = detector.detect(image).above(conf)
            n_frames += 1

            gts = [b for b in scored.get(fi, []) if b.box_2d(drive.calib) is not None]
            gt_ids = [b.track_id for b in gts]
            gt_boxes = np.array([b.box_2d(drive.calib) for b in gts]).reshape(-1, 4)
            for b in gts:
                # key must match MOTAccumulator._sequence, which starts at 0
                gt_range[(seq_i, b.track_id)].append(b.ranges_m(drive.calib)[0])

            ign = np.array(
                [
                    b.box_2d(drive.calib)
                    for b in ignored.get(fi, [])
                    if b.box_2d(drive.calib) is not None
                ]
            ).reshape(-1, 4)

            for name in TRACKERS:
                tracks = trackers[name].update(dets.boxes, dets.scores)
                hyp_boxes = np.array([t.box for t in tracks]).reshape(-1, 4)
                hyp_ids = [t.track_id for t in tracks]
                keep = drop_ignored(hyp_boxes, ign)
                accs[name].update(
                    gt_ids, gt_boxes, [h for h, k in zip(hyp_ids, keep, strict=True) if k],
                    hyp_boxes[keep],
                )  # fmt: skip

    return {"accs": accs, "gt_range": gt_range, "n_frames": n_frames}


def report(
    out: dict, split: str, conf: float, iou_thr: float, max_age: int, match_iou: float
) -> None:
    accs, gt_range = out["accs"], out["gt_range"]

    print(f"\n{'=' * 92}")
    print(f"PHASE 4 TRACKING -- split '{split}', {out['n_frames']} frames")
    print(f"identical detections to both trackers (YOLOv8n, conf {conf}); "
          f"assoc IoU {iou_thr}, max_age {max_age}; metric IoU {match_iou}")  # fmt: skip
    print("labels = KITTI raw tracklets; metrics self-implemented. NOT leaderboard-comparable.")
    print("=" * 92)

    print("\n[1] HEADLINE  (hard rule 8)")
    hdr = (f"{'tracker':<9}{'MOTA':>8}{'MOTP':>7}{'IDF1':>8}{'IDsw':>7}{'Frag':>7}"
           f"{'FP':>8}{'FN':>8}{'GT':>8}{'IDs':>7}")  # fmt: skip
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name in TRACKERS:
        r = accs[name].result()
        results[name] = r
        print(
            f"{name:<9}{r.mota:>8.3f}{r.motp:>7.3f}{r.idf1:>8.3f}{r.num_switches:>7}"
            f"{r.fragmentations:>7}{r.num_fp:>8}{r.num_misses:>8}{r.num_gt:>8}{r.num_hyp_ids:>7}"
        )

    base = results["iou"]
    for name in ("sort", "sort_det"):
        r = results[name]
        print(f"\n  {name} vs IoU baseline:")
        print(f"    MOTA {r.mota - base.mota:+.3f}   IDF1 {r.idf1 - base.idf1:+.3f}   "
              f"MOTP {r.motp - base.motp:+.3f}   "
              f"ID switches {r.num_switches - base.num_switches:+d}")  # fmt: skip
    print("\n  'sort' emits the Kalman-filtered box; 'sort_det' emits the raw associated")
    print("  detection with identical association. The gap between them is the cost or")
    print("  benefit of SMOOTHING alone; sort_det vs iou is ASSOCIATION alone.")

    print("\n[2] ID SWITCHES BY RANGE -- per ground-truth object, binned by its median range")
    print("    Phase 5 reads box height per track; a switch splices two objects'")
    print("    height histories and manufactures a closing speed from the seam.")
    print(f"  {'bin (m)':<10}{'GT objs':>9}" + "".join(f"{n:>12}" for n in TRACKERS))
    medians = {k: float(np.median(v)) for k, v in gt_range.items()}
    for i, lab in enumerate(BIN_LABELS):
        lo, hi = BINS[i]
        keys = [k for k, r in medians.items() if lo <= r < hi]
        if not keys:
            continue
        cells = []
        for name in TRACKERS:
            sw = accs[name].switches_by_gt
            cells.append(f"{sum(sw.get(k, 0) for k in keys):>12}")
        print(f"  {lab:<10}{len(keys):>9}" + "".join(cells))
    total_keys = list(medians)
    print(f"  {'ALL':<10}{len(total_keys):>9}"
          + "".join(f"{sum(accs[n].switches_by_gt.values()):>12}" for n in TRACKERS))  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--assoc-iou", type=float, default=0.3, help="tracker association threshold")
    ap.add_argument("--match-iou", type=float, default=0.5, help="metric matching threshold")
    ap.add_argument("--max-age", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    detector = Detector(args.weights, device=args.device)
    print(f"tracking on split '{args.split}': {' vs '.join(TRACKERS)}")
    out = run(args.split, detector, args.conf, args.assoc_iou, args.max_age,
              args.match_iou, args.max_frames)  # fmt: skip
    if out["n_frames"] == 0:
        print("no frames")
        return 1
    report(out, args.split, args.conf, args.assoc_iou, args.max_age, args.match_iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
