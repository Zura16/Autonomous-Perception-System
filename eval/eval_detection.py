"""Phase 2 detection baseline: AP, recall vs range, and latency vs budget.

    python eval/eval_detection.py --split val
    python eval/eval_detection.py --split dev --device cpu --max-frames 50

Reports at two granularities and no finer (docs/decisions.md D-014):
class-agnostic AP as the headline, and a vehicle/VRU split. Fine-grained KITTI
classes are not recoverable from a COCO-trained model.

The row that matters most for this project is **recall vs range**, not AP. If the
detector cannot see a vehicle at 60 m, then the monocular range curve beyond 60 m
is characterising nothing, and the operating envelope is bounded by detection
rather than by geometry. AP pools that away into one number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import IGNORE_CLASSES, KITTI_TO_GROUP, Detector  # noqa: E402
from aps.kitti import load_split, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402
from aps.matching import average_precision, match_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUDGET_MS = 103.56  # measured sensor rate, 9.657 Hz (D-012)
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]
HEIGHT_BINS = [(0, 25), (25, 40), (40, 80), (80, np.inf)]
HEIGHT_LABELS = ["<25", "25-40", "40-80", "80+"]


def bin_index(values: np.ndarray, bins: list[tuple[float, float]]) -> np.ndarray:
    idx = np.full(len(values), -1, dtype=int)
    for i, (lo, hi) in enumerate(bins):
        idx[(values >= lo) & (values < hi)] = i
    return idx


def run(split: str, detector: Detector, max_frames: int | None, iou: float) -> dict:
    """Detect over the split and accumulate per-detection and per-label records."""
    det_score, det_tp, det_ignored, det_group = [], [], [], []
    gt_range, gt_height, gt_occ, gt_group, gt_found = [], [], [], [], []
    gt_match_score: list[float] = []
    matched_range, matched_box, matched_score = [], [], []
    latencies: list[float] = []
    n_frames = 0

    for drive in load_split(split):
        tracklets = load_tracklets(drive.dir / "tracklet_labels.xml")
        scored = boxes_by_frame(tracklets, classes=FCW_CLASSES)
        ignored = boxes_by_frame(tracklets, classes=frozenset(IGNORE_CLASSES))
        frames = sorted(set(scored) | set(ignored))[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} frames", flush=True)

        warmed = False
        for fi in frames:
            image = drive.frame(fi).image()
            if not warmed:
                detector.warmup(image)
                warmed = True

            dets = detector.detect(image, time_it=True)
            latencies.append(dets.latency_ms)
            n_frames += 1

            gts = [b for b in scored.get(fi, []) if b.box_2d(drive.calib) is not None]
            gt_boxes = np.array([b.box_2d(drive.calib) for b in gts]).reshape(-1, 4)
            ign_boxes = np.array(
                [
                    b.box_2d(drive.calib)
                    for b in ignored.get(fi, [])
                    if b.box_2d(drive.calib) is not None
                ]
            ).reshape(-1, 4)

            m = match_frame(dets.boxes, dets.scores, gt_boxes, iou, ign_boxes)

            det_score.append(dets.scores)
            det_tp.append(m.tp)
            det_ignored.append(m.ignored)
            det_group.append(dets.groups)

            for j, tb in enumerate(gts):
                near_m, _ = tb.ranges_m(drive.calib)
                box = gt_boxes[j]
                gt_range.append(near_m)
                gt_height.append(box[3] - box[1])
                gt_occ.append(tb.occlusion)
                gt_group.append(KITTI_TO_GROUP.get(tb.object_type, "vehicle"))
                gt_found.append(bool(m.gt_matched[j]))
                # Score of the detection that claimed this label, so recall can
                # be recomputed at any operating point without re-running the
                # detector. -inf when unmatched.
                claimed = np.flatnonzero(m.det_to_gt == j)
                gt_match_score.append(float(dets.scores[claimed[0]]) if len(claimed) else -np.inf)

            # Detector boxes paired with their label's range -- Phase 3 consumes
            # this to build the monocular estimator's training/eval set.
            for d in np.flatnonzero(m.tp):
                g = gts[m.det_to_gt[d]]
                matched_range.append(g.ranges_m(drive.calib)[0])
                matched_box.append(dets.boxes[d])
                matched_score.append(dets.scores[d])

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])  # noqa: E731
    return {
        "det_score": cat(det_score),
        "det_tp": cat(det_tp).astype(bool),
        "det_ignored": cat(det_ignored).astype(bool),
        "det_group": cat(det_group).astype(str),
        "gt_range": np.array(gt_range),
        "gt_height": np.array(gt_height),
        "gt_occ": np.array(gt_occ),
        "gt_group": np.array(gt_group, dtype=str),
        "gt_found": np.array(gt_found, dtype=bool),
        "gt_match_score": np.array(gt_match_score),
        "matched_range": np.array(matched_range),
        "matched_box": np.array(matched_box).reshape(-1, 4),
        "matched_score": np.array(matched_score),
        "latency_ms": np.array(latencies),
        "n_frames": n_frames,
    }


def report(r: dict, split: str, detector: Detector, iou: float, conf: float) -> None:
    keep = ~r["det_ignored"]
    scores, tp, groups = r["det_score"][keep], r["det_tp"][keep], r["det_group"][keep]
    n_gt = len(r["gt_range"])

    print(f"\n{'=' * 88}")
    print(f"PHASE 2 DETECTION BASELINE -- split '{split}', {r['n_frames']} frames, {n_gt} labels")
    print(f"model {detector.weights} @ {detector.imgsz}px, device {detector.device}, IoU {iou}")
    print("=" * 88)

    ap, _, _ = average_precision(scores, tp, n_gt)
    print(f"\n[1] AVERAGE PRECISION @ IoU {iou}  (class-agnostic = the headline)")
    print(f"  class-agnostic      AP = {ap:.3f}   over {n_gt} labels, {len(scores)} detections")
    for g in ("vehicle", "vru"):
        gm, gn = groups == g, int((r["gt_group"] == g).sum())
        if gn:
            gap, _, _ = average_precision(scores[gm], tp[gm], gn)
            print(f"  {g:<20}AP = {gap:.3f}   over {gn} labels")
    print(f"  ignored detections (fell on Tram/Misc/Person-sitting): {int(r['det_ignored'].sum())}")

    print(f"\n[2] RECALL vs RANGE at conf >= {conf}  <-- the row that bounds the envelope")
    found_at_conf = _recall_mask(r, conf)
    idx = bin_index(r["gt_range"], BINS)
    print(f"{'bin (m)':<10}{'labels':>8}{'found':>8}{'recall':>9}{'vehicle':>10}{'vru':>8}")
    for i, lab in enumerate(BIN_LABELS):
        m = idx == i
        if not m.any():
            continue
        rec = found_at_conf[m].mean()
        cells = []
        for g, width in (("vehicle", 10), ("vru", 8)):
            gm = m & (r["gt_group"] == g)
            # Below 10 labels a percentage is noise wearing a decimal point.
            cells.append(
                f"{found_at_conf[gm].mean() * 100:>{width - 1}.0f}%"
                if gm.sum() >= 10
                else f"{'-':>{width}}"
            )
        print(
            f"{lab:<10}{m.sum():>8}{int(found_at_conf[m].sum()):>8}"
            f"{rec * 100:>8.0f}%{cells[0]}{cells[1]}"
        )
    print(f"{'ALL':<10}{n_gt:>8}{int(found_at_conf.sum()):>8}{found_at_conf.mean() * 100:>8.0f}%")

    print(f"\n[3] RECALL vs BOX HEIGHT (px) at conf >= {conf} -- the mechanism behind [2]")
    hidx = bin_index(r["gt_height"], HEIGHT_BINS)
    print(f"{'height':<10}{'labels':>8}{'recall':>9}{'median range':>14}")
    for i, lab in enumerate(HEIGHT_LABELS):
        m = hidx == i
        if not m.any():
            continue
        print(
            f"{lab:<10}{m.sum():>8}{found_at_conf[m].mean() * 100:>8.0f}%"
            f"{np.median(r['gt_range'][m]):>13.1f} m"
        )

    print(f"\n[4] RECALL vs OCCLUSION at conf >= {conf}")
    names = {-1: "unset", 0: "visible", 1: "partly", 2: "fully"}
    print(f"{'occlusion':<12}{'labels':>8}{'recall':>9}")
    for k in (0, -1, 1, 2):
        m = r["gt_occ"] == k
        if m.sum():
            print(f"{names[k]:<12}{m.sum():>8}{found_at_conf[m].mean() * 100:>8.0f}%")

    lat = r["latency_ms"]
    lat = lat[np.isfinite(lat)]
    print(f"\n[5] LATENCY -- detection only, vs the {BUDGET_MS} ms sensor budget")
    print(
        f"  device {detector.device}, {detector.imgsz}px, batch 1, N={len(lat)} frames, warmup excluded"
    )
    for q, name in [(50, "p50"), (95, "p95"), (99, "p99")]:
        v = np.percentile(lat, q)
        print(f"  {name} = {v:7.2f} ms   {100 * v / BUDGET_MS:5.1f}% of budget")
    print(f"  max = {lat.max():7.2f} ms   {100 * lat.max() / BUDGET_MS:5.1f}% of budget")
    print("  NOTE: detection only. Track/range/KF/decide stages are not yet in this sum.")


def _recall_mask(r: dict, conf: float) -> np.ndarray:
    """Which labels are claimed by a detection scoring at least `conf`.

    Uses the per-label score of the detection that matched it, so recall can be
    reported at any operating point from a single pass. NOT `gt_found`, which
    was computed over every detection down to conf=0.05 and would quote the
    recall of an operating point nobody would ship.
    """
    return r["gt_found"] & (r["gt_match_score"] >= conf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25, help="operating point for recall rows")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz)
    print(f"detecting over split '{args.split}' with {args.weights} on {args.device}")
    r = run(args.split, detector, args.max_frames, args.iou)
    report(r, args.split, detector, args.iou, args.conf)

    out = args.out or REPO / "artifacts" / f"detections_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
