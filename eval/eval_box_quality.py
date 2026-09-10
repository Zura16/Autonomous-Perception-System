"""Detector box geometry vs the label's own box -- per object, not per class.

    python eval/eval_box_quality.py --split val

Phase 3 attributed the monocular range bias to detector boxes being ~12% short,
measured by comparing an *implied* object height (`box_h * range / f_y`) against
the pooled class mean of 1.595 m. That comparison conflates three things:

  1. the detector's box error            <- what we wanted
  2. the pooled class's 20.1% spread     <- an individual car is not the mean
  3. selection effects                   <- detected objects are not a random
                                            sample of the class

This harness measures (1) alone, by comparing each detector box to the LABEL BOX
FOR THE SAME OBJECT IN THE SAME FRAME. It also validates the implied-height
method itself, by checking the label box against the pinhole prediction
`f_y * H_3d / range` -- if that ratio is not ~1.0, the amodal 3D annotation and
the projected 2D extent disagree and the whole approach has a confound.

See docs/decisions.md D-017 and D-018.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import Detector  # noqa: E402
from aps.kitti import load_split, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402
from aps.matching import match_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FY = 721.5377
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]
MIN_N = 20  # below this a per-bin ratio is noise wearing a decimal point


def run(split: str, detector: Detector, conf: float, iou: float, max_frames: int | None) -> dict:
    cols: dict[str, list] = {
        k: []
        for k in ["det_h", "label_h", "det_bottom", "label_bottom", "det_w", "label_w",
                  "range_m", "h3d", "occlusion", "truncation", "group", "score"]
    }  # fmt: skip

    for drive in load_split(split):
        scored = boxes_by_frame(
            load_tracklets(drive.dir / "tracklet_labels.xml"), classes=FCW_CLASSES
        )
        frames = sorted(scored)[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} labelled frames", flush=True)
        warmed = False

        for fi in frames:
            image = drive.frame(fi).image()
            if not warmed:
                detector.warmup(image)
                warmed = True
            dets = detector.detect(image).above(conf)
            gts = [b for b in scored[fi] if b.box_2d(drive.calib) is not None]
            if len(dets) == 0 or not gts:
                continue

            gt_boxes = np.array([b.box_2d(drive.calib) for b in gts]).reshape(-1, 4)
            m = match_frame(dets.boxes, dets.scores, gt_boxes, iou)

            for di in np.flatnonzero(m.tp):
                gi = int(m.det_to_gt[di])
                g, db, gb = gts[gi], dets.boxes[di], gt_boxes[gi]
                cols["det_h"].append(db[3] - db[1])
                cols["label_h"].append(gb[3] - gb[1])
                cols["det_bottom"].append(db[3])
                cols["label_bottom"].append(gb[3])
                cols["det_w"].append(db[2] - db[0])
                cols["label_w"].append(gb[2] - gb[0])
                cols["range_m"].append(g.ranges_m(drive.calib)[0])
                cols["h3d"].append(g.dims_lwh_m[2])
                cols["occlusion"].append(g.occlusion)
                cols["truncation"].append(g.truncation)
                cols["group"].append(dets.groups[di])
                cols["score"].append(dets.scores[di])

    return {k: np.array(v) for k, v in cols.items()}


def report(d: dict, split: str) -> None:
    rng, det_h, lab_h = d["range_m"], d["det_h"], d["label_h"]
    bottom_delta = d["det_bottom"] - d["label_bottom"]
    ratio = det_h / lab_h
    clean = (d["occlusion"] == 0) & (d["truncation"] == 0)

    print(f"\n{'=' * 84}")
    print(f"DETECTOR BOX QUALITY -- split '{split}', {len(rng)} matched detector/label pairs")
    print("=" * 84)

    print("\n[1] METHOD CHECK -- label box height vs the pinhole prediction f*H_3d/range")
    print("    If this is not ~1.0, comparing an implied height to a 3D annotation")
    print("    has a confound and the whole approach is unsound.")
    pred = FY * d["h3d"] / rng
    print(f"  {'bin (m)':<10}{'N':>7}{'label/pred':>13}")
    for i, lab in enumerate(BIN_LABELS):
        lo, hi = BINS[i]
        m = (rng >= lo) & (rng < hi)
        if m.sum() < MIN_N:
            continue
        print(f"  {lab:<10}{m.sum():>7}{np.median(lab_h[m] / pred[m]):>13.3f}")
    print(f"  {'ALL':<10}{len(rng):>7}{np.median(lab_h / pred):>13.3f}")

    print("\n[2] DETECTOR BOX HEIGHT vs LABEL BOX HEIGHT -- same object, same frame")
    print("    This is the detector's box error ALONE, free of class-mean spread.")
    print(f"  {'bin (m)':<10}{'N':>7}{'det h px':>10}{'label h':>9}{'ratio':>8}{'shortfall':>11}")
    for i, lab in enumerate(BIN_LABELS):
        lo, hi = BINS[i]
        m = (rng >= lo) & (rng < hi)
        if m.sum() < MIN_N:
            continue
        r = np.median(ratio[m])
        print(
            f"  {lab:<10}{m.sum():>7}{np.median(det_h[m]):>10.1f}{np.median(lab_h[m]):>9.1f}"
            f"{r:>8.3f}{100 * (r - 1):>+10.1f}%"
        )
    r_all = np.median(ratio)
    print(
        f"  {'ALL':<10}{len(rng):>7}{np.median(det_h):>10.1f}{np.median(lab_h):>9.1f}"
        f"{r_all:>8.3f}{100 * (r_all - 1):>+10.1f}%"
    )

    print("\n[3] BOX BOTTOM EDGE offset (px; negative = detector bottom sits HIGH)")
    print("    Drives the contact-point estimator directly: a bottom edge too high")
    print("    reads as an object further away.")
    print(f"  {'bin (m)':<10}{'N':>7}{'median dv':>11}{'p25':>8}{'p75':>8}{'range bias':>13}")
    for i, lab in enumerate(BIN_LABELS):
        lo, hi = BINS[i]
        m = (rng >= lo) & (rng < hi)
        if m.sum() < MIN_N:
            continue
        dv = np.median(bottom_delta[m])
        # A dv px shift at range D changes contact-point range by ~ -D*dv/(fy*h/D)
        med_r = np.median(rng[m])
        below = FY * 1.655 / med_r
        implied = med_r * (below / (below + dv) - 1.0)
        print(
            f"  {lab:<10}{m.sum():>7}{dv:>11.1f}{np.percentile(bottom_delta[m], 25):>8.1f}"
            f"{np.percentile(bottom_delta[m], 75):>8.1f}{implied:>+12.2f} m"
        )

    print("\n[4] BY OCCLUSION / GROUP (height ratio; clean = visible + untruncated)")
    print(f"  {'subset':<24}{'N':>7}{'ratio':>8}")
    for name, m in [
        ("all", np.ones(len(rng), bool)),
        ("clean (occ=0, trunc=0)", clean),
        ("occluded or truncated", ~clean),
        ("vehicle", d["group"] == "vehicle"),
        ("vru", d["group"] == "vru"),
    ]:
        if m.sum() >= MIN_N:
            print(f"  {name:<24}{m.sum():>7}{np.median(ratio[m]):>8.3f}")

    print("\n[5] CLEAN SUBSET by range -- the figure to quote")
    print(f"  {'bin (m)':<10}{'N':>7}{'ratio':>8}{'shortfall':>11}{'bottom dv':>12}")
    for i, lab in enumerate(BIN_LABELS):
        lo, hi = BINS[i]
        m = clean & (rng >= lo) & (rng < hi)
        if m.sum() < MIN_N:
            continue
        r = np.median(ratio[m])
        print(
            f"  {lab:<10}{m.sum():>7}{r:>8.3f}{100 * (r - 1):>+10.1f}%"
            f"{np.median(bottom_delta[m]):>11.1f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    detector = Detector(args.weights, device=args.device)
    print(f"box quality on split '{args.split}' with {args.weights}")
    d = run(args.split, detector, args.conf, args.iou, args.max_frames)
    if len(d.get("range_m", [])) == 0:
        print("no matched pairs")
        return 1
    report(d, args.split)

    out = args.out or REPO / "artifacts" / f"box_quality_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"\nwrote {out}  ({len(d['range_m'])} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
