"""Build per-detection range ground truth, and characterise the ruler itself.

Runs every LiDAR-in-box estimator in `aps.groundtruth` against the labelled 3D
boxes, per range bin, and writes a per-detection table for downstream evaluation.

The comparison is the point. Picking an estimator by argument and then measuring
a monocular method against it would hide the ruler's own error inside the
estimator's error, where it is unrecoverable. Here the ruler is scored first,
against an independent annotation, and its residual becomes a documented floor
under every range number this project reports.

    python tools/build_range_gt.py --split dev
    python tools/build_range_gt.py --split val --out artifacts/range_gt_val.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aps.groundtruth import (
    GT_INVALID,
    GT_RELAXED,
    GT_STRICT,
    estimate_range,
    gt_tier,
    project_frame,
)
from aps.kitti import load_split, load_tracklets
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame

REPO = Path(__file__).resolve().parents[1]

ESTIMATORS = ["min", "p10", "p20", "median", "shrink_p20", "cluster_p20"]
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]


def bin_index(r: np.ndarray) -> np.ndarray:
    idx = np.full(len(r), -1, dtype=int)
    for i, (lo, hi) in enumerate(BINS):
        idx[(r >= lo) & (r < hi)] = i
    return idx


def collect(split: str, max_frames: int | None) -> dict[str, np.ndarray]:
    """Walk the split, returning parallel arrays -- one row per labelled box."""
    rows: dict[str, list] = {
        k: []
        for k in ["drive", "frame", "track_id", "cls", "label_near_m", "label_center_m",
                  "x1", "y1", "x2", "y2", "occlusion", "truncation", "n_in_box", "tier",
                  "shrink_used"]
    }  # fmt: skip
    for est in ESTIMATORS:
        rows[est] = []
        rows[f"{est}_n"] = []

    for drive in load_split(split):
        by_frame = boxes_by_frame(
            load_tracklets(drive.dir / "tracklet_labels.xml"), classes=FCW_CLASSES
        )
        frames = sorted(by_frame)[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} labelled frames", flush=True)

        for fi in frames:
            frame = drive.frame(fi)
            uv, depth = project_frame(frame.points(min_forward_m=0.5), drive.calib)
            for tbox in by_frame[fi]:
                box = tbox.box_2d(drive.calib)
                if box is None:
                    continue
                near_m, center_m = tbox.ranges_m(drive.calib)
                rows["drive"].append(drive.id)
                rows["frame"].append(fi)
                rows["track_id"].append(tbox.track_id)
                rows["cls"].append(tbox.object_type)
                rows["label_near_m"].append(near_m)
                rows["label_center_m"].append(center_m)
                rows["x1"].append(box[0])
                rows["y1"].append(box[1])
                rows["x2"].append(box[2])
                rows["y2"].append(box[3])
                rows["occlusion"].append(tbox.occlusion)
                rows["truncation"].append(tbox.truncation)
                rows["tier"].append(gt_tier(tbox.occlusion, tbox.truncation))
                n_in = -1
                for est in ESTIMATORS:
                    gt = estimate_range(uv, depth, box, estimator=est)
                    rows[est].append(gt.range_m)
                    rows[f"{est}_n"].append(gt.n_points)
                    if est == "shrink_p20":
                        rows["shrink_used"].append(gt.shrink)
                    n_in = gt.n_in_box
                rows["n_in_box"].append(n_in)

    return {k: np.array(v) for k, v in rows.items()}


def _bin_cells(err: np.ndarray, ok: np.ndarray, idx: np.ndarray) -> list[str]:
    cells = []
    for i in range(len(BINS)):
        m = ok & (idx == i)
        cells.append(f"{np.abs(err[m]).mean():>9.2f}" if m.sum() else f"{'-':>9}")
    return cells


def report(data: dict[str, np.ndarray], split: str) -> None:
    truth = data["label_near_m"].astype(float)
    idx = bin_index(truth)
    tier = data["tier"]
    n_total = len(truth)

    print(f"\n{'=' * 92}\nGROUND-TRUTH RULER CHARACTERISATION -- split '{split}', N = {n_total}")
    print("LiDAR-in-box estimators scored against the labelled 3D box near face.")
    print("=" * 92)

    # ── 1. Coverage: what fraction of labelled objects can be ground-truthed at all
    print("\n[1] GROUND-TRUTH COVERAGE by range bin -- what the ruler can even reach")
    print(f"{'bin (m)':<10}{'labels':>8}{'strict':>9}{'relaxed':>9}{'invalid':>9}"
          f"{'usable':>9}{'med pts':>9}")  # fmt: skip
    for i, lab in enumerate(BIN_LABELS):
        m = idx == i
        if not m.any():
            continue
        s = (tier[m] == GT_STRICT).sum()
        r = (tier[m] == GT_RELAXED).sum()
        v = (tier[m] == GT_INVALID).sum()
        pts = np.median(data["n_in_box"][m].astype(float))
        print(f"{lab:<10}{m.sum():>8}{s:>9}{r:>9}{v:>9}{100 * (s + r) / m.sum():>8.0f}%{pts:>9.0f}")
    usable = np.isin(tier, [GT_STRICT, GT_RELAXED])
    print(f"{'ALL':<10}{n_total:>8}{(tier == GT_STRICT).sum():>9}"
          f"{(tier == GT_RELAXED).sum():>9}{(tier == GT_INVALID).sum():>9}"
          f"{100 * usable.mean():>8.0f}%")  # fmt: skip
    print("\n  'invalid' = occluded or truncated. A box around an object you cannot see")
    print("  contains the OCCLUDER's returns, so a LiDAR estimate there is a confident")
    print("  measurement of the wrong object (measured: MAE 11.9 m on fully-occluded).")
    print("  These are abstentions, not failures -- but they bound where any range")
    print("  number in this project can be verified at all.")

    # ── 2. The estimator comparison, run per tier
    for tier_name, mask in [
        (GT_STRICT, tier == GT_STRICT),
        (GT_RELAXED, np.isin(tier, [GT_STRICT, GT_RELAXED])),
        (GT_INVALID, tier == GT_INVALID),
    ]:
        if not mask.any():
            continue
        label = {
            GT_STRICT: "STRICT tier (visible, untruncated) -- use for headline numbers",
            GT_RELAXED: "STRICT+RELAXED (occlusion unlabelled allowed) -- coverage tier",
            GT_INVALID: "INVALID tier -- shown ONLY to justify excluding it",
        }[tier_name]
        print(f"\n[2] {label}   N = {mask.sum()}")
        header = f"{'estimator':<13}{'valid':>7}{'MAE':>8}{'bias':>8}{'p95|e|':>8}   " + "".join(
            f"{lab:>9}" for lab in BIN_LABELS
        )
        print(header)
        print("-" * len(header))
        for est in ESTIMATORS:
            pred = data[est].astype(float)
            ok = np.isfinite(pred) & mask
            if not ok.any():
                continue
            err = pred - truth
            print(
                f"{est:<13}{100 * ok.sum() / mask.sum():>6.0f}%{np.abs(err[ok]).mean():>8.2f}"
                f"{err[ok].mean():>+8.2f}{np.percentile(np.abs(err[ok]), 95):>8.2f}   "
                + "".join(_bin_cells(err, ok, idx))
            )

    # ── 3. The two conventions, measured rather than asserted (D-003)
    gap = data["label_center_m"].astype(float) - truth
    print(
        f"\n[3] D-003 -- centroid minus near face: mean {gap.mean():.2f} m "
        f"(p05 {np.percentile(gap, 5):.2f}, p95 {np.percentile(gap, 95):.2f} m)."
    )
    print("    Scoring a near-face estimator against centroid truth would manufacture")
    print("    exactly that as a pure bias, then blame it on the geometry.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-frames", type=int, default=None, help="per drive, for a quick pass")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"building range ground truth for split '{args.split}'")
    data = collect(args.split, args.max_frames)
    if len(data["label_near_m"]) == 0:
        print("no labelled boxes found")
        return 1
    report(data, args.split)

    out = args.out or REPO / "artifacts" / f"range_gt_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"\nwrote {out}  ({len(data['label_near_m'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
