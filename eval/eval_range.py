"""Phase 3: monocular range error vs range -- the project's headline artifact.

    python eval/eval_range.py --split val
    python eval/eval_range.py --split dev --max-frames 40

Runs the real pipeline end to end: detector boxes, the Phase 1 LiDAR ruler
applied to *those* boxes (not to labels), and both monocular estimators on the
same box. Nothing is matched to a label -- which is the point of having built
ground truth that works on any box.

The camera-to-road pitch is measured per frame from LiDAR and used ONLY to
decompose the error afterwards. The estimators never see it: they use the fixed
nominal from configs/camera.yaml, exactly as a deployed camera-only system
would. That is what makes the pitch treatment a measured sensitivity curve
rather than a hypothetical sweep (docs/decisions.md D-016).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import Detector  # noqa: E402
from aps.geometry import (  # noqa: E402
    CameraModel,
    crossover_range_m,
    range_contact_point,
    range_size_prior,
)
from aps.groundplane import fit_road_plane  # noqa: E402
from aps.groundtruth import estimate_range, project_frame  # noqa: E402
from aps.kitti import load_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]
ESTIMATORS = ["contact_point", "size_prior"]


def bin_index(values: np.ndarray) -> np.ndarray:
    idx = np.full(len(values), -1, dtype=int)
    for i, (lo, hi) in enumerate(BINS):
        idx[(values >= lo) & (values < hi)] = i
    return idx


def run(
    split: str, detector: Detector, camera: CameraModel, max_frames: int | None, conf: float
) -> dict:
    rows: dict[str, list] = {
        k: []
        for k in ["drive", "frame", "gt_range", "contact_point", "size_prior", "group",
                  "score", "box_h", "box_bottom", "pitch_deg", "plane_h", "n_pts"]
    }  # fmt: skip

    for drive in load_split(split):
        frames = list(range(len(drive)))[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} frames", flush=True)
        warmed = False

        for fi in frames:
            frame = drive.frame(fi)
            if not frame.has_velodyne:
                continue  # no LiDAR scan: this frame cannot be ground-truthed
            image = frame.image()
            if not warmed:
                detector.warmup(image)
                warmed = True

            dets = detector.detect(image).above(conf)
            if len(dets) == 0:
                continue

            points = frame.points(min_forward_m=0.5)
            uv, depth = project_frame(points, drive.calib)
            plane = fit_road_plane(points, drive.calib)  # GT only -- analysis, never input

            for i in range(len(dets)):
                box, group = dets.boxes[i], dets.groups[i]
                gt = estimate_range(uv, depth, box, estimator="shrink_p20")
                if not gt.is_valid:
                    continue  # no ground truth for this box; abstain, do not guess
                rows["drive"].append(drive.id)
                rows["frame"].append(fi)
                rows["gt_range"].append(gt.range_m)
                rows["contact_point"].append(range_contact_point(box, drive.calib, camera))
                rows["size_prior"].append(range_size_prior(box, drive.calib, camera, group))
                rows["group"].append(group)
                rows["score"].append(dets.scores[i])
                rows["box_h"].append(box[3] - box[1])
                rows["box_bottom"].append(box[3])
                rows["pitch_deg"].append(plane.pitch_deg if plane.is_valid else np.nan)
                rows["plane_h"].append(plane.height_m if plane.is_valid else np.nan)
                rows["n_pts"].append(gt.n_points)

    return {k: np.array(v) for k, v in rows.items()}


def _stats(err: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    """(MAE, MAPE%, bias, p95|e|)."""
    return (
        float(np.abs(err).mean()),
        float(100 * np.mean(np.abs(err) / truth)),
        float(err.mean()),
        float(np.percentile(np.abs(err), 95)),
    )


def report(d: dict, split: str, camera: CameraModel) -> None:
    truth = d["gt_range"]
    idx = bin_index(truth)
    n = len(truth)

    print(f"\n{'=' * 94}")
    print(f"PHASE 3 -- MONOCULAR RANGE ERROR vs RANGE -- split '{split}', N = {n} detections")
    print("Reference: Phase 1 LiDAR ruler on the SAME detector box (MAE 0.25 m, p95 0.63 m)")
    print("=" * 94)

    print("\n[1] ERROR vs RANGE  <-- the headline artifact")
    for est in ESTIMATORS:
        pred = d[est]
        ok = np.isfinite(pred)
        print(f"\n  {est}   (valid on {100 * ok.mean():.0f}% of detections)")
        print(f"  {'bin (m)':<9}{'N':>6}{'MAE':>8}{'MAPE':>8}{'bias':>8}{'p95|e|':>9}")
        for i, lab in enumerate(BIN_LABELS):
            m = ok & (idx == i)
            if m.sum() < 10:
                print(f"  {lab:<9}{m.sum():>6}{'  -- too few to report --':>33}")
                continue
            mae, mape, bias, p95 = _stats(pred[m] - truth[m], truth[m])
            print(f"  {lab:<9}{m.sum():>6}{mae:>8.2f}{mape:>7.1f}%{bias:>+8.2f}{p95:>9.2f}")
        mae, mape, bias, p95 = _stats(pred[ok] - truth[ok], truth[ok])
        print(f"  {'ALL':<9}{ok.sum():>6}{mae:>8.2f}{mape:>7.1f}%{bias:>+8.2f}{p95:>9.2f}")

    print("\n[2] WHICH ESTIMATOR WINS, BY RANGE  (MAPE, lower is better)")
    print(f"  {'bin (m)':<9}{'N':>6}{'contact':>10}{'size':>9}   winner")
    for i, lab in enumerate(BIN_LABELS):
        m = (idx == i) & np.isfinite(d["contact_point"]) & np.isfinite(d["size_prior"])
        if m.sum() < 10:
            continue
        cp = 100 * np.mean(np.abs(d["contact_point"][m] - truth[m]) / truth[m])
        sp = 100 * np.mean(np.abs(d["size_prior"][m] - truth[m]) / truth[m])
        print(
            f"  {lab:<9}{m.sum():>6}{cp:>9.1f}%{sp:>8.1f}%   {'contact-point' if cp < sp else 'size-prior'}"
        )

    print("\n  Predicted crossovers from the error budget (docs/error-budget.md):")
    for grp, cv in (("vehicle", 0.201), ("vru", 0.087)):
        for label, pitch in (("typical 0.319 deg", 0.319), ("p95 0.715 deg", 0.715)):
            x = crossover_range_m(cv, pitch, camera.height_m)
            print(f"    {grp:<8} CV {100 * cv:4.1f}%  at {label:<18} -> crossover {x:5.1f} m")

    print("\n[3] PITCH SENSITIVITY -- contact-point error vs MEASURED camera-to-road pitch")
    print("    Pitch is measured per frame from LiDAR and used only to stratify;")
    print("    the estimator always assumed 0.0 deg (docs/decisions.md D-016).")
    pitch = np.abs(d["pitch_deg"])
    ok = np.isfinite(d["contact_point"]) & np.isfinite(pitch)
    edges = [0.0, 0.2, 0.4, 0.6, 10.0]
    print(f"  {'|pitch|':<12}{'N':>6}{'MAPE':>8}{'bias':>9}   predicted MAPE at 30 m")
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = ok & (pitch >= lo) & (pitch < hi)
        if m.sum() < 20:
            continue
        err = d["contact_point"][m] - truth[m]
        mape = 100 * np.mean(np.abs(err) / truth[m])
        mid = (lo + min(hi, 1.0)) / 2
        pred = 100 * np.tan(np.radians(mid)) * 30.0 / camera.height_m
        print(
            f"  {f'{lo:.1f}-{hi:.1f}':<12}{m.sum():>6}{mape:>7.1f}%{err.mean():>+9.2f}{pred:>18.1f}%"
        )

    print("\n[4] BY GROUP (MAPE)")
    print(f"  {'group':<10}{'N':>7}{'contact':>10}{'size':>9}")
    for g in ("vehicle", "vru"):
        m = (d["group"] == g) & np.isfinite(d["contact_point"]) & np.isfinite(d["size_prior"])
        if m.sum() < 10:
            continue
        cp = 100 * np.mean(np.abs(d["contact_point"][m] - truth[m]) / truth[m])
        sp = 100 * np.mean(np.abs(d["size_prior"][m] - truth[m]) / truth[m])
        print(f"  {g:<10}{m.sum():>7}{cp:>9.1f}%{sp:>8.1f}%")

    print("\n[5] CREDIBLE OPERATING ENVELOPE")
    print("    Per-bin pass/fail against a MAPE threshold, plus the best single")
    print("    estimator per bin. A contiguous-from-zero envelope is reported only")
    print("    if one exists -- quoting 'out to X m' when the near bin fails would")
    print("    describe a band the system does not actually have.")
    for thr in (10, 15, 20):
        marks, contiguous = [], None
        for i, lab in enumerate(BIN_LABELS):
            m = idx == i
            best = np.inf
            for est in ESTIMATORS:
                mm = m & np.isfinite(d[est])
                if mm.sum() >= 10:
                    best = min(best, 100 * np.mean(np.abs(d[est][mm] - truth[mm]) / truth[mm]))
            if not np.isfinite(best):
                marks.append(f"{lab}:n/a")
                continue
            ok_bin = best <= thr
            marks.append(f"{lab}:{'PASS' if ok_bin else 'fail'}")
            if ok_bin and (contiguous is not None or i == 0):
                contiguous = BINS[i][1]
            elif not ok_bin and i == 0:
                contiguous = None
        span = f"0-{contiguous:g} m" if contiguous else "none contiguous from 0"
        print(f"  MAPE <= {thr:>2}% (best estimator per bin):  {'  '.join(marks)}   -> {span}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    camera = CameraModel.from_config()
    detector = Detector(args.weights, device=args.device)
    print(f"monocular range on split '{args.split}'")
    print(
        f"  camera height {camera.height_m} m, assumed pitch {camera.pitch_deg} deg (flat ground)"
    )
    d = run(args.split, detector, camera, args.max_frames, args.conf)
    if len(d.get("gt_range", [])) == 0:
        print("no ground-truthable detections")
        return 1
    report(d, args.split, camera)

    out = args.out or REPO / "artifacts" / f"range_eval_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"\nwrote {out}  ({len(d['gt_range'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
