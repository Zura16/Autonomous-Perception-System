"""Phase 5: closing speed and TTC on real data, and the phantom-closing rate.

    python eval/eval_ttc.py --split val
    python eval/eval_ttc.py --split dev --max-frames 80

Pipeline, exactly as shipped: YOLOv8n -> `sort_det` tracker (D-021) -> per-track
scale-rate filter over u = 1/h (aps/motion.py) -> TTC, and closing speed scaled by
the contact-point range (Phase 3). The detector runs once per frame and every
deadband variant consumes the same tracks, so only the deadband differs.

GROUND TRUTH. Each track box is matched to a labelled tracklet at IoU >= 0.5.
Ground-truth range is that tracklet's near-face range; ground-truth closing speed
is the least-squares slope of that range over a +/-3-frame window against the
sensor timestamps. Differencing is acceptable HERE and only here: tracklet poses
are smooth human annotations of a 3D box, not a noisy per-frame depth estimate,
which is precisely the distinction hard rule 3 draws. The window and its minimum
support are stated so the reference is reproducible.

Three questions, in the order hard rules 3, 4 and the reference table ask them:

  1. how wrong is the closing speed and the TTC, where it matters (TTC < 3 s)?
  2. how often does a relatively STATIONARY object report closing (phantom)?
  3. how often is a genuinely closing object flagged at all?
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import Detections, Detector  # noqa: E402
from aps.geometry import CameraModel, range_contact_point  # noqa: E402
from aps.groundtruth import KINEMATICS_WINDOW, tracklet_kinematics  # noqa: E402
from aps.kitti import Calibration, Tracklet, load_split, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402
from aps.matching import iou_matrix  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEADBANDS = ["none", "fixed", "sigma"]
MATCH_IOU = 0.5
STATIONARY_MPS = 0.5  # |GT closing speed| below this counts as relatively stationary
CLOSING_MPS = -2.0  # GT closing speed below this counts as genuinely closing
BINS = [(0, 10), (10, 20), (20, 30), (30, 50)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50"]


def gt_kinematics(tracklets: list[Tracklet], calib: Calibration, times_s: np.ndarray) -> dict:
    """(track_id, frame) -> (range_m, closing_speed_mps). Thin adapter over the
    shared definition in aps.groundtruth.tracklet_kinematics."""
    kin = tracklet_kinematics(tracklets, calib, times_s, FCW_CLASSES)
    return {k: (v.range_m, v.closing_speed_mps) for k, v in kin.items()}


def run(split: str, detector: Detector, max_frames: int | None, conf: float) -> dict:
    camera = CameraModel.from_config()
    mcfg = MotionConfig.from_config()
    rows: dict[str, list] = defaultdict(list)

    for drive in load_split(split):
        tracklets = load_tracklets(drive.dir / "tracklet_labels.xml")
        labelled = boxes_by_frame(tracklets, classes=FCW_CLASSES)
        gt = gt_kinematics(tracklets, drive.calib, drive.timestamps_s)
        frames = list(range(len(drive)))[: max_frames or None]
        print(f"  {drive.id}: {len(frames)} frames, {len(gt)} GT kinematic samples", flush=True)

        tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
        estimators = {db: ScaleRateEstimator(mcfg, deadband=db) for db in DEADBANDS}
        last_t: dict[int, float] = {}
        warmed = False

        for fi in frames:
            image = drive.frame(fi).image()
            if not warmed:
                detector.warmup(image)
                warmed = True
            now = float(drive.timestamps_s[fi])
            tracks = tracker.update(*_boxes_scores(detector.detect(image).above(conf)))

            gts = [b for b in labelled.get(fi, []) if b.box_2d(drive.calib) is not None]
            gt_boxes = np.array([b.box_2d(drive.calib) for b in gts]).reshape(-1, 4)

            for tr in tracks:
                dt = now - last_t.get(tr.track_id, now)
                last_t[tr.track_id] = now
                h = float(tr.box[3] - tr.box[1])
                rng_est = range_contact_point(tr.box, drive.calib, camera)
                states = {
                    db: est.update(tr.track_id, h, dt, rng_est) for db, est in estimators.items()
                }

                # Match this track box to a labelled object for ground truth.
                gt_range = gt_speed = np.nan
                if len(gts):
                    ious = iou_matrix(tr.box[None, :], gt_boxes)[0]
                    j = int(np.argmax(ious))
                    if ious[j] >= MATCH_IOU and (gts[j].track_id, fi) in gt:
                        gt_range, gt_speed = gt[(gts[j].track_id, fi)]

                rows["drive"].append(drive.id)
                rows["frame"].append(fi)
                rows["track"].append(tr.track_id)
                rows["gt_range"].append(gt_range)
                rows["gt_speed"].append(gt_speed)
                rows["est_range"].append(rng_est)
                for db, s in states.items():
                    rows[f"{db}_closing"].append(s.is_closing)
                    rows[f"{db}_ttc"].append(s.ttc_s)
                    rows[f"{db}_speed"].append(s.closing_speed_mps)
                    rows[f"{db}_n"].append(s.n_updates)

    return {k: np.array(v) for k, v in rows.items()}


def _boxes_scores(dets: Detections) -> tuple[np.ndarray, np.ndarray]:
    return dets.boxes, dets.scores


def report(d: dict, split: str) -> None:
    gt_r, gt_v = d["gt_range"], d["gt_speed"]
    has_gt = np.isfinite(gt_r) & np.isfinite(gt_v)
    gt_ttc = np.where(gt_v < 0, -gt_r / np.where(gt_v < 0, gt_v, -1), np.inf)
    mature = d["none_n"] >= 3

    print(f"\n{'=' * 90}")
    print(f"PHASE 5 -- CLOSING SPEED + TTC -- split '{split}', {len(gt_r)} track-frames, "
          f"{int(has_gt.sum())} with ground truth")  # fmt: skip
    print(
        f"GT closing speed = LSQ slope of annotated near-face range over +/-{KINEMATICS_WINDOW} frames"
    )
    print("=" * 90)

    stationary = has_gt & mature & (np.abs(gt_v) < STATIONARY_MPS)
    closing = has_gt & mature & (gt_v < CLOSING_MPS)
    print("\n[1] GROUND-TRUTH POPULATION (filter has >= 3 updates)")
    print(f"  relatively stationary |v| < {STATIONARY_MPS} m/s : {int(stationary.sum())}")
    print(f"  genuinely closing      v < {CLOSING_MPS} m/s   : {int(closing.sum())}")
    print(
        f"  GT TTC < 3 s                                : {int((has_gt & mature & (gt_ttc < 3)).sum())}"
    )

    print("\n[2] PHANTOM vs DETECTION -- hard rule 4, on real tracks")
    print(f"  {'deadband':<10}{'phantom%':>10}{'detect%':>10}")
    for db in DEADBANDS:
        flag = d[f"{db}_closing"].astype(bool)
        ph = 100 * flag[stationary].mean() if stationary.any() else np.nan
        de = 100 * flag[closing].mean() if closing.any() else np.nan
        print(f"  {db:<10}{ph:>9.1f}%{de:>9.1f}%")

    print("\n[3] CLOSING-SPEED ERROR where both GT and estimate say closing (m/s)")
    print("    Closing speed = -D_est/TTC, so it INHERITS the Phase 3 range error.")
    print(f"  {'deadband':<10}{'bin (m)':<9}{'N':>6}{'median err':>12}{'MAE':>8}{'rel MAE':>9}")
    for db in DEADBANDS:
        est_v = d[f"{db}_speed"]
        both = closing & d[f"{db}_closing"].astype(bool) & np.isfinite(est_v)
        for (lo, hi), lab in zip(BINS, BIN_LABELS, strict=True):
            m = both & (gt_r >= lo) & (gt_r < hi)
            if m.sum() < 15:
                continue
            e = est_v[m] - gt_v[m]
            print(f"  {db:<10}{lab:<9}{m.sum():>6}{np.median(e):>+12.2f}{np.abs(e).mean():>8.2f}"
                  f"{100 * np.mean(np.abs(e) / np.abs(gt_v[m])):>8.0f}%")  # fmt: skip

    print("\n[4] TTC ERROR where it matters -- GT TTC below each horizon")
    print("    TTC is calibration-free (f and H cancel), so it does NOT inherit range error.")
    print(
        f"  {'deadband':<10}{'horizon':<9}{'GT N':>6}{'flagged':>9}{'median err s':>14}{'MAE s':>8}{'rel MAE':>9}"
    )
    for db in DEADBANDS:
        est_t = d[f"{db}_ttc"]
        flag = d[f"{db}_closing"].astype(bool)
        for horizon in (3.0, 6.0, 10.0):
            pop = has_gt & mature & (gt_ttc < horizon)
            if pop.sum() < 10:
                print(f"  {db:<10}{f'<{horizon:g}s':<9}{int(pop.sum()):>6}   -- too few --")
                continue
            m = pop & flag & np.isfinite(est_t)
            if m.sum() < 5:
                print(
                    f"  {db:<10}{f'<{horizon:g}s':<9}{int(pop.sum()):>6}{int(m.sum()):>9}   -- too few flagged --"
                )
                continue
            e = est_t[m] - gt_ttc[m]
            print(f"  {db:<10}{f'<{horizon:g}s':<9}{int(pop.sum()):>6}{int(m.sum()):>9}"
                  f"{np.median(e):>+14.2f}{np.abs(e).mean():>8.2f}"
                  f"{100 * np.mean(np.abs(e) / gt_ttc[m]):>8.0f}%")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    detector = Detector(args.weights, device=args.device)
    print(f"closing speed + TTC on split '{args.split}'")
    d = run(args.split, detector, args.max_frames, args.conf)
    if len(d.get("gt_range", [])) == 0:
        print("no tracks")
        return 1
    report(d, args.split)
    out = REPO / "artifacts" / f"ttc_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
