"""Phase 7: forward-collision warning evaluated as a detector -- TPR and FP/hour.

    python eval/eval_fcw.py --split val --collect   # detector + tracker + TTC, once (~10 min)
    python eval/eval_fcw.py --split val             # sweep decision parameters offline

Collection runs the shipped pipeline once and stores one row per track-frame for
all three Phase 5 deadbands. The sweep then replays the decision layer
(aps/fcw.py) over those rows for every (TTC threshold, deadband, persistence)
combination, so every configuration sees identical perception.

GROUND-TRUTH THREATS come from the annotations alone
(aps.groundtruth.tracklet_kinematics): an annotated object is a threat in a frame
when it is closing, inside the same straight corridor the decision layer uses,
and its annotated TTC is below the threshold. Consecutive threat frames form an
event. Because this is computed independently of detection, threats the
detector never saw stay in the TPR denominator.

AN ALARM is a warning ONSET. It is justified when, at onset, the warned track
matches (IoU >= 0.5) an annotated object that is closing, in the corridor, and
has annotated TTC below threshold + `JUSTIFY_TOL_S`. The tolerance accepts a
warning slightly early -- an early warning is not a false one -- and its value is
stated. Every other onset is a false positive, including onsets on tracks that
match no annotated object at all.

Two caveats travel with every number:

  * EXPOSURE IS MINUTES, NOT HOURS. False positives are reported as counts with
    an exact Poisson interval; the per-hour figure is that interval scaled, and
    it is enormous because the denominator is tiny.
  * THE OPERATING POINT IS SELECTED ON VAL. Figures at the chosen point are
    optimistic by construction; the unbiased number waits for the held-out test
    drives in Phase 9 (D-024).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import Detector  # noqa: E402
from aps.fcw import (  # noqa: E402
    FcwConfig,
    FcwDecider,
    corridor_range_m,
    lateral_offset_m,
    poisson_interval,
    threat_events,
)
from aps.geometry import CameraModel, range_contact_point, range_size_prior  # noqa: E402
from aps.groundtruth import tracklet_kinematics  # noqa: E402
from aps.kitti import load_split, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402
from aps.matching import iou_matrix  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEADBANDS = ["none", "fixed", "sigma"]
MATCH_IOU = 0.5
JUSTIFY_TOL_S = 1.0
TTC_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
PERSISTENCE = [(1, 1), (2, 3), (3, 3)]


def rows_path(split: str) -> Path:
    return REPO / "artifacts" / f"fcw_rows_{split}.npz"


# ── collection ───────────────────────────────────────────────────────────────


def collect(split: str, detector: Detector, conf: float) -> None:
    camera = CameraModel.from_config()
    mcfg = MotionConfig.from_config()
    fcw = FcwConfig.from_config()
    est: dict[str, list] = defaultdict(list)
    gt: dict[str, list] = defaultdict(list)
    exposure_s = 0.0

    for d_i, drive in enumerate(load_split(split)):
        tracklets = load_tracklets(drive.dir / "tracklet_labels.xml")
        labelled = boxes_by_frame(tracklets, classes=FCW_CLASSES)
        kin = tracklet_kinematics(tracklets, drive.calib, drive.timestamps_s, FCW_CLASSES)
        t = drive.timestamps_s
        exposure_s += float(t[-1] - t[0]) + drive.mean_dt_s
        print(
            f"  {drive.id}: {len(drive)} frames, {len(kin)} annotated kinematic samples", flush=True
        )

        for (tid, f), k in kin.items():
            gt["drive"].append(d_i)
            gt["track"].append(tid)
            gt["frame"].append(f)
            gt["range"].append(k.range_m)
            gt["speed"].append(k.closing_speed_mps)
            gt["lateral"].append(k.lateral_m)

        fx, _, cx, _ = drive.calib.intrinsics(2)
        tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
        motion = {db: ScaleRateEstimator(mcfg, deadband=db) for db in DEADBANDS}
        last_t: dict[int, float] = {}
        warmed = False

        for fi in range(len(drive)):
            image = drive.frame(fi).image()
            if not warmed:
                detector.warmup(image)
                warmed = True
            dets = detector.detect(image).above(conf)
            tracks = tracker.update(dets.boxes, dets.scores)
            now = float(t[fi])

            gts = [b for b in labelled.get(fi, []) if b.box_2d(drive.calib) is not None]
            gt_boxes = np.array([b.box_2d(drive.calib) for b in gts]).reshape(-1, 4)
            group_of = {
                tuple(np.round(b, 3)): g for b, g in zip(dets.boxes, dets.groups, strict=True)
            }

            for tr in tracks:
                dt = now - last_t.get(tr.track_id, now)
                last_t[tr.track_id] = now
                box = tr.box
                h = float(box[3] - box[1])
                group = group_of.get(tuple(np.round(box, 3)), "vehicle")
                contact = range_contact_point(box, drive.calib, camera)
                prior = range_size_prior(box, drive.calib, camera, group)
                r_corr = corridor_range_m(contact, prior, fcw)
                lateral = lateral_offset_m(0.5 * (box[0] + box[2]), fx, cx, r_corr)

                match = -1
                if len(gts):
                    ious = iou_matrix(box[None, :], gt_boxes)[0]
                    j = int(np.argmax(ious))
                    if ious[j] >= MATCH_IOU:
                        match = gts[j].track_id

                est["drive"].append(d_i)
                est["frame"].append(fi)
                est["track"].append(tr.track_id)
                est["lateral"].append(lateral)
                est["gt_track"].append(match)
                for db, m in motion.items():
                    s = m.update(tr.track_id, h, dt, contact)
                    est[f"{db}_closing"].append(s.is_closing)
                    est[f"{db}_ttc"].append(s.ttc_s)

    out = {f"est_{k}": np.array(v) for k, v in est.items()}
    out.update({f"gt_{k}": np.array(v) for k, v in gt.items()})
    out["exposure_s"] = np.array(exposure_s)
    out["n_frames"] = np.array(len(set(zip(est["drive"], est["frame"], strict=True))))
    np.savez_compressed(rows_path(split), **out)
    print(f"\nwrote {rows_path(split)}  exposure {exposure_s:.1f} s")


# ── offline decision sweep ───────────────────────────────────────────────────


def evaluate(r: dict, ttc_thr: float, deadband: str, k: int, n: int, corridor: float) -> dict:
    fcw = FcwConfig(
        ttc_warning_s=ttc_thr,
        ttc_aeb_s=min(1.0, ttc_thr),
        corridor_half_width_m=corridor,
        persistence_k=k,
        persistence_n=n,
        fallback_range_m=6.0,
    )

    # Ground-truth threat events, from annotations alone.
    g_ttc = np.where(r["gt_speed"] < 0, -r["gt_range"] / np.minimum(r["gt_speed"], -1e-9), np.inf)
    g_inpath = np.abs(r["gt_lateral"]) < corridor
    g_threat = g_inpath & (g_ttc < ttc_thr)
    lookup = {
        (int(dr), int(tr), int(fr)): (tt, ip)
        for dr, tr, fr, tt, ip in zip(
            r["gt_drive"], r["gt_track"], r["gt_frame"], g_ttc, g_inpath, strict=True
        )
    }
    events: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for dr, tr in set(zip(r["gt_drive"].tolist(), r["gt_track"].tolist(), strict=True)):
        m = (r["gt_drive"] == dr) & (r["gt_track"] == tr)
        ev = threat_events(r["gt_frame"][m], g_threat[m])
        if ev:
            events[(dr, tr)] = ev

    # Replay the decision layer, in frame order per drive.
    order = np.lexsort((r["est_frame"], r["est_drive"]))
    decider_by_drive: dict[int, FcwDecider] = {}
    onsets = []
    for i in order:
        dr = int(r["est_drive"][i])
        dec = decider_by_drive.setdefault(dr, FcwDecider(fcw))
        out = dec.update(
            int(r["est_track"][i]),
            bool(r[f"est_{deadband}_closing"][i]),
            float(r[f"est_{deadband}_ttc"][i]),
            float(r["est_lateral"][i]),
        )
        if out.warning_onset:
            onsets.append((dr, int(r["est_frame"][i]), int(r["est_gt_track"][i])))

    tp_onsets, fp = [], 0
    for dr, fr, gtr in onsets:
        truth = lookup.get((dr, gtr, fr)) if gtr >= 0 else None
        if truth is not None and truth[1] and truth[0] < ttc_thr + JUSTIFY_TOL_S:
            tp_onsets.append((dr, gtr, fr, truth[0]))
        else:
            fp += 1

    early_frames = int(np.ceil(JUSTIFY_TOL_S / 0.10356))
    detected, lead = 0, []
    n_events = sum(len(v) for v in events.values())
    for (dr, tr), evs in events.items():
        mine = sorted((fr, tt) for d2, t2, fr, tt in tp_onsets if d2 == dr and t2 == tr)
        for start, end in evs:
            hits = [(fr, tt) for fr, tt in mine if start - early_frames <= fr <= end]
            if hits:
                detected += 1
                lead.append(hits[0][1])

    hours = float(r["exposure_s"]) / 3600.0
    lo, hi = poisson_interval(fp)
    return dict(
        events=n_events,
        detected=detected,
        tpr=detected / n_events if n_events else float("nan"),
        fp=fp,
        fp_per_h=fp / hours,
        fp_lo=lo / hours,
        fp_hi=hi / hours,
        onsets=len(onsets),
        lead_median=float(np.median(lead)) if lead else float("nan"),
    )


def sweep(split: str, corridor: float) -> None:
    raw = np.load(rows_path(split), allow_pickle=True)
    r = {k: raw[k] for k in raw.files}
    exposure = float(r["exposure_s"])
    print(f"\n{'=' * 104}")
    print(f"PHASE 7 FCW -- split '{split}', {int(r['n_frames'])} frames, "
          f"exposure {exposure:.1f} s = {exposure / 60:.2f} min = {exposure / 3600:.4f} h")  # fmt: skip
    print(
        f"corridor +/-{corridor} m (straight); alarm justified if GT TTC < threshold + {JUSTIFY_TOL_S} s"
    )
    print("OPERATING POINT SELECTED ON VAL -- figures optimistic; unbiased number is Phase 9 test.")
    print("=" * 104)
    hdr = (f"{'TTC thr':>7} {'deadband':<8}{'k/n':>5}{'events':>8}{'caught':>8}{'TPR':>7}"
           f"{'onsets':>8}{'FP':>5}{'FP/h':>8}{'FP/h 95% CI':>18}{'lead TTC':>10}")  # fmt: skip
    print(hdr)
    print("-" * len(hdr))
    for thr, db, (k, n) in product(TTC_THRESHOLDS, DEADBANDS, PERSISTENCE):
        m = evaluate(r, thr, db, k, n, corridor)
        persistence = f"{k}/{n}"
        interval = f"{m['fp_lo']:.0f}-{m['fp_hi']:.0f}"
        print(
            f"{thr:>7.1f} {db:<8}{persistence:>5}{m['events']:>8}{m['detected']:>8}"
            f"{100 * m['tpr']:>6.0f}%{m['onsets']:>8}{m['fp']:>5}{m['fp_per_h']:>8.0f}"
            f"{interval:>18}{m['lead_median']:>9.2f}s"
        )
        if (k, n) == PERSISTENCE[-1] and db == DEADBANDS[-1]:
            print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--corridor", type=float, default=FcwConfig.from_config().corridor_half_width_m)
    args = ap.parse_args()
    if args.collect or not rows_path(args.split).exists():
        collect(args.split, Detector(args.weights, device=args.device), args.conf)
    sweep(args.split, args.corridor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
