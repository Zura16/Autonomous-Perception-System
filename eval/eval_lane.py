"""Phase 6: ego-lane estimation scored on a labelled set, with its failure set.

    python eval/eval_lane.py
    python eval/eval_lane.py --render 8     # write the worst frames to docs/figures/lanes/

Scored on the 95 human-labelled `um_lane` frames of the KITTI road benchmark
(aps/kitti/road.py), never on a hand-picked clip (hard rule 9). Solver parameters
in configs/lanes.yaml were committed before this harness first ran (27904fe).

Two geometries run on the same frames (D-023):

  nominal  fixed level camera at 1.655 m -- what a deployed system has
  oracle   each frame's shipped camera-to-road fit -- isolates marking detection

Ground truth always uses the per-frame fit.

What this set CAN and CANNOT measure. 13 frames meet the warning rule, but 8 of
them clear it by less than 0.09 m -- they graze the threshold rather than depart
-- and only 5 have the vehicle body over a line. A true-positive rate over 5 is
reported with its confidence interval, never as a bare rate. The false-warning
rate over the 82 clear frames IS measurable (D-023).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.kitti.road import RoadFrame, RoadGeometry, load_lane_frames  # noqa: E402
from aps.lanes import LaneConfig, LaneEstimate, estimate_lane, lane_from_mask  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GEOMETRIES = ["nominal", "oracle"]
Z_BINS = [(7.0, 12.0), (12.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
FIGURES = REPO / "docs" / "figures" / "lanes"


def geometry_for(name: str, frame: RoadFrame, cfg: LaneConfig) -> RoadGeometry:
    if name == "oracle":
        return frame.gt_geometry()
    return RoadGeometry.nominal(frame.P2, cfg.camera_height_m)


def boundary_errors(est: LaneEstimate, gt: LaneEstimate) -> dict[tuple[float, float], list[float]]:
    """Lateral error of each boundary, per forward band, where BOTH cover it."""
    out: dict[tuple[float, float], list[float]] = {b: [] for b in Z_BINS}
    for side in ("left", "right"):
        e, g = getattr(est, side), getattr(gt, side)
        if e is None or g is None:
            continue
        for lo, hi in Z_BINS:
            z = np.linspace(lo, hi, 5)
            ok = np.array([e.covers(zi) and g.covers(zi) for zi in z])
            if ok.any():
                out[(lo, hi)].extend((e.x_at(z[ok]) - g.x_at(z[ok])).tolist())
    return out


def run(cfg: LaneConfig) -> dict:
    rows: dict[str, list] = {
        "stem": [],
        "gt_detected": [],
        "gt_offset": [],
        "gt_width": [],
        "gt_warn": [],
    }
    for g in GEOMETRIES:
        for k in ("detected", "offset", "width", "warn"):
            rows[f"{g}_{k}"] = []
    band_err: dict[str, dict[tuple[float, float], list[float]]] = {
        g: {b: [] for b in Z_BINS} for g in GEOMETRIES
    }

    frames = load_lane_frames()
    print(f"scoring {len(frames)} labelled frames")
    for fr in frames:
        img = fr.image()
        ego, _ = fr.lane_mask()
        gt = lane_from_mask(ego, fr.gt_geometry(), cfg)
        rows["stem"].append(fr.stem)
        rows["gt_detected"].append(gt.detected)
        rows["gt_offset"].append(gt.centre_offset_m)
        rows["gt_width"].append(gt.width_m)
        rows["gt_warn"].append(gt.departure_warning)
        for g in GEOMETRIES:
            est, _ = estimate_lane(img, geometry_for(g, fr, cfg), cfg)
            rows[f"{g}_detected"].append(est.detected)
            rows[f"{g}_offset"].append(est.centre_offset_m)
            rows[f"{g}_width"].append(est.width_m)
            rows[f"{g}_warn"].append(est.departure_warning)
            for b, errs in boundary_errors(est, gt).items():
                band_err[g][b].extend(errs)

    out = {k: np.array(v) for k, v in rows.items()}
    out["band_err"] = band_err
    return out


def _robust(e: np.ndarray) -> str:
    return (
        f"median {np.median(e):+.2f}  MAE {np.abs(e).mean():.2f}  "
        f"p90|e| {np.percentile(np.abs(e), 90):.2f}"
    )


def report(d: dict) -> None:
    n = len(d["stem"])
    gt_ok = d["gt_detected"].astype(bool)
    print(f"\n{'=' * 88}")
    print(f"PHASE 6 LANES -- KITTI road benchmark um_lane, N = {n} labelled frames")
    print("ground truth: labelled ego-lane mask projected with each frame's road fit")
    print("=" * 88)
    print(f"\n  ground truth has both boundaries in {gt_ok.sum()} / {n} frames")

    print("\n[1] DETECTION -- both ego boundaries found")
    for g in GEOMETRIES:
        det = d[f"{g}_detected"].astype(bool)
        print(f"  {g:<8} {det[gt_ok].sum():>3} / {gt_ok.sum()}  = {100 * det[gt_ok].mean():5.1f}%")

    print("\n[2] LANE-CENTRE OFFSET and WIDTH error, frames where estimate and GT both exist (m)")
    for g in GEOMETRIES:
        both = gt_ok & d[f"{g}_detected"].astype(bool)
        eo = d[f"{g}_offset"][both] - d["gt_offset"][both]
        ew = d[f"{g}_width"][both] - d["gt_width"][both]
        print(f"  {g:<8} N={both.sum():>3}  offset: {_robust(eo)}")
        print(
            f"  {'':<8} {'':>5}  width:  {_robust(ew)}  (rel width MAE {100 * np.mean(np.abs(ew) / d['gt_width'][both]):.1f}%)"
        )

    print("\n[3] BOUNDARY POSITION error by distance ahead (m, per boundary sample)")
    print(f"  {'band (m)':<10}" + "".join(f"{g:>34}" for g in GEOMETRIES))
    for b in Z_BINS:
        cells = []
        for g in GEOMETRIES:
            e = np.array(d["band_err"][g][b])
            cells.append(
                f"{'N=' + str(len(e)):>8}  MAE {np.abs(e).mean():.2f}  med {np.median(e):+.2f}"
                if len(e) >= 20
                else f"{'-- too few --':>34}"
            )
        print(f"  {f'{b[0]:g}-{b[1]:g}':<10}" + "".join(f"{c:>34}" for c in cells))

    print("\n[4] DEPARTURE WARNING -- evaluated per frame")
    gw = d["gt_warn"].astype(bool) & gt_ok
    neg = gt_ok & ~d["gt_warn"].astype(bool)
    print(f"  ground truth: {gw.sum()} warning frames, {neg.sum()} non-warning frames")
    print("  (a true-positive RATE over so few positives would be noise; counts are shown)")
    for g in GEOMETRIES:
        w = d[f"{g}_warn"].astype(bool)
        print(f"  {g:<8} false warnings {w[neg].sum():>2} / {neg.sum()} = {100 * w[neg].mean():4.1f}%   "
              f"caught {w[gw].sum()} of {gw.sum()} GT warnings")  # fmt: skip


def render_worst(d: dict, cfg: LaneConfig, k: int) -> list[str]:
    """Worst nominal frames -- misses first, then largest offset error."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    gt_ok = d["gt_detected"].astype(bool)
    det = d["nominal_detected"].astype(bool)
    err = np.abs(d["nominal_offset"] - d["gt_offset"])
    score = np.where(gt_ok & ~det, np.inf, np.where(gt_ok & det, err, -np.inf))
    order = [i for i in np.argsort(-score) if np.isfinite(score[i]) or score[i] == np.inf][:k]
    frames = {f.stem: f for f in load_lane_frames()}
    written = []
    for i in order:
        fr = frames[str(d["stem"][i])]
        img = fr.image()
        ego, _ = fr.lane_mask()
        gt_geom, nom_geom = fr.gt_geometry(), RoadGeometry.nominal(fr.P2, cfg.camera_height_m)
        gt = lane_from_mask(ego, gt_geom, cfg)
        est, _ = estimate_lane(img, nom_geom, cfg)
        for lane, geom, colour in ((gt, gt_geom, (60, 220, 60)), (est, nom_geom, (40, 40, 240))):
            for b in (lane.left, lane.right):
                if b is None:
                    continue
                z = np.linspace(b.z_min_m, b.z_max_m, 60)
                u, v = geom.road_to_pixel(b.x_at(z), z)
                pts = np.stack([u, v], axis=1)
                pts = pts[np.isfinite(pts).all(axis=1)].astype(np.int32)
                cv2.polylines(img, [pts], False, colour, 3, cv2.LINE_AA)
        label = "MISSED" if not est.detected else f"offset err {err[i]:.2f} m"
        cv2.putText(img, f"{fr.stem}  {label}   green=GT  red=estimate (nominal)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
        out = FIGURES / f"{fr.stem}.jpg"
        cv2.imwrite(str(out), cv2.resize(img, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 70])
        written.append(f"{out.relative_to(REPO)}  ({label})")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", type=int, default=0, help="write the N worst frames")
    args = ap.parse_args()
    cfg = LaneConfig.from_config()
    d = run(cfg)
    report(d)
    np.savez_compressed(
        REPO / "artifacts" / "lanes_um.npz", **{k: v for k, v in d.items() if k != "band_err"}
    )
    if args.render:
        print("\n[5] FAILURE SET written:")
        for line in render_worst(d, cfg, args.render):
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
