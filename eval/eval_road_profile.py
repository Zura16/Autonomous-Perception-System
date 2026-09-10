"""Does the real road deviate from the assumed flat plane, and by how much?

    python eval/eval_road_profile.py --split val

The contact-point estimator assumes the ground sits at a fixed `h_cam` below the
camera and is level. This measures the road's ACTUAL height against that
assumption, from LiDAR, as a function of depth -- and turns it into a predicted
range bias with **no free parameters**:

    a contact point at depth D with real ground height y_act projects to
        v = cy + f_y * y_act / D
    the estimator, assuming y = h_cam, returns
        D_est = f_y * h_cam / (v - cy) = D * h_cam / y_act
    so the fractional bias is
        h_cam / y_act - 1

Nothing is fitted to the observed error; the prediction comes entirely from the
measured road geometry, which is what makes the comparison a test rather than a
curve fit. See docs/decisions.md D-020.

LiDAR is ground truth here and never an estimator input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.geometry import CameraModel  # noqa: E402
from aps.kitti import load_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Depth bins, finer than the standard five because the road profile is smooth
# and the interesting structure is where the bias changes sign.
Z_EDGES = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65]

# Plausible-road gate, in the rectified camera frame (y down, z forward).
# The height band is wide enough to admit a real slope and narrow enough to
# exclude car bodies above and any sub-surface artefacts below; the lateral
# limit covers where detected objects actually sit.
ROAD_BAND_ABOVE_M = 0.6  # metres above the assumed plane
ROAD_BAND_BELOW_M = 1.2  # metres below it
ROAD_LATERAL_M = 6.0
MIN_POINTS_PER_BIN = 30
MIN_FRAMES_PER_BIN = 20


def measure(split: str, camera: CameraModel, stride: int) -> tuple[dict, int]:
    """Median ground height per depth bin, aggregated over frames."""
    per_bin: dict[int, list[float]] = {i: [] for i in range(len(Z_EDGES) - 1)}
    n_frames = 0

    for drive in load_split(split):
        print(f"  {drive.id}: every {stride}th of {len(drive)} frames", flush=True)
        for fi in range(0, len(drive), stride):
            rect = drive.calib.velo_to_rect0(drive.frame(fi).points(min_forward_m=0.5))
            x, y, z = rect[:, 0], rect[:, 1], rect[:, 2]
            road = (
                (y > camera.height_m - ROAD_BAND_ABOVE_M)
                & (y < camera.height_m + ROAD_BAND_BELOW_M)
                & (np.abs(x) < ROAD_LATERAL_M)
                & (z > 4)
            )
            yr, zr = y[road], z[road]
            for i, (lo, hi) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:], strict=False)):
                m = (zr >= lo) & (zr < hi)
                if m.sum() >= MIN_POINTS_PER_BIN:
                    per_bin[i].append(float(np.median(yr[m])))
            n_frames += 1

    return per_bin, n_frames


def report(per_bin: dict, n_frames: int, camera: CameraModel, split: str) -> dict:
    h = camera.height_m
    print(f"\n{'=' * 88}")
    print(f"ROAD PROFILE vs THE ASSUMED FLAT PLANE -- split '{split}', {n_frames} frames")
    print(f"assumed ground: level, {h} m below the camera")
    print("=" * 88)

    print("\n[1] MEASURED ROAD HEIGHT (rectified frame, y down; larger y = further below)")
    print(
        f"  {'depth (m)':<11}{'frames':>8}{'median y':>11}{'vs assumed':>12}{'predicted bias':>18}"
    )
    predicted: dict[tuple[int, int], float] = {}
    for i, (lo, hi) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:], strict=False)):
        v = np.array(per_bin[i])
        if len(v) < MIN_FRAMES_PER_BIN:
            continue
        y_med = float(np.median(v))
        frac = h / y_med - 1.0
        predicted[(lo, hi)] = frac
        mid = (lo + hi) / 2
        print(
            f"  {f'{lo}-{hi}':<11}{len(v):>8}{y_med:>11.3f}{y_med - h:>+12.3f}"
            f"{frac * 100:>+12.1f}% = {mid * frac:+.2f} m"
        )

    print("\n  The road falls steadily away from the assumed plane with depth.")
    print("  A contact point on ground LOWER than assumed projects lower in the")
    print("  image, and the estimator reads the object as CLOSER than it is.")
    return predicted


def compare(predicted: dict, split: str) -> None:
    """Predicted bias against the observed contact-point bias, if available."""
    path = REPO / "artifacts" / f"range_eval_{split}.npz"
    if not path.exists():
        print(f"\n[2] skipped -- no {path.name}; run eval/eval_range.py first")
        return

    d = np.load(path, allow_pickle=True)
    truth, cp, group = d["gt_range"], d["contact_point"], d["group"]
    ok = np.isfinite(cp) & (group == "vehicle")

    print("\n[2] PREDICTION vs OBSERVATION -- vehicles, contact-point estimator")
    print("    Nothing here is fitted. The prediction comes from road geometry alone.")
    print(f"  {'depth (m)':<11}{'N':>6}{'predicted':>13}{'observed':>13}{'residual':>12}")
    for (lo, hi), frac in predicted.items():
        m = ok & (truth >= lo) & (truth < hi)
        if m.sum() < 40:
            continue
        pred = float(np.median(truth[m])) * frac
        obs = float(np.median(cp[m] - truth[m]))
        print(
            f"  {f'{lo}-{hi}':<11}{m.sum():>6}{pred:>+12.2f} m{obs:>+12.2f} m{obs - pred:>+11.2f}"
        )

    print("\n  Small residuals beyond ~20 m mean road non-flatness accounts for the")
    print("  far-range bias. A persistent positive residual in the near field means")
    print("  something else is acting there -- see docs/decisions.md D-020.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--stride", type=int, default=6, help="sample every Nth frame")
    args = ap.parse_args()

    camera = CameraModel.from_config()
    print(f"measuring road profile on split '{args.split}'")
    per_bin, n_frames = measure(args.split, camera, args.stride)
    if n_frames == 0:
        print("no frames")
        return 1
    predicted = report(per_bin, n_frames, camera, args.split)
    compare(predicted, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
