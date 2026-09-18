"""Per-stage and end-to-end latency for the full perception pipeline.

    python bench/run_bench.py --drive 2011_09_26_drive_0059 --frames 300
    python bench/run_bench.py --frames 300 --no-lanes

Hard rule 10: every latency number carries hardware, resolution, batch size,
N frames, warmup excluded, and p50/p95/p99, reported against the measured
**103.56 ms/frame** sensor budget (9.657 Hz, D-006). "Real-time" with no
denominator is marketing.

Three things this harness does that a naive timing loop gets wrong:

  **The total is measured, not summed.** Percentiles are not additive: the sum
  of the stages' p99s is not the p99 of the total, because the stages do not
  peak on the same frames. A stack whose stage-p99s sum to 90 ms can still blow
  a 103 ms budget, and one that appears to blow it can be fine. Total latency is
  therefore timed per frame as its own series, and the summed-p99 figure is
  printed only to show how far it misleads.

  **Async work is synchronized before the clock stops.** MPS and CUDA dispatch
  is asynchronous; timing a launch measures the launch. Every stage that can
  touch the accelerator is followed by a device sync inside the timed region.

  **Image decode is reported but excluded from the budget.** A deployed system
  receives frames from a sensor, not a PNG decoder. Counting `cv2.imread` as
  perception cost would flatter nothing and mislead: it is real work this
  offline harness does that the modelled system would not.

The pipeline timed here is the one `app/replay.py` runs, minus drawing. The HUD
is a demo, not perception, and its cost is not charged to the budget.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aps.detect import Detector, _make_sync  # noqa: E402
from aps.fcw import FcwConfig, FcwDecider, corridor_range_m, lateral_offset_m  # noqa: E402
from aps.geometry import CameraModel, range_contact_point, range_size_prior  # noqa: E402
from aps.kitti import load_split  # noqa: E402
from aps.kitti.dataset import RawDrive  # noqa: E402
from aps.kitti.road import RoadGeometry  # noqa: E402
from aps.lanes import BevGrid, LaneConfig, estimate_lane  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUDGET_MS = 103.56  # measured sensor period, 9.657 Hz (D-006)

STAGES = ["detect", "track", "range", "motion+ttc", "fcw", "lanes"]


class Stopwatch:
    """Accumulates per-stage samples, synchronizing the device before each stop."""

    def __init__(self, sync: Callable[[], None]) -> None:
        self.sync = sync
        self.samples: dict[str, list[float]] = {}
        self._t0 = 0.0
        self._name = ""

    def start(self, name: str) -> Stopwatch:
        self._name = name
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        self.sync()
        dt = (time.perf_counter() - self._t0) * 1000.0
        self.samples.setdefault(self._name, []).append(dt)
        return dt

    def __enter__(self) -> Stopwatch:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


class Pipeline:
    """The perception stack under test, assembled once and timed per frame.

    A struct rather than a long argument list: the benchmark needs every stage's
    state to persist ACROSS drives -- the tracker, the scale-rate filter and the
    FCW persistence counters all carry history, and rebuilding them per drive
    would time a permanently cold pipeline.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.camera = CameraModel.from_config()
        self.lane_cfg = LaneConfig.from_config()
        self.fcw_cfg = FcwConfig.from_config()
        self.detector = Detector(args.weights, device=args.device)
        self.tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
        self.motion = ScaleRateEstimator(MotionConfig.from_config(), deadband="fixed")
        self.decider = FcwDecider(self.fcw_cfg)
        self.grid = BevGrid.from_config(self.lane_cfg)
        self.sync = _make_sync(args.device)
        self.sw = Stopwatch(self.sync)
        self.decode: list[float] = []
        self.total: list[float] = []
        self._last_t: dict[int, float] = {}

    def frame(
        self, drive: RawDrive, fi: int, lane_geom: RoadGeometry, fx: float, cx: float
    ) -> None:
        """Time one frame through every stage."""
        args = self.args
        t_decode = time.perf_counter()
        image = drive.frame(fi).image()
        self.decode.append((time.perf_counter() - t_decode) * 1000.0)

        # The budget clock starts once a frame is in hand, as it would be on a
        # sensor feed, and runs to the last decision of the frame.
        t_total = time.perf_counter()
        sw = self.sw

        sw.start("detect")
        dets = self.detector.detect(image).above(args.conf)
        sw.stop()

        sw.start("track")
        tracks = self.tracker.update(dets.boxes, dets.scores)
        sw.stop()

        now = float(drive.timestamps_s[fi])
        group_of = {tuple(np.round(b, 3)): g for b, g in zip(dets.boxes, dets.groups, strict=True)}

        sw.start("range")
        per_track = []
        for tr in tracks:
            group = group_of.get(tuple(np.round(tr.box, 3)), "vehicle")
            contact = range_contact_point(tr.box, drive.calib, self.camera)
            prior = range_size_prior(tr.box, drive.calib, self.camera, group)
            per_track.append((tr, contact, prior))
        sw.stop()

        sw.start("motion+ttc")
        states = []
        for tr, contact, _prior in per_track:
            dt = now - self._last_t.get(tr.track_id, now)
            self._last_t[tr.track_id] = now
            states.append(
                self.motion.update(tr.track_id, float(tr.box[3] - tr.box[1]), dt, contact)
            )
        sw.stop()

        sw.start("fcw")
        for (tr, contact, prior), state in zip(per_track, states, strict=True):
            lateral = lateral_offset_m(
                0.5 * (tr.box[0] + tr.box[2]),
                fx,
                cx,
                corridor_range_m(contact, prior, self.fcw_cfg),
            )
            self.decider.update(tr.track_id, state.is_closing, state.ttc_s, lateral)
        sw.stop()

        if args.lanes:
            sw.start("lanes")
            estimate_lane(image, lane_geom, self.lane_cfg, self.grid)
            sw.stop()

        self.sync()
        self.total.append((time.perf_counter() - t_total) * 1000.0)


def run(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Time the pipeline, returning per-stage millisecond series.

    Also records which drives were measured on `args`, so `report` states the
    real context instead of echoing the unused `--drive` default. A latency row
    naming the wrong sequence violates hard rule 10 as surely as a missing p99.
    """
    drives = load_split(args.split) if args.split else []
    if not drives:
        one = next((d for d in load_split(["dev", "val"]) if d.id == args.drive), None)
        if one is None:
            raise SystemExit(f"unknown drive {args.drive}")
        drives = [one]

    args.measured_drives = [d.id for d in drives]
    pipe = Pipeline(args)
    pipe.detector.warmup(drives[0].frame(0).image())

    for drive in drives:
        lane_geom = RoadGeometry.nominal(drive.calib.P_rect[2], pipe.lane_cfg.camera_height_m)
        fx, _, cx, _ = drive.calib.intrinsics(2)
        n = len(drive) if args.split else min(args.frames + args.warmup, len(drive))
        print(f"  {drive.id}: {n} frames", flush=True)
        for fi in range(n):
            pipe.frame(drive, fi, lane_geom, fx, cx)

    # Warmup is dropped here rather than skipped in the loop: every series gets
    # exactly one sample per frame, so one slice keeps them aligned. Branching
    # inside the loop had to undo a recording it had already made, and got that
    # wrong for the first frame of every stage.
    w = args.warmup
    out = {k: np.array(v[w:]) for k, v in pipe.sw.samples.items()}
    out["TOTAL (measured)"] = np.array(pipe.total[w:])
    out["image decode (not charged)"] = np.array(pipe.decode[w:])
    return out


def report(data: dict[str, np.ndarray], args: argparse.Namespace, n: int) -> None:
    width = 92
    print(f"\n{'=' * width}")
    print("END-TO-END LATENCY vs THE SENSOR BUDGET")
    print("=" * width)
    print(f"  hardware      {platform.machine()} / {platform.system()} {platform.release()}, "
          f"device={args.device}")  # fmt: skip
    print(f"  model         {args.weights}, 640 px inference, batch 1, conf {args.conf}")
    names = getattr(args, "measured_drives", [args.drive])
    shown = ", ".join(names) if len(names) <= 3 else f"{len(names)} drives: " + ", ".join(names)
    print(f"  input         1242x375, {shown}")
    print(f"  N             {n} frames measured, {args.warmup} warmup frames excluded")
    print(f"  budget        {BUDGET_MS} ms/frame (9.657 Hz measured sensor rate, D-006)")
    print(f"  lane solver   {'included' if args.lanes else 'EXCLUDED (--no-lanes)'}")

    order = [s for s in STAGES if s in data] + ["TOTAL (measured)", "image decode (not charged)"]
    print(
        f"\n  {'stage':<28}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>9}{'p99 % budget':>14}"
    )
    print("  " + "-" * (width - 2))
    for name in order:
        st = percentiles(data[name])
        share = 100 * st["p99"] / BUDGET_MS
        mark = "" if name.startswith("image decode") else f"{share:>13.1f}%"
        print(f"  {name:<28}{st['mean']:>8.2f}{st['p50']:>8.2f}{st['p95']:>8.2f}"
              f"{st['p99']:>8.2f}{st['max']:>9.2f}{mark:>14}")  # fmt: skip

    total = data["TOTAL (measured)"]
    over = int((total > BUDGET_MS).sum())
    tp = percentiles(total)

    print(f"\n  [1] FRAMES OVER BUDGET: {over} of {len(total)} ({100 * over / len(total):.1f}%), "
          f"worst {tp['max']:.1f} ms")  # fmt: skip
    if over:
        print(f"      A p99 inside budget with {over} frames over it is the Phase 2 finding")
        print("      repeating: the tail is where a real-time claim actually fails.")

    summed = sum(percentiles(data[s])["p99"] for s in STAGES if s in data)
    print("\n  [2] WHY THE TOTAL IS MEASURED, NOT SUMMED")
    print(
        f"      sum of per-stage p99 = {summed:.2f} ms   vs   measured total p99 = "
        f"{tp['p99']:.2f} ms   ({100 * (summed / tp['p99'] - 1):+.1f}%)"
    )
    print("      Percentiles do not add: stages peak on different frames. Quoting the")
    print("      sum would report a pipeline that does not exist.")

    verdict = "FITS" if tp["p99"] <= BUDGET_MS else "MISSES"
    print(
        f"\n  [3] VERDICT: p99 {tp['p99']:.2f} ms {verdict} the {BUDGET_MS} ms budget "
        f"({100 * tp['p99'] / BUDGET_MS:.0f}% of it)."
    )
    if tp["p99"] <= BUDGET_MS:
        print("      No optimization work is justified (hard rule 11): optimizing a stage")
        print("      that already fits is theatre. Recorded so the decision is on file.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default="2011_09_26_drive_0059")
    ap.add_argument(
        "--split",
        nargs="*",
        help="bench every drive in these splits instead of one drive. At N=300 the p99 is the "
        "3rd-worst frame and the rare host stalls Phase 2 found (5 in 1450) will usually be "
        "missed, so a tail claim needs the whole split",
    )
    ap.add_argument("--frames", type=int, default=300, help="frames to measure")
    ap.add_argument("--warmup", type=int, default=30, help="frames excluded from statistics")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-lanes", dest="lanes", action="store_false")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = run(args)
    report(data, args, len(data["TOTAL (measured)"]))

    out = args.out or REPO / "artifacts" / "bench_latency.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
