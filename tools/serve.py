"""Run the pipeline as a live local process, behind the site's own UI.

    python tools/serve.py                       # http://127.0.0.1:8000
    python tools/serve.py --drive 2011_09_26_drive_0084 --start 130

The frames are computed here, now, one at a time -- this is the real pipeline
executing, not the pre-rendered clip the published page plays. Per-frame timings
come from the same clock `bench/run_bench.py` uses, so the latency the UI shows
is the latency the stage actually took.

**Local only, and that distinction is the whole point.** The published page is
static and must never claim otherwise: the v1 site shipped play/pause controls
wired to `http://127.0.0.1:5000`, so every visitor's browser tried to reach a
backend on *their own* machine and every control was dead. Here the server binds
loopback, serves `site/` itself, and the page enables live controls only when it
is being served from localhost. On GitHub Pages it never probes for an API,
because there is none to find.

Why KITTI rather than a webcam: every range number depends on KITTI's
intrinsics, its 1242x375 frame and a LiDAR-measured 1.655 m mount. On another
camera, range and lane offset are wrong by an unknown scale and there is no
ground truth to catch it. (TTC would survive -- it is calibration-free -- but a
live mode that silently invalidates two of its four outputs is not worth the
demo.)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.replay import build_frame  # noqa: E402
from aps.detect import Detector  # noqa: E402
from aps.fcw import FcwConfig, FcwDecider, corridor_range_m, lateral_offset_m  # noqa: E402
from aps.geometry import (  # noqa: E402
    CameraModel,
    CredibleEnvelope,
    range_contact_point,
    range_size_prior,
)
from aps.kitti import load_split  # noqa: E402
from aps.kitti.road import RoadGeometry  # noqa: E402
from aps.lanes import BevGrid, LaneConfig, estimate_lane  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402
from aps.viz import CLASS_COLOURS, DEFAULT_COLOUR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"


def _clean(v: float) -> float | None:
    """NaN cannot cross JSON, and 0.0 would be a lie. Abstentions go as null."""
    return float(v) if np.isfinite(v) else None


class Pipeline:
    """The stack, stepped one frame at a time from a worker thread.

    All pipeline state lives here and is touched only by that thread. The HTTP
    handlers read a snapshot under a lock and never call into the model: the
    detector is not thread-safe, and two requests entering it concurrently would
    corrupt tracker and filter state in ways that look like estimator bugs.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        drives = {d.id: d for d in load_split(["dev", "val"])}
        if args.drive not in drives:
            raise SystemExit(f"unknown drive {args.drive}. Available: {sorted(drives)}")
        self.drive = drives[args.drive]
        self.args = args

        self.camera = CameraModel.from_config()
        self.envelope = CredibleEnvelope.from_config()
        self.lane_cfg = LaneConfig.from_config()
        self.fcw_cfg = FcwConfig.from_config()
        self.detector = Detector(args.weights, device=args.device)
        self.lane_geom = RoadGeometry.nominal(
            self.drive.calib.P_rect[2], self.lane_cfg.camera_height_m
        )
        self.grid = BevGrid.from_config(self.lane_cfg)
        self.fx, _, self.cx, _ = self.drive.calib.intrinsics(2)

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {"frame": None, "objects": [], "warning": False, "aeb_request": False}
        self.index = args.start
        self.playing = not args.paused
        self._reset_filters()

        self.detector.warmup(self.drive.frame(args.start).image())
        threading.Thread(target=self._loop, daemon=True).start()

    def _reset_filters(self) -> None:
        """Tracker, filter and decider all carry history across frames.

        Seeking without clearing them splices two moments of the drive together
        and manufactures a closing speed from the seam -- the same failure an ID
        switch causes (docs/decisions.md D-021).
        """
        self.tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
        self.motion = ScaleRateEstimator(MotionConfig.from_config(), deadband="fixed")
        self.decider = FcwDecider(self.fcw_cfg)
        self._last_t: dict[int, float] = {}

    def seek(self, index: int) -> None:
        with self._lock:
            self.index = max(0, min(len(self.drive) - 1, index))
            self._reset_filters()

    def _loop(self) -> None:
        period = 1.0 / self.args.fps if self.args.fps else 0.0
        while True:
            if not self.playing:
                time.sleep(0.05)
                continue
            started = time.perf_counter()
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001 - a worker death must be visible
                # Without this the thread dies silently and the UI freezes on the
                # last good frame, which reads as "slow" rather than "broken".
                print(f"pipeline error at frame {self.index}: {exc!r}", file=sys.stderr)
                self.playing = False
                continue
            elapsed = time.perf_counter() - started
            if period > elapsed:
                time.sleep(period - elapsed)

    def _step(self) -> None:
        fi = self.index
        t0 = time.perf_counter()
        image = self.drive.frame(fi).image()
        t_read = time.perf_counter()

        dets = self.detector.detect(image).above(self.args.conf)
        t_detect = time.perf_counter()

        tracks = self.tracker.update(dets.boxes, dets.scores)
        now = float(self.drive.timestamps_s[fi])
        group_of = {tuple(np.round(b, 3)): g for b, g in zip(dets.boxes, dets.groups, strict=True)}

        objects, any_warn, any_aeb = [], False, False
        for tr in tracks:
            box = tr.box
            group = group_of.get(tuple(np.round(box, 3)), "vehicle")
            contact = range_contact_point(box, self.drive.calib, self.camera)
            prior = range_size_prior(box, self.drive.calib, self.camera, group)
            dt = now - self._last_t.get(tr.track_id, now)
            self._last_t[tr.track_id] = now
            state = self.motion.update(tr.track_id, float(box[3] - box[1]), dt, contact)
            lateral = lateral_offset_m(
                0.5 * (box[0] + box[2]), self.fx, self.cx,
                corridor_range_m(contact, prior, self.fcw_cfg),
            )  # fmt: skip
            out = self.decider.update(tr.track_id, state.is_closing, state.ttc_s, lateral)
            any_warn |= out.warning
            any_aeb |= out.aeb_request
            objects.append(
                {
                    "track_id": int(tr.track_id),
                    "box": [float(v) for v in box],
                    "range_m": contact,
                    "ttc_s": state.ttc_s if state.is_closing else float("nan"),
                    "lateral_m": lateral,
                    "warning": bool(out.warning),
                    "group": group,
                }
            )
        t_estimate = time.perf_counter()

        lane, _ = estimate_lane(image, self.lane_geom, self.lane_cfg, self.grid)
        t_lanes = time.perf_counter()

        canvas = build_frame(
            image,
            [
                {
                    **o,
                    "colour": CLASS_COLOURS.get(
                        "Car" if o["group"] == "vehicle" else "Pedestrian", DEFAULT_COLOUR
                    ),
                }
                for o in objects
            ],
            lane,
            self.fcw_cfg,
            self.lane_geom,
            any_warn,
            any_aeb,
        )
        if canvas.shape[1] > self.args.width:
            s = self.args.width / canvas.shape[1]
            canvas = cv2.resize(canvas, (self.args.width, int(round(canvas.shape[0] * s))))
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])

        snapshot = {
            "frame": fi,
            "t_s": round(now, 3),
            "warning": any_warn,
            "aeb_request": any_aeb,
            "objects": [
                {
                    "id": o["track_id"],
                    "group": o["group"],
                    "range_m": _clean(o["range_m"]),
                    "ttc_s": _clean(o["ttc_s"]),
                    "lateral_m": _clean(o["lateral_m"]),
                    "in_envelope": self.envelope.contains(o["range_m"]),
                    "warning": o["warning"],
                }
                for o in objects
            ],
            # Measured here, this run, on this machine -- not read from a
            # benchmark row. A UI quoting benchmarks.md while claiming to be
            # live would be reporting someone else's clock.
            "timing_ms": {
                "read": round(1000 * (t_read - t0), 2),
                "detect": round(1000 * (t_detect - t_read), 2),
                "track+range+ttc+fcw": round(1000 * (t_estimate - t_detect), 2),
                "lanes": round(1000 * (t_lanes - t_estimate), 2),
                "total": round(1000 * (time.perf_counter() - t_read), 2),
            },
            "budget_ms": 103.56,
        }

        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
            self._state = snapshot
            self.index = (fi + 1) % len(self.drive)
            if self.index == 0:
                self._reset_filters()  # wrapped around: do not splice the loop seam

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self._lock:
            return self._jpeg, dict(self._state)

    def status(self) -> dict:
        with self._lock:
            return {
                "live": True,
                "drive": self.drive.id,
                "split": self.drive.split,
                "scene": self.drive.scene,
                "n_frames": len(self.drive),
                "index": self.index,
                "playing": self.playing,
                "device": self.args.device,
                "envelope_m": [self.envelope.min_range_m, self.envelope.max_range_m],
                "note": (
                    "Live local pipeline over KITTI. Offline dataset, open-loop, "
                    "nothing actuated."
                ),
            }


class Handler(SimpleHTTPRequestHandler):
    pipeline: Pipeline

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, directory=str(SITE), **kw)

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api/") and "frame" not in self.path:
            super().log_message(fmt, *args)  # frame polling would drown the log

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        route = urlparse(self.path).path
        if route == "/api/status":
            return self._json(self.pipeline.status())
        if route == "/api/telemetry":
            return self._json(self.pipeline.snapshot()[1])
        if route == "/api/frame":
            jpeg, _ = self.pipeline.snapshot()
            if jpeg is None:
                return self._json({"error": "no frame yet"}, 503)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(jpeg)
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route != "/api/control":
            return self._json({"error": "unknown endpoint"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        params = parse_qs(self.rfile.read(length).decode())
        action = (params.get("action") or [""])[0]
        if action == "play":
            self.pipeline.playing = True
        elif action == "pause":
            self.pipeline.playing = False
        elif action == "seek":
            self.pipeline.seek(int(float((params.get("index") or ["0"])[0])))
        else:
            return self._json({"error": f"unknown action {action!r}"}, 400)
        return self._json(self.pipeline.status())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default="2011_09_26_drive_0084")
    ap.add_argument("--start", type=int, default=130)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fps", type=float, default=9.657, help="0 = as fast as it computes")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--paused", action="store_true")
    args = ap.parse_args()

    print(f"starting pipeline on {args.drive} (device={args.device})…")
    Handler.pipeline = Pipeline(args)

    # Loopback only. This serves a process that runs a 2 GB model on demand;
    # binding 0.0.0.0 would expose it to the network with no auth at all.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  live UI:  http://127.0.0.1:{args.port}/\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
