# Autonomous Perception System (APS)

A **monocular** vision perception and forward-collision-warning stack, evaluated
against LiDAR ground truth before anything is claimed about it.

```
KITTI frame ─► detection ─► 2D tracking ─► monocular range ─► scale-rate velocity + KF ─► TTC ─► FCW / AEB-request
            └► lane solver ────────────► lane-departure metric ───────────────────────────────┘
Ground truth: projected LiDAR range
```

## Scope, stated plainly

A **single forward camera**. No LiDAR, no radar, and no fusion in the estimation
path — LiDAR appears only as ground truth for evaluation. No vehicle actuation.
The system emits **warnings and an AEB *request***, offline and open-loop.
**It does not brake anything.**

Production AEB is a safety-critical function under ISO 26262 and ISO 21448
(SOTIF), and production stacks fuse radar with camera precisely *because*
monocular range is unreliable. This project is not that, and the point of it is
to be able to say exactly why, with numbers.

## Status

**Phase 1 of 9 complete.** The ruler exists; the estimator it will measure does
not yet. That ordering is deliberate — an estimator with no ground truth is a
decoration.

| Phase | | |
|---|---|---|
| 0 | Repo, dataset, calibration, one frame rendered | ✅ |
| 1 | **LiDAR ground-truth harness** | ✅ |
| 2 | Detection baseline (mAP + latency) | — |
| 3 | **Monocular range + error-vs-range curve** ← headline | — |
| 4 | Tracking: IoU → SORT, MOTA/IDF1/ID-switches | — |
| 5 | Closing speed + TTC (scale-rate + KF) | — |
| 6 | Lane detection + departure metric | — |
| 7 | FCW decision layer: TPR **and FP/hour** | — |
| 8 | HUD + top-down view *(deliberately last)* | — |
| 9 | Writeup | — |

## What Phase 1 established

Ground truth is built by projecting the Velodyne sweep into the left colour
camera and reducing the returns inside each 2D box to a near-surface range.

![LiDAR projected into the camera, with labelled 3D boxes](docs/figures/phase0_projection_check.png)

**The ruler's own error, measured against independent human 3D annotations:**

| range bin | 0–10 m | 10–20 m | 20–30 m | 30–50 m | 50+ m |
|---|---|---|---|---|---|
| MAE | 0.05 m | 0.08 m | 0.15 m | 0.18 m | 0.22 m |

Overall MAE **0.13 m**, p95 error 0.37 m — roughly 20× tighter than the monocular
error Phase 3 expects to find, which is what makes it usable as a ruler.

**Three findings that shaped everything downstream:**

1. **Occluded objects have no ground truth.** Scored naively across all labelled
   boxes, the "ruler" had 3.19 m MAE. Split by the annotator's occlusion flag:
   0.10 m on visible objects, **11.86 m on fully-occluded ones** — because a box
   around an object you cannot see contains the *occluder's* surface, so the
   estimator confidently measures the wrong car. Those boxes are now abstentions.
   Consequence: ground truth exists for **64%** of labelled objects, and every
   range number this project reports is measured on unoccluded, untruncated
   objects and is therefore a **lower bound** on real-world error.

2. **Coverage is not a quality metric.** An adaptive fallback that raised the
   estimator's validity from 86% to 100% recovered boxes carrying **12× the
   error** (1.56 m vs 0.13 m); four of them degraded a whole range bin by 4.5×.
   Reverted, and a regression test now fails if it comes back.

3. **KITTI is not 10 Hz.** Measured frame interval is 103.56 ms — **9.657 Hz**.
   Hard-coding the documented rate would put a 3.4% systematic error into every
   closing-speed and TTC figure, indistinguishable afterwards from estimator bias.

Full context, with N and caveats, in [docs/benchmarks.md](docs/benchmarks.md).
Every non-obvious choice and why it was made: [docs/decisions.md](docs/decisions.md).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

python tools/fetch_kitti.py --split dev val   # ~4 GB, resumable, prunes unused sensors
python tools/fetch_kitti.py --check           # verify

python tools/build_range_gt.py --split dev    # characterise the ruler
python -m pytest tests/ && ruff check . && black --check .
```

The dataset is not committed; `tools/fetch_kitti.py` reproduces it exactly and
records URLs, byte counts, and SHA-256 in `data/kitti/MANIFEST.json`.

## Repository layout

| Path | |
|---|---|
| `aps/kitti/` | Calibration, drive loading, tracklet labels |
| `aps/groundtruth.py` | The ruler: LiDAR-in-box range, validity tiers |
| `aps/viz.py` | Debug rendering (not the HUD — that is Phase 8) |
| `tools/` | Dataset fetch, label audit, GT build, frame render |
| `configs/dataset.yaml` | The split. Changing it invalidates every benchmark row |
| `docs/` | Benchmarks, decisions, error budget, glossary, history |
| `tests/` | Closed-form geometry cases; 38 tests |
| `legacy/` | The v1 demo, archived. **Not a baseline** — see below |

## About `legacy/`

The first version of this project was a working demo: YOLO detection, an IoU
tracker, geometric depth, a lane solver, and a Flask HUD showing distances, a TTC
readout, and a BRAKE indicator. It had **no ground truth anywhere in it**, so
there was no way to know whether any number it displayed was right.

It is kept, intact and excluded from lint and tests, as the worked example of the
failure mode this repository is a correction to. It cannot appear in
`benchmarks.md` and is not a baseline — a baseline requires a measurement.

## License / data

KITTI raw data is distributed by Karlsruhe Institute of Technology and Toyota
Technological Institute at Chicago under CC BY-NC-SA 3.0. This repository
contains no KITTI data, only code that fetches it.
