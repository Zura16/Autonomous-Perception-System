# Claims

Every external-facing claim — README, résumé, portfolio, interview answer —
mapped to the benchmark row and commit hash that proves it.

**Rule:** claims are *generated from* [benchmarks.md](benchmarks.md), never
authored. Writing a claim before its measurement exists is the one thing that
gets a change rejected outright. `/claim-check` audits this file; any unbacked
row is a failure, not a warning.

---

## Backed claims

| # | Claim | Backed by | Commit | Caveat that must travel with it |
|---|---|---|---|---|
| C-1 | "Built a LiDAR-projection ground-truth harness for monocular range, with the ruler's own error characterised at 0.13 m MAE (p95 0.37 m) across 0–50+ m." | [Row 1.1](benchmarks.md) | *pending* | dev split, N=248 usable boxes, one drive. Val figures not yet computed. |
| C-2 | "Found that ground truth is undefined for occluded objects — a naive LiDAR-in-box estimator reports the occluder at 11.9 m MAE — so 36% of labelled objects are abstentions and every range figure is a lower bound." | [Row 1.2](benchmarks.md) | *pending* | dev split. The 64% coverage figure is per-drive and will shift on val. |
| C-3 | "Measured KITTI's actual frame interval at 103.56 ms (9.657 Hz) rather than the documented 10 Hz, avoiding a 3.4% systematic error in every TTC figure." | [Dataset](benchmarks.md) | *pending* | Measured on `drive_0013`; consistent across the drive (σ = 0.06 ms). |

Commit hashes are filled in by `/claim-check` once the measurement and the claim
are in the same committed state.

---

## Explicitly NOT claimable

Kept here because the tempting overstatement is more dangerous than the gap.

| Tempting phrasing | Why it is false here |
|---|---|
| "Sensor fusion" | LiDAR is ground truth only. It is not in the estimation path. |
| "Autonomous emergency braking" | The system emits a *request*. Nothing is actuated. |
| "Real-time" | No latency budget has been named or measured yet. |
| "Accurate to ±X m" (single figure) | Forbidden by hard rule 2 — error must be binned by range. |
| Any tracking number "vs KITTI leaderboard" | Raw tracklets, not the official tracking split ([D-004](decisions.md)). |
| Anything about detection, TTC, or FCW | Not built. Phases 2–7. |
| "TensorRT / INT8 speedup" | No CUDA on this hardware. Not reproducible here. |
