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
| C-1 | "Built a LiDAR-projection ground-truth harness for monocular range, with the ruler's own error characterised at 0.25 m MAE (p95 0.63 m) across 0–50+ m." | [Row 1.6](benchmarks.md) | *pending* | val split, 5 drives, N=3315 gated boxes. 50+ m rests on 61 boxes. |
| C-2 | "Found that ground truth is undefined for occluded objects — a naive LiDAR-in-box estimator reports the occluder at 11.9 m MAE — so only 38% of labelled objects receive ground truth and every range figure is a lower bound." | [Row 1.2](benchmarks.md) | *pending* | val split. |
| C-4 | "Showed the ground-truth error beyond 50 m was bimodal (median 0.12 m, mean 4.75 m) and gated it on depth spread, cutting MAE from 0.41 m to 0.25 m and the failure rate from 1.06% to 0.27%." | [Row 1.3, 1.6](benchmarks.md) | *pending* | val split. Costs 5% of measurements, 27% at 50+ m. |
| C-5 | "Measured ego pitch excursions of 2.0–2.2° peak-to-peak in ordinary city driving — above the level that costs >10% monocular range error — including a sustained 1.03° offset on one drive." | [Row 1.5](benchmarks.md) | *pending* | OXTS vehicle pitch, not camera-relative-to-road pitch. Indicator, not correction ([D-009](decisions.md)). |
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
| Any range figure without "on the unoccluded 38%" | Coverage is part of the number. Omitting it overstates by an unknown amount. |
| The superseded dev figures (0.13 m, 64%) | Overturned by val. Quoting them would be quoting a ten-box sample. |
| "TensorRT / INT8 speedup" | No CUDA on this hardware; ladder dropped as inherited scope ([D-012](decisions.md)). |
| "Optimized inference by Nx" | No optimization performed. It is gated on a measured budget miss, and nothing has been measured yet. |
