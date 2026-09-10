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

| C-6 | "Established a zero-shot YOLOv8n detection baseline on KITTI: class-agnostic AP@0.5 of 0.615, with recall characterised against range and object pixel height." | [Row 2.1–2.3](benchmarks.md) | *pending* | val, 1450 frames, 8705 labels. COCO weights, no fine-tuning. Precision is pessimistic — KITTI raw does not label every object. |
| C-7 | "Showed VRU detection recall collapses from 75% under 10 m to 2% at 30–50 m, bounding any pedestrian/cyclist collision-warning claim at roughly 30 m." | [Row 2.2](benchmarks.md) | *pending* | val. YOLOv8n at 640 px specifically; a larger model or input size would move this. |
| C-8 | "Determined the project's credible evaluation envelope is 0–50 m by joining detection recall with ground-truth coverage per label — 30 usable objects remain beyond 50 m." | [Row 2.6](benchmarks.md) | *pending* | val. A statement about evidence availability, NOT about estimator accuracy ([D-015](decisions.md)). |
| C-9 | "Measured detection latency at 33% of the 103.56 ms sensor budget at p99 on an M2, and identified five host stalls exceeding the budget that the p99 concealed." | [Row 2.5](benchmarks.md) | *pending* | Detection only; other pipeline stages not yet measured. MPS, 640 px, batch 1. |

| C-10 | "Produced a monocular range error-vs-range curve on 6122 detections, graded by a LiDAR ruler applied to the detector's own boxes, establishing a credible operating envelope of 10–30 m at ≤15% MAPE." | [Row 3.1–3.2](benchmarks.md) | *pending* | val. No bin reaches 10% MAPE. Measured on the ~33% of objects both detected and groundtruthable — a lower bound. |
| C-11 | "Found the monocular error curve is U-shaped — worst inside 10 m at 22.7% MAPE, not at range." | [Row 3.1](benchmarks.md) | *pending* | val. The *cause* is only partly identified — see C-14. Do not attribute it wholly to box bias. |
| C-14 | "Measured detector box-height bias per object against its own label (−4.5% overall, −7.1% inside 10 m), correcting an earlier figure that overstated it 2.6× by comparing against a class mean." | [Row 2.7](benchmarks.md) | *pending* | val, 5254 matched pairs. Accounts for ~80% of the mid-range range bias and only ~17% of the near-field bias. |
| C-12 | "Measured true camera-to-road pitch by fitting the road plane to LiDAR (mean +0.150°, p95 0.715°), correcting an OXTS-derived figure that overstated it 2–3× by including road grade." | [Row 3.7](benchmarks.md) | *pending* | Drives with no hard braking, so a floor on the operational distribution. |
| C-13 | "Showed by measurement that the flat-ground pitch assumption is not this system's bottleneck: error is flat across measured pitch where geometry predicts an 8× span." | [Row 3.4](benchmarks.md) | *pending* | True only while the box bias dominates. Pitch becomes binding once that is fixed. |

| C-15 | "Measured that road non-flatness is the dominant far-range error in monocular contact-point ranging, predicting the bias from LiDAR road geometry with no free parameters and matching observation to 0.1–0.6 m beyond 20 m." | [Row 3.9](benchmarks.md) | *pending* | val. Inside 13 m it explains almost nothing; a +0.7 m residual is unexplained. |
| C-16 | "Ablated SORT against an IoU baseline on identical detections: association buys 29% fewer ID switches, while the canonical Kalman-smoothed output costs MOTP and was discarded." | [Row 4.1–4.2](benchmarks.md) | *pending* | val, self-implemented metrics, raw-tracklet labels. NOT leaderboard-comparable. |
| C-17 | "Showed MOTA on this data is detection-bound rather than association-bound — ~3450 of 8705 objects are never detected in any tracker configuration." | [Row 4.4](benchmarks.md) | *pending* | val. Consistent with the measured 60% detection recall. |

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
| Any tracking number "vs KITTI leaderboard" | Raw tracklets AND self-implemented metrics ([D-004](decisions.md), [D-021](decisions.md)). Two independent reasons. |
| "Implemented SORT" without the ablation | The shipped tracker deliberately discards SORT's output smoothing; saying "SORT" alone misstates what was built and drops the finding. |
| Anything about TTC or FCW | Not built. Phases 5–7. |
| Per-class detection mAP (Car vs Van vs Truck) | Not recoverable from a COCO-trained model; the number would measure the mapping ([D-014](decisions.md)). |
| "Monocular range is unreliable beyond 50 m" | NOT measured. 50 m is where the *evidence* runs out, not where the estimator fails. |
| "Accurate to X% monocular range" (single figure) | Forbidden by hard rule 2, and the curve is U-shaped — a single figure hides that the near field is worst. |
| "Pitch does not matter for monocular range" | It does; it is currently *masked* by other error ([D-017](decisions.md)). The claim is conditional, not general. |
| "Detector box bias explains the U-shaped error curve" | Measured: it accounts for ~80% of the mid-range bias and only ~17% of the near-field bias. The near-field cause is **unidentified** ([D-018](decisions.md)). |
| The superseded −12% box-bias figure | Overstated 2.6×; it absorbed the class's 20.1% height spread. |
| Comparing AP 0.615 to a KITTI leaderboard | Zero-shot COCO model, raw-tracklet labels, class-agnostic protocol. Not the same task. |
| Any range figure without "on the unoccluded 38%" | Coverage is part of the number. Omitting it overstates by an unknown amount. |
| The superseded dev figures (0.13 m, 64%) | Overturned by val. Quoting them would be quoting a ten-box sample. |
| "TensorRT / INT8 speedup" | No CUDA on this hardware; ladder dropped as inherited scope ([D-012](decisions.md)). |
| "Optimized inference by Nx" | No optimization performed. It is gated on a measured budget miss, and nothing has been measured yet. |
