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
| C-1 | "Built a LiDAR-projection ground-truth harness for monocular range, with the ruler's own error characterised at 0.25 m MAE (p95 0.63 m) across 0–50+ m." | [Row 1.6](benchmarks.md) | `3edc08d` | val split, 5 drives, N=3315 gated boxes. 50+ m rests on 61 boxes. |
| C-2 | "Found that ground truth is undefined for occluded objects — a naive LiDAR-in-box estimator reports the occluder at 11.9 m MAE — so only 38% of labelled objects receive ground truth and every range figure is a lower bound." | [Row 1.2](benchmarks.md) | `b1f5eb7` | val split. |
| C-4 | "Showed the ground-truth error beyond 50 m was bimodal (median 0.12 m, mean 4.75 m) and gated it on depth spread, cutting MAE from 0.41 m to 0.25 m and the failure rate from 1.06% to 0.27%." | [Row 1.3, 1.6](benchmarks.md) | `b1f5eb7` | val split. Costs 5% of measurements, 27% at 50+ m. |
| C-5 | "Measured ego pitch excursions of 2.0–2.2° peak-to-peak in ordinary city driving — above the level that costs >10% monocular range error — including a sustained 1.03° offset on one drive." | [Row 1.5](benchmarks.md) | `b1f5eb7` | OXTS vehicle pitch, not camera-relative-to-road pitch. Indicator, not correction ([D-009](decisions.md)). |
| C-3 | "Measured KITTI's actual frame interval at 103.56 ms (9.657 Hz) rather than the documented 10 Hz, avoiding a 3.4% systematic error in every TTC figure." | [Row 0.1](benchmarks.md) | `fce762e` | Measured on `drive_0013`; consistent across the drive (σ = 0.06 ms). |

| C-6 | "Established a zero-shot YOLOv8n detection baseline on KITTI: class-agnostic AP@0.5 of 0.615, with recall characterised against range and object pixel height." | [Row 2.1–2.3](benchmarks.md) | `8c7254c` | val, 1450 frames, 8705 labels. COCO weights, no fine-tuning. Precision is pessimistic — KITTI raw does not label every object. |
| C-7 | "Showed VRU detection recall collapses from 75% under 10 m to 2% at 30–50 m, bounding any pedestrian/cyclist collision-warning claim at roughly 30 m." | [Row 2.2](benchmarks.md) | `8c7254c` | val. YOLOv8n at 640 px specifically; a larger model or input size would move this. |
| C-8 | "Determined the project's credible evaluation envelope is 0–50 m by joining detection recall with ground-truth coverage per label — 30 usable objects remain beyond 50 m." | [Row 2.6](benchmarks.md) | `8c7254c` | val. A statement about evidence availability, NOT about estimator accuracy ([D-015](decisions.md)). |
| C-9 | "Measured detection latency at 33% of the 103.56 ms sensor budget at p99 on an M2, and identified five host stalls exceeding the budget that the p99 concealed." | [Row 2.5](benchmarks.md) | `8c7254c` | Detection only; other pipeline stages not yet measured. MPS, 640 px, batch 1. |

| C-10 | "Produced a monocular range error-vs-range curve on 6122 detections, graded by a LiDAR ruler applied to the detector's own boxes, and held the resulting operating envelope to a held-out split — where it failed." | [Row 3.1–3.2, 9.3](benchmarks.md) | `fce762e` | **The 10–30 m ≤15% MAPE envelope was declared on val and did NOT hold on test (17.2% / 15.6%).** Never quote the val envelope alone. Post-audit envelope: 20–50 m at ≤13% on both splits. Measured on the ~33% of objects both detected and ground-truthable — a lower bound. |
| C-11 | "Found the monocular error curve is U-shaped — worst inside 10 m at 21.0% MAPE, not at range." | [Row 3.1, 9.7](benchmarks.md) | `fce762e` | val, post-D-025. The *cause* is only partly identified — see C-14. Do not attribute it wholly to box bias. Held-out test agrees (22.5%). |
| C-14 | "Measured detector box-height bias per object against its own label (−4.5% overall, −7.1% inside 10 m), correcting an earlier figure that overstated it 2.6× by comparing against a class mean." | [Row 2.7](benchmarks.md) | `5e75904` | val, 5254 matched pairs. Accounts for ~80% of the mid-range range bias and only ~17% of the near-field bias. |
| C-12 | "Measured true camera-to-road pitch by fitting the road plane to LiDAR (mean +0.150°, p95 0.715°), correcting an OXTS-derived figure that overstated it 2–3× by including road grade." | [Row 3.7](benchmarks.md) | `56495e9` | Drives with no hard braking, so a floor on the operational distribution. |
| C-13 | "Showed by measurement that the flat-ground pitch assumption is not this system's bottleneck: error is flat across measured pitch where geometry predicts an 8× span." | [Row 3.4](benchmarks.md) | `56495e9` | True only while the box bias dominates. Pitch becomes binding once that is fixed. |

| C-15 | "Measured that road non-flatness is the dominant far-range error in monocular contact-point ranging, predicting the bias from LiDAR road geometry with no free parameters and matching observation to 0.1–0.6 m beyond 20 m." | [Row 3.9](benchmarks.md) | `a46bf1b` | val. Inside 13 m it explains almost nothing; a +0.7 m residual is unexplained. |
| C-16 | "Ablated SORT against an IoU baseline on identical detections: association buys 29% fewer ID switches, while the canonical Kalman-smoothed output costs MOTP and was discarded." | [Row 4.1–4.2](benchmarks.md) | `b768952` | val, self-implemented metrics, raw-tracklet labels. NOT leaderboard-comparable. |
| C-17 | "Showed MOTA on this data is detection-bound rather than association-bound — ~3450 of 8705 objects are never detected in any tracker configuration." | [Row 4.4](benchmarks.md) | `b768952` | val. Consistent with the measured 60% detection recall. |

| C-18 | "Built a calibration-free monocular TTC estimator — a Kalman filter over inverse box height — with 0.41 s MAE and 0.01 s median error at TTC under 3 s." | [Row 5.4](benchmarks.md) | `40659a7` | val, common subset N=1930, raw tracklet ground truth. Offline, open-loop. |
| C-19 | "Measured that without a deadband 54% of stationary objects report a phantom closing speed, and predicted that rate from a measured detector noise model to within a tenth of a point." | [Row 5.1–5.2](benchmarks.md) | `40659a7` | val. The prediction holds for none/fixed deadbands; sigma's real rate is 13× its synthetic one. |
| C-20 | "Showed an adaptive (n-sigma) deadband's apparent TTC accuracy gain was selection bias: it goes silent on 10% of imminent-threat frames." | [Row 5.4](benchmarks.md) | `40659a7` | val, GT TTC < 3 s. |

| C-21 | "Built a metric bird's-eye lane solver scored on 95 human-labelled KITTI road frames with parameters frozen before evaluation: 97.9% detection, 0.18 m lateral offset error." | [Row 6.1](benchmarks.md) | `7d5310a` | Urban daytime single frames, N=95. 0.18 m excludes 2 straddling frames identified from ground truth; 0.26 m including them. |
| C-22 | "Showed per-frame lane-departure hit rates at a hard threshold are dominated by threshold-grazing frames, and reported departure detection by ground-truth condition instead." | [Row 6.3](benchmarks.md) | `7d5310a` | 3/5 body-over-line, 95% CI 23–88% — too few to claim a rate. |

| C-23 | "Built and swept a monocular forward-collision-warning layer, and established that the dataset cannot validate it: 2 in-path threat events below a 2 s TTC in 2.7 minutes of driving." | [Row 7.1–7.2](benchmarks.md) | `2065557` | val. No TPR or FP/hour published; no operating point selected. |
| C-24 | "Measured that warning persistence cuts false alarms six-fold for 0.2 s of latency, and that half of all false alarms are detector ghosts rather than decision errors." | [Row 7.3–7.4](benchmarks.md) | `2065557` | val, 2 events — direction is legible, magnitude is not. |

| C-25 | "Ran the frozen pipeline once on a held-out split and reported that it broke my headline claim: the 10–30 m ≤15% MAPE envelope came back at 17.2% and 15.6%." | [Row 9.3](benchmarks.md) | `fce762e` | test split, 744 frames, N=3214. Ruler (0.27 m MAE) and detector (AP 0.737) both transferred or improved, so the failure is the estimator's. |
| C-26 | "Showed the generalisation gap was entirely in the tail: median APE moved ~1–3 points while mean APE moved 8, so the headline metric described a collapse that never happened to the typical object." | [Row 9.4](benchmarks.md) | `fce762e` | test vs val, contact-point. Consequence: every range row now reports median beside mean. |
| C-27 | "Traced the failure to 10 detections of 199 carrying 81% of a block's bias — boxes sitting 4.7 px below the horizon, where the contact-point relation diverges and returned 253 m for an object at 25.8 m — after eliminating degraded ground truth, occlusion, and road slope by measurement." | [Row 9.5–9.6](benchmarks.md) | `fce762e` | test. Occlusion is a real but secondary contributor (8 of 199 boxes). The three eliminated hypotheses are part of the claim, not omitted from it. |
| C-28 | "Found a validity guard that was only a division-by-zero check — it permitted a 398 m answer at 33% range error per pixel — and replaced it with abstention beyond the already-declared 50 m envelope, which also cut val's 30–50 m error from 18.7% to 11.8%." | [Row 9.7](benchmarks.md) | `fce762e` | **The defect was found by inspecting test, so post-fix test numbers are not clean held-out evidence.** The fix does not rescue the claim: test 10–20 m remains 15.8%. Costs coverage — contact-point validity 90% on test, 71% in the 0–10 m bin. |
| C-29 | "Measured scene-to-scene variation larger than every effect the project had previously characterised: 7.7% MAPE on one test sequence against 23.3% on the other, and 6.6% against 55.5% between blocks of the same drive." | [Row 9.5](benchmarks.md) | `fce762e` | Contact-point, 10–30 m. Five val sequences spanning 8.5–12.4% had looked like convergence. |

Commit hashes are filled in by `/claim-check` once the measurement and the claim
are in the same committed state.

---

## Explicitly NOT claimable

Kept here because the tempting overstatement is more dangerous than the gap.

| Tempting phrasing | Why it is false here |
|---|---|
| "Sensor fusion" | LiDAR is ground truth only. It is not in the estimation path. |
| "Autonomous emergency braking" | The system emits a *request*. Nothing is actuated. |
| "Real-time" | Only *detection* has been measured against the 103.56 ms budget (p99 34.01 ms). The other stages are untimed, so the pipeline has no measured end-to-end latency. |
| "Accurate to ±X m" (single figure) | Forbidden by hard rule 2 — error must be binned by range. |
| Any tracking number "vs KITTI leaderboard" | Raw tracklets AND self-implemented metrics ([D-004](decisions.md), [D-021](decisions.md)). Two independent reasons. |
| "Implemented SORT" without the ablation | The shipped tracker deliberately discards SORT's output smoothing; saying "SORT" alone misstates what was built and drops the finding. |
| An FCW detection rate or false-alarms-per-hour figure | **Deliberately not produced** — 2 threat events, 0.045 h exposure ([D-024](decisions.md)). Quoting one would be fitting noise. |
| "Tuned the FCW operating point" | No operating point was selected; the sweep is published, the choice is not made. |
| "Lane departure warning with X% detection rate" | 5 real departure frames. The interval is 23–88%; a single rate would be fiction. |
| Lane performance in night, rain, construction, curves | Not in the labelled set. Uncharacterised. |
| "Accurate closing speed" | TTC is accurate; closing speed inherits range error (78% relative at 30–50 m). Say which one. |
| "The sigma deadband is more accurate" | It is not — identical TTC on a common subset. It emits fewer TTCs. |
| Per-class detection mAP (Car vs Van vs Truck) | Not recoverable from a COCO-trained model; the number would measure the mapping ([D-014](decisions.md)). |
| "Monocular range is unreliable beyond 50 m" | NOT measured. 50 m is where the *evidence* runs out, not where the estimator fails — and since D-025 the estimator **abstains** there, so it is now unmeasurable there by construction. |
| "Accurate to X% monocular range" (single figure) | Forbidden by hard rule 2, and the curve is U-shaped — a single figure hides that the near field is worst. |
| "Pitch does not matter for monocular range" | It does; it is currently *masked* by other error ([D-017](decisions.md)). The claim is conditional, not general. |
| "Validated on a held-out split" (unqualified) | The held-out split **refuted** the declared envelope ([Row 9.3](benchmarks.md)). It was run, not passed. Saying "validated" inverts the result. |
| "10–30 m at ≤15% MAPE" | The val-only envelope, **refuted on test** (17.2% / 15.6%). Superseded by 20–50 m at ≤13% on both splits. |
| Post-D-025 test numbers as held-out evidence | The defect was found by inspecting test, so those numbers are post-hoc ([D-025](decisions.md)). Only the as-frozen result is clean. |
| A single MAPE figure for any range bin | Mean and median diverge by up to 8 points on test. Quote both, or quote the median and say so. |
| "Detector box bias explains the U-shaped error curve" | Measured: it accounts for ~80% of the mid-range bias and only ~17% of the near-field bias. The near-field cause is **unidentified** ([D-018](decisions.md)). |
| The superseded −12% box-bias figure | Overstated 2.6×; it absorbed the class's 20.1% height spread. |
| Comparing AP 0.615 to a KITTI leaderboard | Zero-shot COCO model, raw-tracklet labels, class-agnostic protocol. Not the same task. |
| Any range figure without "on the unoccluded 38%" | Coverage is part of the number. Omitting it overstates by an unknown amount. |
| The superseded dev figures (0.13 m, 64%) | Overturned by val. Quoting them would be quoting a ten-box sample. |
| "TensorRT / INT8 speedup" | No CUDA on this hardware; ladder dropped as inherited scope ([D-012](decisions.md)). |
| "Optimized inference by Nx" | No optimization performed. It is gated on a measured budget miss, and nothing has been measured yet. |
