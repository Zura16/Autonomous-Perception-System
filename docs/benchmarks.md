# Benchmarks

**The only source of numbers.** Nothing in the README, `claims.md`, a résumé, or
an interview answer may state a figure that is not a row here. Every row carries
its full context — dataset, split, N, hardware, and the caveat that makes it
honest. A row without context is deleted, not fixed.

Rows are appended, never edited. A superseded row gets a `→ superseded by` note
so the trajectory stays visible.

---

## Environment fingerprint

Every latency row below depends on this. **Changing any of it invalidates every
latency number**; accuracy rows survive a hardware change but not a split change.

| | |
|---|---|
| Machine | Apple M2, macOS 26.5.1, arm64 |
| Accelerator | **No CUDA.** Metal (MPS) available via torch |
| Python | 3.12.4 |
| numpy / opencv / torch | 2.5.1 / 5.0.0 / 2.13.0 |
| Recorded at commit | `da10a46` (Phase 0/1 work follows) |

**Consequence to state plainly:** the charter's TensorRT precision ladder
(FP32 → FP16 → INT8) is **not reproducible on this machine**. Any such claim
would require hardware this project does not have. Latency work here is
CPU/MPS on an M2 and will be labelled as such.

---

## Dataset

| | |
|---|---|
| Source | KITTI **raw**, date `2011_09_26` ([D-004](decisions.md)) |
| Camera | `image_02`, left colour, rectified, 1242 × 375 |
| Intrinsics | fx = fy = 721.5377 px · cx = 609.5593 · cy = 172.8540 · HFOV 81.4° |
| Frame rate | **9.657 Hz** measured (dt = 103.56 ms, σ = 0.06 ms) — *not* 10 Hz ([D-006](decisions.md)) |
| Split | by sequence ([D-002](decisions.md)) — dev `0013` · val `0059` `0084` `0091` `0056` `0027` · test `0009` `0015` |

### Label inventory

Near-face range bins, FCW classes (Car/Van/Truck/Pedestrian/Cyclist), boxes that
project into the image.

| split | drives | frames | boxes | 0–10 | 10–20 | 20–30 | 30–50 | 50+ |
|---|---|---|---|---|---|---|---|---|
| dev | 0013 | 144 | 385 | 54 | 58 | 102 | 130 | 41 |
| val | 0059 | 373 | 3810 | 465 | 823 | 769 | 1149 | 577 |
| val | 0084 | 383 | 2818 | 333 | 553 | 553 | 987 | 392 |
| val | 0091 | 340 | 1295 | 345 | 621 | 256 | 73 | 0 |
| val | 0056 | 294 | 740 | — | — | — | — | — |
| val | 0027 | 188 | 69 | — | — | — | — | — |
| test | 0009, 0015 | 744 | *unaudited by design* | | | | | |

`0091` is the only drive with meaningful VRU coverage (42 pedestrians,
8 cyclists) and contributes **zero** boxes beyond 50 m. `0027` is retained for
its road/closing-speed regime despite 69 boxes and is never pooled silently into
city numbers ([D-005](decisions.md)).

---

## Phase 1 — Ground-truth ruler

**What is being measured:** the error of the LiDAR-in-box range estimator itself,
scored against the independent human 3D box annotations. This is the ruler's own
error, and it is the floor under every range number this project will report.

**Method:** project the velodyne sweep into `image_02`, select returns inside each
labelled box, reduce to a near-surface range, compare to the labelled 3D box's
near face ([D-003](decisions.md)).

### Row 1.1 — Estimator selection, dev split

- **Data:** dev split (`2011_09_26_drive_0013`), 144 frames, 385 labelled boxes
- **Subset:** usable GT tier (strict + relaxed), N = 248 boxes
- **Metric:** absolute error vs labelled 3D box near face, metres
- **Commit:** *(this commit)*

| estimator | valid | MAE | bias | p95 \|e\| | 0–10 | 10–20 | 20–30 | 30–50 | 50+ |
|---|---|---|---|---|---|---|---|---|---|
| `min` | 99% | 2.08 | −2.02 | 14.44 | 0.14 | 0.34 | 0.83 | 3.57 | 6.38 |
| `p10` | 99% | 0.81 | −0.61 | 2.27 | 0.07 | 0.09 | 0.33 | 1.35 | 2.74 |
| `p20` | 99% | 0.41 | +0.03 | 0.69 | 0.23 | 0.13 | 0.36 | 0.52 | 1.11 |
| `median` | 99% | 1.64 | +1.63 | 5.78 | 2.81 | 0.69 | 1.54 | 1.75 | 2.28 |
| **`shrink_p20`** ← selected | **85%** | **0.13** | **+0.09** | **0.37** | **0.05** | **0.08** | **0.15** | **0.18** | **0.22** |
| `cluster_p20` | 98% | 1.05 | −0.75 | 7.82 | 0.19 | 0.11 | 0.24 | 2.24 | 1.92 |

**Reading it.** `shrink_p20` wins at every range bin, not on average — the
distinction hard rule 2 exists to enforce. It is the only estimator whose error
stays flat with range; `min` and `cluster_p20` degrade 30–45× from the near bin
to the far one, because at distance a thin foreground object (pole, sign, mirror)
supplies the nearest returns in the box.

**It also abstains most** (85% valid vs 98–99%). That is the trade being bought
deliberately: see Row 1.3.

### Row 1.2 — Ground-truth coverage, dev split

What fraction of labelled objects can be ground-truthed **at all**.

| bin (m) | labels | strict | relaxed | invalid | usable | median LiDAR pts/box |
|---|---|---|---|---|---|---|
| 0–10 | 54 | 30 | 0 | 24 | 56% | 1887 |
| 10–20 | 58 | 57 | 0 | 1 | 98% | 580 |
| 20–30 | 102 | 47 | 3 | 52 | 49% | 144 |
| 30–50 | 130 | 59 | 29 | 42 | 68% | 74 |
| 50+ | 41 | 0 | 23 | 18 | 56% | 24 |
| **all** | **385** | **193** | **55** | **137** | **64%** | |

Ruler error by tier (same estimator, dev split):

| tier | N | MAE |
|---|---|---|
| strict — annotator says fully visible | 227 | **0.10 m** |
| relaxed — occlusion unlabelled | 33 | 0.28 m |
| *(excluded)* partly occluded | 22 | 1.33 m |
| *(excluded)* fully occluded | 79 | **11.86 m** |

**The caveat that travels with every range number in this project:** ground truth
exists for 64% of labelled objects, and those are the *unoccluded, untruncated*
ones. Every reported range error is therefore measured on the easy half and is a
**lower bound** on real-world error ([D-010](decisions.md)).

LiDAR support also falls ~1/range² — from 1887 returns per box at 0–10 m to 24 at
50+ m. The ruler weakens exactly where the monocular estimator it will measure
also weakens.

### Row 1.3 — Negative result: coverage bought at 12× the error

Adaptive shrink fallback (0.25 → 0.125 → 0.0), dev split, usable tier.

| shrink used | N | MAE | p95 \|e\| |
|---|---|---|---|
| 0.25 (no fallback needed) | 212 | **0.130** | 0.37 |
| 0.125 (fallback) | 25 | **1.563** | 11.15 |
| 0.0 (fallback) | 9 | 0.481 | 1.81 |

Validity rose 86% → 100%; the 20–30 m bin degraded 0.15 m → 0.69 m on the
strength of **four** recovered boxes. Reverted ([D-011](decisions.md)), and a
regression test now fails if the fallback returns.

### Row 1.4 — Convention gap, measured

| quantity | dev split |
|---|---|
| centroid range − near-face range | mean **+2.03 m** (p05 1.79, p95 2.69) |

Scoring a near-face estimator against centroid truth would manufacture exactly
this as a pure bias and then attribute it to the geometry ([D-003](decisions.md)).

### Row 1.5 — Ego pitch excursion, dev split

| | |
|---|---|
| mean | +0.250° |
| range | −0.253° … +0.794° |
| **peak-to-peak** | **1.05°** |

On a gentle city drive **containing no hard braking**. Against the stated
sensitivity (2° ≈ >10% range error), this is already a material term before the
manoeuvre an FCW system exists for ([D-009](decisions.md)).

---

## Pending — nothing measured yet

These rows are deliberately empty. A value here that is not a measurement is the
failure mode this file exists to prevent.

| Row | Blocks on |
|---|---|
| 2.x — Detection mAP + latency ladder (M2/MPS) | Phase 2 |
| 3.x — **Monocular range error vs range** ← headline | Phase 3 |
| 3.y — Credible operating envelope (range where error < X%) | Phase 3 |
| 4.x — Tracking MOTA / IDF1 / ID-switches, IoU vs SORT | Phase 4 |
| 5.x — Closing-speed error; TTC error at TTC < 3 s | Phase 5 |
| 6.x — Lane departure detection rate / FP rate | Phase 6 |
| 7.x — FCW TPR and **FP per hour** at a named TTC threshold | Phase 7 |
| 8.x — Per-stage latency p50/p95/p99 | Phase 8 |
