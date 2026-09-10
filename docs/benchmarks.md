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

**Latency budget: 103.56 ms/frame** (the measured 9.657 Hz sensor rate). Every
latency row is reported against this denominator; "real-time" without it is
marketing.

**Consequence to state plainly:** the TensorRT precision ladder
(FP32 → FP16 → INT8) is **not reproducible on this machine** and has been
dropped as inherited scope from the superseded FuseTrack charter
([D-012](decisions.md)). Optimization is gated on a *measured* budget miss.
Latency work here is CPU/MPS on an M2 and will be labelled as such — with
`torch.mps.synchronize()` before the clock stops, since timing an async
dispatch measures the launch, not the work.

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

> **Rows 1.1–1.5 are VAL** (N = 8705 boxes, 5 drives). The dev-only figures
> reported on 2026-09-01 are superseded and kept at the foot of this section:
> dev is one quiet drive and its far-range sample was 10 boxes, which made the
> 50+ m bin look four times better than it is.

### Row 1.1 — Estimator selection, val split

- **Data:** val split, 5 drives, 1518 labelled frames, 8705 labelled boxes
- **Subset:** usable GT tier (strict + relaxed), measured, **ungated**, N = 3505
- **Metric:** absolute error vs labelled 3D box near face, metres

| estimator | valid | MAE | bias | p95 \|e\| | 0–10 | 10–20 | 20–30 | 30–50 | 50+ |
|---|---|---|---|---|---|---|---|---|---|
| `min` | 98% | 1.53 | −1.44 | 8.11 | 0.29 | 0.53 | 1.01 | 1.81 | 9.56 |
| `p10` | 98% | 0.94 | −0.55 | 1.10 | 0.24 | 0.27 | 0.38 | 0.80 | 7.69 |
| `p20` | 98% | 0.86 | −0.28 | 0.98 | 0.34 | 0.36 | 0.40 | 0.66 | 6.31 |
| `median` | 98% | 1.55 | +0.98 | 4.35 | 1.28 | 1.45 | 1.27 | 1.31 | 4.21 |
| **`shrink_p20`** ← selected | **89%** | **0.41** | **+0.04** | **0.75** | **0.24** | **0.27** | **0.33** | **0.39** | 4.75 |
| `cluster_p20` | 95% | 0.83 | −0.32 | 1.20 | 0.36 | 0.47 | 0.51 | 0.84 | 5.43 |

`shrink_p20` wins in every bin to 50 m, and abstains most (89% vs 95–98%) —
the trade bought deliberately. **Every estimator collapses beyond 50 m**, which
Row 1.3 explains and Row 1.6 fixes.

### Row 1.2 — Ground-truth coverage, val split

What fraction of labelled objects can be ground-truthed **at all**.

| bin (m) | labels | strict | relaxed | invalid | usable | median LiDAR pts/box |
|---|---|---|---|---|---|---|
| 0–10 | 1184 | 484 | 0 | 700 | 41% | 1568 |
| 10–20 | 2162 | 1320 | 4 | 838 | 61% | 506 |
| 20–30 | 1877 | 974 | 14 | 889 | 53% | 217 |
| 30–50 | 2381 | 756 | 4 | 1621 | 32% | 82 |
| 50+ | 1101 | 355 | 11 | 735 | 33% | 22 |
| **all** | **8705** | **3889** | **33** | **4783** | **45%** | |

**Coverage is 45% on val, against 64% on dev** — val's city drives are denser and
more occluded. The full chain to a usable ground-truth value:

```
8705 labelled boxes
 → 3922 usable tier (45%)          occlusion/truncation gate, D-010
 → 3505 measured (>= 8 returns)
 → 3315 pass the spread gate       D-013
 = 38.1% of labelled objects receive a ground-truth range
```

**The caveat that travels with every range number in this project:** ground truth
exists for **38%** of labelled objects, and those are the unoccluded, untruncated,
unambiguous ones. Every reported range error is measured on the easy subset and
is a **lower bound** on real-world error.

### Row 1.3 — Beyond 50 m the error is bimodal, not merely larger

Ungated `shrink_p20`, val, 50+ m bin, N = 84:

| statistic | value |
|---|---|
| median \|e\| | **0.12 m** |
| MAE | **4.75 m** |
| p90 \|e\| | 14.44 m |
| p99 \|e\| | 65.99 m |
| \|e\| < 1 m | 83% |
| \|e\| > 5 m | **14%** |

Mostly excellent, occasionally catastrophic — two populations, not one
distribution. **A mean over that mixture describes neither**, which is why
median and failure rate are now reported alongside MAE everywhere.

### Row 1.4 — Convention gap, measured

| quantity | val |
|---|---|
| centroid range − near-face range | mean **+1.70 m** (p05 0.37, p95 2.42) |

Scoring a near-face estimator against centroid truth would manufacture exactly
this as a pure bias and then attribute it to the geometry ([D-003](decisions.md)).

### Row 1.5 — Ego pitch excursion, val split

Sampled every 5th frame. **This supersedes the dev figure and changes the
conclusion.**

| drive | mean | min | max | peak-to-peak |
|---|---|---|---|---|
| 0059 | −0.257° | −0.838° | +0.279° | 1.117° |
| **0084** | **+1.026°** | −0.007° | +2.004° | **2.011°** |
| **0091** | +0.423° | −0.570° | +1.651° | **2.221°** |
| 0056 | +0.425° | −0.105° | +1.021° | 1.126° |
| 0027 | +0.390° | +0.179° | +0.650° | 0.471° |

**Two of five val drives exceed 2° peak-to-peak** — the level the charter states
already costs >10% range error — on ordinary city driving with **no hard
braking**. Dev's 1.05° was not representative.

`0084` additionally carries a **sustained +1.03° mean**: not an excursion but a
standing offset across a whole drive, which would appear as a systematic range
bias rather than as noise. Pitch is no longer a projected concern; it is a
measured one ([D-009](decisions.md)), and Phase 3 cannot defer the decision.

### Row 1.6 — The spread gate ← **the selected ground-truth configuration**

Abstain when the interquartile depth spread of supporting returns exceeds
**1.5 m** — a box whose middle half of returns spans more than that contains two
surfaces. Threshold set at the 95th percentile of the spread distribution for
boxes that are *correct* (p50 0.241, p90 0.965, **p95 1.313 m**), not from a
geometric argument. The gate reads only LiDAR evidence, never the label, so it
transfers unchanged to detector boxes in Phase 3.

| bin (m) | measured | kept | keep% | MAE | med \|e\| | p95 | fail% (\|e\|>5 m) |
|---|---|---|---|---|---|---|---|
| 0–10 | 484 | 447 | 92% | 0.21 | 0.21 | 0.43 | 0.00% |
| 10–20 | 1324 | 1264 | 95% | 0.24 | 0.21 | 0.62 | 0.08% |
| 20–30 | 988 | 951 | 96% | 0.25 | 0.14 | 0.65 | 0.53% |
| 30–50 | 625 | 592 | 95% | 0.27 | 0.15 | 0.86 | 0.34% |
| 50+ | 84 | 61 | 73% | **0.78** | 0.11 | 0.53 | 1.64% |
| **all** | **3505** | **3315** | **95%** | **0.25** | **0.17** | **0.63** | **0.27%** |

Against ungated: MAE 0.41 → **0.25 m**, failure rate 1.06% → **0.27%**, and the
50+ bin goes from MAE 4.75 / p95 33.79 to **MAE 0.78 / p95 0.53**. Cost: 5% of
measured boxes, concentrated at long range (73% kept at 50+).

Threshold sweep that set it (all usable val boxes, N=3505):

| gate | kept | MAE | residual fail% | failures caught |
|---|---|---|---|---|
| none | 100% | 0.41 | 1.06% | — |
| `spread ≤ 1.0` | 90% | 0.24 | 0.16% | 86% |
| **`spread ≤ 1.5`** | **95%** | **0.25** | **0.27%** | **76%** |
| `spread ≤ 2.0` | 96% | 0.28 | 0.57% | 49% |

**This is the ruler:** MAE **0.25 m**, p95 0.63 m, on 38% of labelled objects.

### Superseded — dev-split figures (2026-09-01)

Kept so the trajectory is visible. Dev = `drive_0013`, N=385, one quiet city drive.

| | dev (superseded) | val (current) |
|---|---|---|
| GT coverage | 64% | 45% |
| ruler MAE | 0.13 m | 0.25 m (gated) |
| by bin | 0.05/0.08/0.15/0.18/0.22 | 0.21/0.24/0.25/0.27/0.78 |
| 50+ m sample | **10 boxes** | 84 boxes |
| pitch p2p | 1.05° | up to **2.22°** |

→ superseded by Rows 1.1–1.6. The dev 50+ figure of 0.22 m rested on ten boxes
and was a small-sample artefact; val's 84 boxes exposed the bimodality that
motivated the spread gate. **Re-running on val before quoting the dev number
anywhere was the single most valuable step of the session.**

---

## Phase 2 — Detection baseline

**Model:** YOLOv8n (COCO-pretrained, no fine-tuning), 640 px, batch 1.
**Split:** val, 1450 frames, 8705 labels. **IoU 0.5.** Commit: *(this commit)*

Reported class-agnostic and at a coarse vehicle/VRU split, and no finer: KITTI's
classes are not recoverable from a COCO-trained model without inventing
information ([D-014](decisions.md)).

### Row 2.1 — Average precision @ IoU 0.5

| | AP | labels |
|---|---|---|
| **class-agnostic** ← headline | **0.615** | 8705 |
| vehicle | 0.662 | 7068 |
| VRU | 0.450 | 1637 |

23447 detections scored; **556 discarded** as landing on Tram/Misc/Person-sitting
ignore regions. Without ignore handling those would have counted against
precision for correctly finding real objects KITTI raw does not grade.

**This is a zero-shot COCO model on KITTI**, not a KITTI-trained detector. It is
the honest denominator for later work, not a competitive number.

### Row 2.2 — Recall vs range, conf ≥ 0.25 ← **the row that bounds the project**

| bin (m) | labels | recall | vehicle | VRU |
|---|---|---|---|---|
| 0–10 | 1184 | 86% | 91% | 75% |
| 10–20 | 2162 | 76% | 79% | 69% |
| 20–30 | 1877 | 65% | 72% | 36% |
| 30–50 | 2381 | 46% | 50% | **2%** |
| 50+ | 1101 | 24% | 25% | **0%** |
| **all** | **8705** | **60%** | | |

**VRU recall collapses to 2% at 30–50 m and 0% beyond.** Pedestrians and cyclists
are effectively invisible to this model past 30 m — and they are the
safety-critical class, the one where a missed detection is a person. Any FCW
claim about VRUs is bounded at ~30 m by detection alone.

### Row 2.3 — Recall vs box height: the mechanism

| box height (px) | labels | recall | median range |
|---|---|---|---|
| < 25 | 1400 | 31% | 54.9 m |
| 25–40 | 2207 | 48% | 36.1 m |
| 40–80 | 2906 | 68% | 21.4 m |
| 80+ | 2192 | 81% | 9.6 m |

Range is not the cause; **apparent size is**. Recall tracks pixel height, and
range only matters because it shrinks the object. This is the same `h = f·H/D`
relation that governs the monocular range error — the detector and the estimator
degrade for one shared reason.

### Row 2.4 — Recall vs occlusion

| occlusion | labels | recall |
|---|---|---|
| visible | 4432 | 81% |
| partly | 2423 | 53% |
| fully | 1806 | 21% |
| unset | 44 | 18% |

### Row 2.5 — Latency vs the 103.56 ms budget

Apple M2, MPS, 640 px, batch 1, N=1450, warmup excluded, `torch.mps.synchronize()`
before the clock stops.

| | ms | % of budget |
|---|---|---|
| p50 | 17.12 | 16.5% |
| p95 | 24.75 | 23.9% |
| p99 | 34.01 | 32.8% |
| p99.9 | 247.92 | 239.4% |
| **max** | **552.72** | **533.7%** |

**Five frames of 1450 (0.34%) exceeded the budget**, at 180 / 216 / 235 / 259 /
553 ms. Mean excluding those five is 16.73 ms. They are scattered through the run
(frame indices 32, 871, 1132, 1195, 1399), so they are not a per-drive warmup
artefact — they look like host-level stalls, and on a real system each is a
dropped frame.

Reporting the p99 alone (33% of budget) would hide them, which is exactly why
hard rule 10 requires the tail. **Detection currently uses a third of the budget
at p99; track / range / KF / decide are not yet in this sum.**

### Row 3.8 — Attribution: which estimator's bias is explained

| estimator | source | predicted | observed | verdict |
|---|---|---|---|---|
| size-prior (vehicle) | −4.5% box height | +0.99 m | **+1.01 m** | **explained** |
| size-prior (VRU) | −4.9% box height | +0.75 m | +1.41 m | partly |
| contact-point | box bottom edge | sign flips vs measured | — | **NOT explained** |

Contact-point: the bottom-edge offset needed to explain the observed bias is
−20.7 px at 0–10 m and **+1.5 / +2.2 px** at 20–50 m, against a measured offset
of −4.8 / −1.2 / −0.9 px that never changes sign. A range-dependent term is
acting that is not the detector box ([D-019](decisions.md)).

**Overhang hypothesis refuted.** Vehicles (≈0.8 m overhang) show median bias
**−0.03 m**; VRUs (no overhang) show **+0.38 m** — both groups wrong, and
opposite to the prediction.

### Row 2.7 — Detector box geometry vs the label's own box

Each detector box against **the label box for the same object in the same
frame** — 5254 matched pairs, val, conf 0.25, IoU 0.5.
Harness: `eval/eval_box_quality.py`.

**Method check first.** Label box height against the pinhole prediction
`f_y·H_3d/range` gives **1.000 / 1.009 / 1.011 / 1.011 / 1.009** across the five
bins. The amodal 3D annotation and the projected 2D extent agree to ~1%, so the
implied-height *approach* is sound — it was the *reference* that was wrong.

| bin (m) | N | det/label height | shortfall | bottom-edge offset |
|---|---|---|---|---|
| 0–10 | 1023 | 0.929 | **−7.1%** | −4.8 px |
| 10–20 | 1643 | 0.955 | −4.5% | −2.0 px |
| 20–30 | 1222 | 0.976 | −2.4% | −1.2 px |
| 30–50 | 1105 | 0.959 | −4.1% | −0.9 px |
| 50+ | 261 | 0.995 | −0.5% | −0.1 px |
| **all** | **5254** | **0.955** | **−4.5%** | −1.8 px |

Clean subset (visible, untruncated, N=3120) gives the same picture: ratio 0.956
overall, 0.936 at 0–10 m. Occlusion barely moves it (0.956 clean vs 0.952
occluded), and neither does group (vehicle 0.956, VRU 0.951) — this is a
**general property of the detector**, not an artefact of hard cases.

**How much of the range error does it explain?** Converting the bottom-edge
offset into a contact-point range bias and comparing with Row 3.1:

| bin (m) | predicted from box offset | measured | accounted for |
|---|---|---|---|
| 0–10 | +0.18 m | **+1.05 m** | **17%** |
| 10–20 | +0.39 m | +0.48 m | 81% |
| 20–30 | +0.59 m | +0.10 m | — |
| 30–50 | +1.18 m | +0.91 m | 77% |

**It accounts well for mid-range bias and NOT for the near field**, where the
observed bias is ~6× what the box offset predicts. The near-field degradation
therefore has a further, **currently unidentified** cause. The earlier claim that
box bias explains the U shape is stronger than the evidence supports.

### Row 2.6 — Compound evidence coverage ← **what Phase 3 can actually measure**

A label is usable for evaluating monocular range only if it is **both detected
and groundtruthable**. Computed as a true per-label join, not a product of
marginals — the two are positively correlated (occluded objects fail both), so
the product understates by ~10 points.

| bin (m) | labels | detected | GT-able | **both** | **N usable** |
|---|---|---|---|---|---|
| 0–10 | 1184 | 86% | 38% | 36% | **430** |
| 10–20 | 2162 | 76% | 58% | 53% | **1142** |
| 20–30 | 1877 | 65% | 51% | 42% | **793** |
| 30–50 | 2381 | 46% | 25% | 20% | **475** |
| 50+ | 1101 | 24% | 6% | 3% | **30** |
| **all** | **8705** | **60%** | **38%** | **33%** | **2870** |

**The 50+ m bin has 30 objects.** That is the same order as the ten-box dev
sample that produced a false 0.22 m figure on 2026-09-01. It cannot carry a
Phase 3 number and will not be asked to ([D-015](decisions.md)).

**The project's credible evaluation envelope is 0–50 m**, well-populated to 30 m
(2365 objects) and thin from 30–50 m (475). Beyond 50 m the limit is *evidence
availability*, not estimator accuracy — a distinction that has to be stated
every time the envelope is quoted, because they are not the same claim.

---

## Phase 3 — Monocular range: the error-vs-range curve

**The project's headline artifact.** Detector boxes (YOLOv8n, conf 0.25), graded
by the Phase 1 LiDAR ruler applied to *those same boxes* — no label matching
anywhere. **Split:** val, N = 6122 detections with valid ground truth.
Camera assumes a fixed **1.655 m height and zero pitch** ([D-016](decisions.md)).

### Row 3.1 — Error vs range ← **THE HEADLINE**

**contact-point** `D = f·h_cam/(v_bottom − v_horizon)` — valid on 98%

| bin (m) | N | MAE | MAPE | bias | p95 \|e\| |
|---|---|---|---|---|---|
| 0–10 | 1006 | 1.28 | **22.7%** | +1.05 | 2.86 |
| 10–20 | 2285 | 1.51 | **10.3%** | +0.48 | 3.97 |
| 20–30 | 1635 | 3.03 | 12.2% | +0.10 | 7.28 |
| 30–50 | 984 | 7.11 | 18.7% | +0.91 | 22.27 |
| 50+ | 91 | 13.72 | 24.0% | −5.87 | 30.44 |
| **all** | **6001** | **2.99** | **14.5%** | +0.45 | 9.80 |

**size-prior** `D = f·H_prior/h_box` — valid on 98%

| bin (m) | N | MAE | MAPE | bias | p95 \|e\| |
|---|---|---|---|---|---|
| 0–10 | 1007 | 1.73 | **29.5%** | +1.48 | 5.52 |
| 10–20 | 2287 | 2.15 | 14.9% | +1.68 | 6.94 |
| 20–30 | 1635 | 2.68 | **11.0%** | +1.82 | 8.81 |
| 30–50 | 984 | 5.80 | 15.3% | +3.40 | 15.68 |
| 50+ | 91 | 9.83 | 16.5% | −7.50 | 26.64 |
| **all** | **6004** | **2.94** | **16.4%** | +1.83 | 10.15 |

**The curve is U-shaped, not monotonic.** Error is *worst at the near field*,
best at 10–20 m, then degrades with range. That is the opposite of what the
pinhole sensitivity argument alone predicts, and Row 3.4 explains why.

### Row 3.2 — Credible operating envelope

Best estimator per bin, against a MAPE threshold:

| threshold | 0–10 | 10–20 | 20–30 | 30–50 | 50+ | envelope |
|---|---|---|---|---|---|---|
| ≤ 10% | fail (23%) | fail (10.3%) | fail (11%) | fail (15%) | fail (17%) | **none** |
| ≤ 15% | fail | **PASS** | **PASS** | fail | fail | **10–30 m** |
| ≤ 20% | fail | PASS | PASS | PASS | PASS | 10–50+ m |
| ≤ 25% | PASS | PASS | PASS | PASS | PASS | 0–50+ m |

**No range bin achieves 10% MAPE.** The best figure anywhere is **10.3% at
10–20 m**.

**The credible envelope at 15% MAPE is 10–30 m — and it excludes the near
field.** A collision-warning system that is least accurate at the closest ranges
is a real and uncomfortable result; it is stated rather than smoothed over.

Two limits bound any wider claim: the 50+ m bin rests on **91 detections**
([D-015](decisions.md)), and the whole curve is measured on the ~33% of objects
that are both detected and groundtruthable, so it is a **lower bound**.

### Row 3.3 — Which estimator wins, and the predicted crossover

| bin (m) | N | contact-point | size-prior | winner |
|---|---|---|---|---|
| 0–10 | 1006 | **22.7%** | 29.5% | contact-point |
| 10–20 | 2285 | **10.3%** | 14.9% | contact-point |
| 20–30 | 1635 | 12.2% | **11.0%** | size-prior |
| 30–50 | 984 | 18.7% | **15.3%** | size-prior |
| 50+ | 91 | 24.0% | **16.5%** | size-prior |

**A crossover exists, at ~20 m**, in the predicted direction: contact-point near,
size-prior far.

Predicted from the error budget (exact relation, [D-016](decisions.md)):

| group | prior CV | at typical pitch 0.319° | at p95 pitch 0.715° |
|---|---|---|---|
| vehicle | 20.1% | 49.7 m | 22.2 m |
| VRU | 8.7% | 23.8 m | 10.6 m |

The observed ~20 m sits near the p95-pitch predictions. **This agreement should
not be read as confirming the pitch model** — Row 3.4 shows pitch is not what
drives the error, so the crossover is real but its predicted mechanism is not the
operative one.

### Row 3.4 — Pitch sensitivity: the assumption is NOT the bottleneck

Pitch measured per frame from LiDAR, used only to stratify; the estimator always
assumed 0.0°.

| \|pitch\| | N | measured MAPE | **predicted MAPE at 30 m** |
|---|---|---|---|
| 0.0–0.2° | 2967 | 14.1% | 3.2% |
| 0.2–0.4° | 1537 | 15.6% | 9.5% |
| 0.4–0.6° | 841 | 13.8% | 15.8% |
| 0.6°+ | 656 | 14.7% | 25.3% |

**Measured error is flat across pitch while the prediction spans 8×.** The
flat-ground assumption costs far less than the geometry says it should, because
something larger is masking it.

**This retroactively validates characterising rather than correcting pitch
([D-016](decisions.md)): a pitch correction would have bought almost nothing.**

### Row 3.5 — Detector box height bias · **CORRECTED 2026-09-09**

> **⚠ This row originally reported −12.2% at 0–10 m**, from comparing an implied
> object height (`box_h × range / f_y`) against the pooled class mean of 1.595 m.
> That overstated the detector's error by ~2.6× because it also absorbed the
> class's 20.1% height spread. Superseded by **Row 2.7**, which measures each
> detector box against the label box for the same object. See
> [D-018](decisions.md).

**Corrected figure: −4.5% overall, −7.1% at 0–10 m.** Detector boxes are
systematically shorter than the objects they contain, worst in the near field,
and both estimators inherit it as a range over-estimate.

The error budget predicted detector box error would be the dominant term. It is a
**bias, not jitter**, so it does not average away over frames — but see Row 2.7
for how much of the error curve it actually accounts for, which is less than this
row originally claimed.

### Row 3.6 — Boxes clipped at the image edge

| | val |
|---|---|
| share of detections | **4.2%** |
| MAPE, clipped | **46.4%** |
| MAPE, unclipped | 13.6% |

An object running past the bottom image edge has no observable contact point and
a truncated pixel height, so both estimators are reading the sensor boundary
rather than the object. Both now abstain ([D-017](decisions.md)). The gate was
declared in `configs/camera.yaml` from the start and **not implemented**; adding
it moved the 0–10 m bin from 24.4% to 22.7% MAPE and the overall figure from
15.0% to 14.5%.

### Row 3.7 — Camera geometry, measured from LiDAR

| | val, N=134 frames |
|---|---|
| camera height above road | **1.655 m** (std 0.027, range 1.583–1.745) |
| camera-to-road pitch | mean **+0.150°**, std 0.319° |
| \|pitch\| p95 / max | **0.715° / 0.937°** |

Independently recovers KITTI's documented mounting. **Supersedes the OXTS-derived
pitch figures** (2.01–2.22° p2p, +1.03° sustained), which measured vehicle
attitude in the navigation frame — including road grade — and overstated the real
camera-to-road error by 2–3× ([D-016](decisions.md)).

**Caveat on every pitch figure here:** measured on drives with no hard braking,
so this is a floor on the operational distribution.

---

## Pending — nothing measured yet

These rows are deliberately empty. A value here that is not a measurement is the
failure mode this file exists to prevent.

| Row | Blocks on |
|---|---|
| 4.x — Tracking MOTA / IDF1 / ID-switches, IoU vs SORT | Phase 4 |
| 5.x — Closing-speed error; TTC error at TTC < 3 s | Phase 5 |
| 6.x — Lane departure detection rate / FP rate | Phase 6 |
| 7.x — FCW TPR and **FP per hour** at a named TTC threshold | Phase 7 |
| 8.x — End-to-end latency p50/p95/p99 vs budget | Phase 8 |
