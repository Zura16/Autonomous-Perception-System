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

### Row 0.1 — Sensor and timing constants

| | measured |
|---|---|
| frame interval | **103.56 ms**, σ 0.06 ms → **9.657 Hz** (not the documented 10 Hz) |
| intrinsics, cam2 rectified | fx = fy = 721.5377 px · cx = 609.5593 · cy = 172.8540 |
| horizontal field of view | 81.4° |
| rect0 → cam2 offset | [+0.0598, −0.0004, +0.0027] m |

Hard-coding the documented 10 Hz would put a **3.4% systematic error** into every
velocity and TTC figure, indistinguishable afterwards from estimator bias
([D-006](decisions.md)).

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

### Row 3.9 — Road non-flatness: the far-range term, predicted then checked

Real road height vs the assumed level plane at 1.655 m, from LiDAR
(`eval/eval_road_profile.py`, val, 265 frames). The bias prediction
`h_cam/y_act − 1` has **no free parameters** — nothing is fitted to the observed
error, which is what makes this a test.

| depth (m) | median road y | vs assumed | predicted bias | observed (vehicle) | residual |
|---|---|---|---|---|---|
| 5–8 | 1.668 | +0.013 | −0.05 | **+0.63** | **+0.68** |
| 8–10 | 1.672 | +0.017 | −0.09 | +0.68 | +0.78 |
| 10–13 | 1.676 | +0.021 | −0.15 | +0.53 | +0.68 |
| 13–16 | 1.684 | +0.029 | −0.25 | −0.10 | +0.14 |
| 16–20 | 1.682 | +0.027 | −0.30 | +0.01 | +0.31 |
| 20–25 | 1.699 | +0.044 | −0.59 | −0.50 | **+0.08** |
| 25–30 | 1.710 | +0.055 | −0.89 | −1.40 | −0.51 |
| 30–40 | 1.741 | +0.086 | −1.73 | −2.26 | −0.60 |
| 40–50 | 1.751 | +0.096 | −2.46 | −2.12 | **+0.35** |

**Beyond 20 m the flat-ground assumption is the dominant error term.** The road
drops ~10 cm below the assumed plane by 50 m, and that alone predicts the
observed bias to within 0.1–0.6 m ([D-020](decisions.md)).

**Inside 13 m it explains almost nothing** — a flat **+0.7 m** residual remains,
of which ~0.3 m is vehicle-specific (vehicle-minus-VRU gap) and ~0.4 m is common
to both groups and **still unexplained**.

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

> **Superseded in part by [Row 9.7](#row-97--the-defect-an-ungated-pole-and-what-closing-it-costs).**
> The contact-point figures below are the *pre-gate* values this phase shipped.
> The D-025 abstention gate changed them on val, and the current values are given
> beneath. Both are kept: the pre-gate column is what Phases 3–8 were built and
> reasoned on, and deleting it would make the project's own history unreadable.

**contact-point** `D = f·h_cam/(v_bottom − v_horizon)` — valid on 98% pre-gate,
**95% post-gate**

| bin (m) | N | MAE | MAPE | bias | p95 \|e\| | **MAPE post-gate** |
|---|---|---|---|---|---|---|
| 0–10 | 1006 | 1.28 | **22.7%** | +1.05 | 2.86 | **21.0%** |
| 10–20 | 2285 | 1.51 | **10.3%** | +0.48 | 3.97 | **10.1%** |
| 20–30 | 1635 | 3.03 | 12.2% | +0.10 | 7.28 | **10.7%** |
| 30–50 | 984 | 7.11 | 18.7% | +0.91 | 22.27 | **11.8%** |
| 50+ | 91 | 13.72 | 24.0% | −5.87 | 30.44 | not evaluable¹ |
| **all** | **6001** | **2.99** | **14.5%** | +0.45 | 9.80 | **12.6%** |

¹ Past 50 m the gate caps every contact-point estimate at the envelope bound, so
each one is an underestimate by construction (bias −14.27 m). The estimator is
not evaluable there; that is the honest reading of a declared support limit, not
a result. Size-prior is unaffected by the gate and its table is unchanged.

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

## Phase 4 — Tracking: IoU baseline vs SORT

**Split:** val, 1450 frames, 8705 labelled boxes, 191 ground-truth objects.
Detector runs **once per frame** and all three trackers consume identical
detections (YOLOv8n, conf 0.25), so only the association strategy varies.
Association IoU 0.3, `max_age` 3, metric matching IoU 0.5.
Harness: `eval/eval_tracking.py`.

**Two caveats on every number below.** Labels are KITTI **raw tracklets**, not
the official tracking benchmark ([D-004](decisions.md)); metrics are
**self-implemented** and validated on hand-computed cases, not the official
devkit ([D-021](decisions.md)). These figures are valid for the comparison made
here and are **not comparable to any published leaderboard**.

### Row 4.1 — Headline (hard rule 8)

| tracker | MOTA | MOTP | IDF1 | ID sw | Frag | FP | FN | hyp IDs |
|---|---|---|---|---|---|---|---|---|
| `iou` baseline | 0.268 | 0.768 | 0.488 | 507 | 365 | 2422 | 3440 | 1078 |
| `sort` (canonical) | 0.270 | **0.735** | 0.515 | **354** | 329 | 2493 | 3512 | 1028 |
| **`sort_det`** ← best | **0.284** | **0.769** | **0.522** | 358 | 364 | 2430 | 3448 | 1028 |

`sort_det` is SORT's association with the **raw associated detection** emitted
instead of the Kalman state. The three rows separate two changes that canonical
SORT bundles together.

### Row 4.2 — The ablation: association vs smoothing

| comparison | isolates | MOTA | MOTP | IDF1 | ID switches |
|---|---|---|---|---|---|
| `sort_det` − `iou` | **association** | +0.015 | +0.001 | **+0.035** | **−149 (−29%)** |
| `sort` − `sort_det` | **smoothing** | −0.014 | **−0.033** | −0.007 | −4 |

**Association is worth having: 29% fewer ID switches.** Predicting where a track
should be, and assigning optimally rather than greedily, does what it claims.

**Kalman smoothing of the OUTPUT is not.** It costs MOTP 0.769 → 0.735 and MOTA
0.284 → 0.270, and buys essentially nothing in identity. At 9.657 Hz with this
detector, the raw detection is better localised than the filtered estimate that
smooths it — the filter's generic process and measurement noise are not tuned to
this detector's actual error, and it lags whenever an object accelerates.

**Canonical SORT is therefore not the best configuration on this data.** Keep the
prediction (for association) and discard the smoothing (for output). That
distinction is invisible unless the two are ablated separately, which is why the
third row exists ([D-021](decisions.md)).

### Row 4.3 — ID switches by range

Per ground-truth object, binned by its median range. Phase 5 reads a per-object
box-height history, so a switch splices two objects' histories together and
manufactures a closing speed from the seam.

| bin (m) | GT objects | `iou` | `sort` | `sort_det` |
|---|---|---|---|---|
| 0–10 | 21 | 28 | **20** | **20** |
| 10–20 | 64 | 259 | **143** | 148 |
| 20–30 | 42 | 85 | **65** | **65** |
| 30–50 | 58 | 125 | **114** | **114** |
| 50+ | 6 | **10** | 12 | 11 |
| **all** | **191** | **507** | **354** | **358** |

**Switches concentrate at 10–20 m** — 259 of 507 for the baseline, over just 64
objects (4 per object). That is the crowded mid-range where objects overlap and
occlude each other, and it is squarely inside the 10–30 m operating envelope
Phase 3 established. SORT's association removes 43% of them there.

At 50+ m SORT is marginally *worse* (10 → 11–12), on 6 objects — too few to
mean anything, and reported rather than dropped.

### Row 4.4 — MOTA is detection-bound, not association-bound

Of 8705 ground-truth boxes, roughly **3450 are missed and 2430 are false
positives** in every configuration. Those two terms dominate MOTA and barely move
between trackers, which is exactly consistent with Phase 2's 60% detection recall
(Row 2.2). No association strategy recovers an object the detector never found.

**Consequence:** the ~0.28 MOTA here is a statement about the *detector*, and
IDF1 and ID-switch counts are where the tracker is actually being measured.

### Superseded — dev-split figures

A 60-frame dev run showed the **opposite** conclusion: `iou` ahead of `sort` on
every metric (MOTA 0.603 vs 0.523, IDF1 0.756 vs 0.583). One drive, 151 GT boxes,
five objects. Val reversed it. Third time in this project a small sample has
pointed the wrong way — see also the 50+ m ruler bin ([D-013](decisions.md)).

---

## Phase 5 — Closing speed + TTC

Scale-rate estimator, never range differencing (hard rule 3). State is the
**inverse apparent height `u = 1/h`**, which is proportional to range, so a
constant-velocity Kalman filter over `[u, u̇]` is exact under constant relative
velocity. `TTC = −u/u̇` (focal length and object size cancel);
closing speed `= −D_est/TTC`. Pipeline as shipped: YOLOv8n → `sort_det` →
filter. Harness: `eval/eval_ttc.py`. Config: `configs/motion.yaml`.

**Ground truth:** each track box matched to a tracklet at IoU ≥ 0.5; GT range is
its near-face range, GT closing speed the least-squares slope of that range over
±3 frames against sensor timestamps. Differencing is acceptable for the
*reference* because tracklet poses are smooth annotations, not noisy per-frame
depth — the exact distinction hard rule 3 draws.

### Row 5.1 — Measurement noise, measured rather than assumed

Detector box height vs label box height, val, 5254 matched pairs (robust sd).

| bin (m) | absolute sd | **relative sd** |
|---|---|---|
| 0–10 | 9.88 px | **5.7%** |
| 10–20 | 4.56 px | **5.5%** |
| 20–30 | 3.39 px | **6.9%** |
| 30–50 | 4.03 px | 13.4% |
| 50+ | 2.48 px | 12.1% |

Absolute spread varies 4× with range; relative spread barely moves inside 30 m.
The noise is **multiplicative**, and `σ_u = σ_rel·u` carries it through `u = 1/h`
unchanged. Configured: 6% for boxes ≥ 35 px, 13% below.

### Row 5.2 — Phantom closing: the synthetic model predicts reality

Phantom = a *relatively stationary* object (|GT closing speed| < 0.5 m/s) that the
filter flags as closing. Synthetic = 400 stationary tracks with 6% height noise.
Real = val tracks.

| deadband | phantom, synthetic | **phantom, real val** | detect, real (GT v < −2 m/s) |
|---|---|---|---|
| none | 54.2% | **54.3%** | 96.0% |
| fixed @ 0.05 (TTC > 20 s rejected) | 10.2% | **11.9%** | 92.7% |
| sigma @ 2.0 | 0.2% | **2.6%** | 72.8% |

Real val: 7852 track-frames, 497 relatively stationary, 3421 genuinely closing.

**Without a deadband over half of stationary objects report closing** — hard
rule 4's bug, measured on real tracks. And the measured-noise synthetic model
predicts the real rate to within a point for `none` and `fixed`. sigma's real
phantom rate is 13× its synthetic one: real tracks carry non-Gaussian error
(splices, clipped boxes) that a covariance-based gate reads as signal.

### Row 5.3 — Deadband sweep (synthetic, 6% noise)

| setting | phantom | detect @ 3 m/s | @ 6 m/s | @ 12 m/s |
|---|---|---|---|---|
| fixed 0.05 | 10.2% | 96% | 100% | 100% |
| fixed 0.10 | 1.5% | 68% | 100% | 100% |
| fixed 0.15 | 0.2% | 20% | 100% | 100% |
| fixed 0.20 | 0.0% | 4% | 96% | 100% |
| fixed 0.30 | 0.0% | 0% | **31%** | 100% |
| sigma 1.0 | 4.2% | 88% | 100% | 100% |
| sigma 1.5 | 1.5% | 66% | 100% | 100% |
| sigma 2.0 | 0.2% | 26% | 100% | 100% |
| sigma 3.0 | 0.0% | 4% | 92% | 100% |

**At matched phantom rates the two mechanisms are equivalent** (fixed 0.10 and
sigma 1.5: 1.5% / 68% vs 66%). The deadband trades slow-approach sensitivity for
phantom suppression; 12 m/s is caught everywhere, and only the strictest settings
begin losing 6 m/s approaches.

### Row 5.4 — TTC accuracy where it matters

A deadband decides **whether** a TTC is emitted, never its value. So accuracy is
reported on the common subset every variant flags:

| GT TTC | common N | MAE | median error | rel MAE | p90 \|e\| |
|---|---|---|---|---|---|
| **< 3 s** | 1930 | **0.41 s** | **+0.01 s** | 30% | 1.02 s |
| < 6 s | 2422 | 0.53 s | −0.01 s | 29% | 1.42 s |

**TTC is unbiased** — median error +0.01 s. The −4.5% detector box-height bias
(Row 2.7) cancels exactly, as the algebra predicts, because TTC is a ratio of
`u` to its own rate.

Coverage of imminent threats (GT TTC < 3 s, N = 2150), i.e. how often a TTC is
emitted at all:

| deadband | emitted | **silent** |
|---|---|---|
| none | 2124 | 1.2% |
| fixed @ 0.05 | 2115 | 1.6% |
| sigma @ 2.0 | 1930 | **10.2%** |

**sigma's apparently better headline TTC error was selection.** The 185 frames
fixed flags and sigma withholds have TTC MAE **3.00 s** — sigma rejects exactly
the worst estimates — but they are real threats, GT TTC median **2.15 s** at
~23 m. sigma trades one in ten imminent-threat frames for 9 fewer points of
phantom rate. That trade is not made here; Phase 7 makes it against FP/hour.

### Row 5.5 — Closing-speed error (inherits range error)

`closing speed = −D_est/TTC`, so every Phase 3 range error propagates in.
No deadband, frames where GT and estimate both say closing:

| bin (m) | N | median error | MAE | rel MAE |
|---|---|---|---|---|
| 0–10 | 661 | −0.74 m/s | 1.88 m/s | 28% |
| 10–20 | 1132 | −0.34 m/s | 2.35 m/s | 32% |
| 20–30 | 680 | +0.94 m/s | 4.04 m/s | 51% |
| 30–50 | 621 | +1.38 m/s | 6.72 m/s | 78% |

**Closing speed is roughly 2.5× worse than TTC in relative terms beyond 20 m**,
because TTC is calibration-free and closing speed is not. For a warning system
this favours deciding on TTC and reporting closing speed as context.

---

## Phase 6 — Lane detection + departure metric

**Labelled set:** KITTI road benchmark, 95 `um_lane` frames — human-labelled
ego-lane masks from the same camera rig, each with a per-frame camera-to-road
fit. KITTI raw has no lane labels ([D-023](decisions.md)). Urban, daytime, single
frames; small N.

**Solver:** metric bird's-eye warp (0.05 m grid) → horizontal morphological
top-hat on lightness → sliding windows → quadratic x(z) in metres. **Parameters in
`configs/lanes.yaml` were committed (`27904fe`) before this set was first
evaluated.** Harness: `eval/eval_lane.py`.

**Geometry:** `nominal` = level camera at 1.655 m (Phase 3, measured on raw
drives, what a deployed system has); `oracle` = each frame's shipped road fit.
Ground truth always uses the per-frame fit. Sanity check before any estimator
ran: ground-truth lane width median **3.45 m** (p5 2.47, p95 4.16).

### Row 6.1 — Detection and lateral offset

| | nominal | oracle |
|---|---|---|
| both boundaries detected | **93/95 (97.9%)** | 90/95 (94.7%) |
| lane-centre offset MAE, all detected | 0.26 m | 0.20 m |
| **offset MAE excluding 2 straddling frames** | **0.18 m** (median 0.12, p90 0.39) | 0.16 m (median 0.07, p90 0.40) |
| width error | median **+0.17 m**, MAE 0.34 | median +0.18 m, MAE 0.39 |

**The two straddling frames are identified from ground truth alone** — the camera
centreline lies *outside* the labelled ego lane (`um_000010`, `um_000045`) — not
from estimator error, so reporting without them is not cherry-picking; both
figures are shown. In both, the estimator's nearer boundary *is* the line being
crossed (within 0.34 m) and the departure warning fires correctly. The ~4 m
"offset error" is a **lane-identity disagreement**: once the car is over the
line, which lane is "ego" is a labelling convention, and the offset metric
charges a full lane width for it.

**The width bias was predicted before running.** The solver finds marking
*centres*; the labels mark the lane *edge*. A 0.12 m marking predicts +0.12 m,
a 0.25 m edge line +0.25 m. Measured +0.17 m — and identical under nominal and
oracle geometry, so it is the labelling convention, not the camera model.

### Row 6.2 — Boundary position error by distance ahead

| band (m) | nominal MAE | nominal median | oracle MAE | oracle median |
|---|---|---|---|---|
| 7–12 | 0.29 | +0.08 | 0.25 | −0.01 |
| 12–20 | 0.38 | +0.13 | 0.33 | −0.00 |
| 20–30 | 0.37 | +0.19 | 0.36 | −0.01 |
| 30–40 | 0.43 | **+0.26** | 0.40 | −0.06 |

**The fixed nominal geometry introduces a lateral bias that grows with distance**
(+0.08 → +0.26 m); the per-frame fit removes it. Spread is almost the same under
both, so the nominal camera costs *bias*, and marking detection costs *spread*.

### Row 6.3 — Departure warning, scored per frame

Warning rule: clearance from the vehicle centreline to the nearer boundary, 7 m
ahead, below half-width 0.91 m + margin 0.20 m = 1.11 m.

| ground-truth condition | frames | nominal warns | oracle warns |
|---|---|---|---|
| **vehicle body over a line** (clearance < 0.91 m) | **5** | **3/5** (95% CI 23–88%) | 2/5 (12–77%) |
| grazing the threshold (0.91–1.11 m) | 8 | 2/8 | 1/8 |
| clear of the threshold | 82 | **2 false (2.4%)** | 3 false (3.7%) |

**13 frames meet the warning rule, but 8 of them clear it by 0.01–0.09 m**
(clearances 1.02–1.10 m). With lateral MAE ~0.18 m, whether a threshold-grazing
frame warns is close to a coin flip. So a per-frame hit rate at a hard
threshold mostly measures how close frames sit to that threshold, not whether
departures are detected. The meaningful rows are the first and last: **3 of 5
body-over-line frames caught, 2.4% false warnings on clear frames** — a
confidence interval of 23–88% says N=5 decides very little.

The two missed body-over-line frames (`um_000004`, `um_000044`) both lost one
boundary entirely (Row 6.4), not its position.

### Row 6.4 — Failure set (worst 8 nominal frames, `docs/figures/lanes/`)

| frame | outcome | cause (inspected) |
|---|---|---|
| `um_000010` | offset err 4.04 m, warning correct | straddling the dashed line; estimator's lane is the one the centreline is in |
| `um_000045` | offset err 3.82 m, warning correct | same, beside tram tracks |
| `um_000004` | missed | residential street, right boundary beside parked cars not found |
| `um_000044` | missed | vehicle on the left line; that line has almost no marking response (coverage 0.04) |
| `um_000005` | offset err 1.15 m, false warning | lane widens (5.0 m labelled); a marking inside the lane taken as the right boundary |
| `um_000084` | offset err 0.80 m, width 3.63 → 5.86 m | dashed left boundary (coverage 0.65) passed over for the solid road edge 1.7 m further out |
| `um_000016` | offset err 0.69 m, width 3.23 → 4.58 m | dashed right boundary (coverage 0.50) passed over for the next solid line |
| `um_000043` | offset err 0.77 m, width 3.39 → 4.19 m | fully visible right boundary (coverage 1.00) passed over for a shadowed verge edge further out |

**All three share one design cause** (FM-17): each boundary's base is the column
with the most marking pixels anywhere in a 0.4–2.8 m band, so the *strongest*
stripe wins over the *nearest*. A dashed line puts down about a third of the
pixels of a solid line or a long shadow edge. The skipped boundaries were not
invisible — their coverage (0.50–1.00) is at or above the p10 of all 190
labelled edges (0.54).

---

## Phase 7 — FCW / AEB request, evaluated as a detector

Pipeline as shipped: YOLOv8n → `sort_det` → scale-rate TTC → decision layer
(`aps/fcw.py`). Perception runs once; every decision configuration replays the
same rows (`eval/eval_fcw.py`). Ground-truth threats come from the annotations
alone, so objects the detector never found stay in the denominator.

**Exposure: 1448 frames = 163.4 s = 2.72 minutes = 0.0454 h.**

### Row 7.1 — The headline is that the data cannot support the metric

Annotated samples on val: 8383. In-path (|lateral| < 1.5 m): 1016. In-path **and
closing**: 648, over **21 distinct objects**. Threat frames and events:

| GT TTC below | threat frames | threat events |
|---|---|---|
| 1.5 s | **4** | 1 |
| 2.0 s | **11** | **2** |
| 2.5 s | 14 | 2 |
| 3.0 s | 18 | 5 |

**The minimum TTC anywhere in-path across the whole split is 1.17 s**, and the
closest in-path object is 5.9 m away. Drivers keep gaps; 2.7 minutes of ordinary
city driving contains almost no imminent forward-collision situations.

A true-positive rate over **2 events** is not a rate, and FP/hour from 0.045 h of
exposure carries an interval spanning an order of magnitude. **Hard rule 7's
metric cannot be honestly produced on this data**, and that is the Phase 7
result ([D-024](decisions.md)).

### Row 7.2 — Threat count is governed by the corridor definition, not the data

| corridor half-width | in-path frames | TTC < 2 s frames | events @ 2 s |
|---|---|---|---|
| **1.5 m** (ego lane) | 1016 | **11** | **2** |
| 2.0 m | 1097 | 15 | 2 |
| 2.5 m | 1375 | 65 | 8 |
| 3.5 m | 2540 | 608 | 49 |
| 5.0 m | 4158 | 1206 | 84 |

Widening the corridor manufactures events, but a 3.5–5 m half-corridor spans the
adjacent and oncoming lanes — traffic an FCW must **not** warn about. So a narrow
corridor leaves nothing to measure and a wide one counts non-threats. **Any TPR
quoted here would describe the in-path definition more than the detector.**

### Row 7.3 — Decision-layer behaviour on the events that do exist

Selected on val, therefore optimistic; N is tiny. Exact Poisson intervals.

| TTC thr | deadband | k/n | events | caught | onsets | FP | FP/h | FP/h 95% CI | lead TTC |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 | none | 1/1 | 2 | 1 | 13 | 12 | 264 | 137–462 | 1.70 s |
| 2.0 | none | 2/3 | 2 | 1 | 8 | 7 | 154 | 62–318 | 2.03 s |
| 2.0 | none | 3/3 | 2 | 1 | 3 | **2** | 44 | 5–159 | 1.92 s |
| 2.0 | sigma | 1/1 | 2 | **0** | 7 | 7 | 154 | 62–318 | — |
| 3.0 | none | 1/1 | 5 | 1 | 20 | 17 | 375 | 218–600 | 1.70 s |
| 3.0 | none | 3/3 | 5 | 2 | 8 | 5 | 110 | 36–257 | 2.41 s |

What is legible despite the sample size:

- **Persistence is the strongest lever.** At 2.0 s, 3-of-3 cuts onsets 13 → 3 and
  false positives 12 → 2 while keeping the same event caught, at a cost of
  0.2 s of warning latency.
- **The Phase 5 deadband barely matters here, and `sigma` can lose the event**
  (0 of 2 caught at 2.0 s, 1-of-1), consistent with D-022: it withholds the
  hardest frames, which are the threats.
- **Lead time is 1.7–2.5 s** where a warning fires.

### Row 7.4 — What the false alarms are

At 2.0 s / no deadband / 1-of-1: 36 candidate frames, split evenly —

| source | frames |
|---|---|
| track matched an annotated object (wrong TTC or wrong lateral) | 18 |
| **track matched no annotated object at all** (ghost) | 18 |

Half the false alarms are not decision errors at all: they are tracks on things
that are not annotated objects. Precision at the decision layer is bounded by
detector precision, exactly as MOTA was bounded by detector recall (Row 4.4).

---

# Phase 9 — the held-out test split

Drives `0009` (city, 447 frames) and `0015` (road, 297 frames), 744 frames,
untouched until 2026-09-16. Every parameter frozen on dev/val beforehand.

### Row 9.1 — The ground-truth ruler transfers

Same estimator (`shrink_p20` + 1.5 m spread gate), same thresholds, run on test
labels for the first time. N = 4442 labelled boxes, 1456 measurable.

| bin (m) | kept | MAE | med \|e\| | p95 | fail% (\|e\|>5 m) |
|---|---|---|---|---|---|
| 0–10 | 132 | 0.24 | 0.18 | 0.38 | 0.00% |
| 10–20 | 339 | 0.18 | 0.13 | 0.45 | 0.00% |
| 20–30 | 418 | 0.21 | 0.15 | 0.55 | 0.00% |
| 30–50 | 416 | 0.40 | 0.22 | 0.66 | 0.48% |
| 50+ | 61 | 0.41 | 0.32 | 0.57 | 0.00% |
| **all** | **1366** | **0.27** | **0.17** | **0.59** | **0.15%** |

Against val's **0.25 m / 0.27% fail**. The ruler carries its own error onto a
split it has never seen without degrading, so every comparison below is against
an instrument known to be sound. Coverage: **38%** of labelled boxes reach a
usable tier (val 45%), and **1366 of 4442 = 30.8%** survive to a ground-truth
range (val 38.1%).

4 frames of `drive_0009` carry no velodyne scan (447 images, 443 scans) and are
skipped and reported, never silently dropped.

### Row 9.2 — Detection on held-out data

| metric | val | test |
|---|---|---|
| AP@0.5 class-agnostic | 0.615 | **0.737** |
| AP vehicle | 0.662 | 0.774 |
| AP VRU | 0.450 | **0.127** (250 labels) |
| recall by bin (%) | 86 / 76 / 65 / 46 / 24 | **93 / 89 / 74 / 61 / 51** |

The test drives are *easier* for the detector. This matters when reading Row
9.3: the range estimator was handed **more** boxes and better ones, and still
did worse.

### Row 9.3 — THE HELD-OUT RESULT, AS FROZEN ← the headline

Pipeline exactly as declared at the end of Phase 3. N = 3214 ground-truthable
detections. Best estimator per bin:

| bin (m) | val (declared on) | test (held out) | verdict |
|---|---|---|---|
| 0–10 | 22.7% | 24.1% | both fail |
| **10–20** | **10.3%** | **17.2%** | **envelope FAILS** |
| **20–30** | **11.0%** | **15.6%** | **envelope FAILS** |
| 30–50 | 15.3% | 14.4% | improved |
| 50+ | 16.5% | 14.3% | improved |

**The credible operating envelope declared in Phase 3 — 10–30 m at ≤15% MAPE —
does not hold on held-out data.** Both bins it rested on miss the threshold,
while the far bins improve. No contiguous-from-zero envelope exists at any of
10 / 15 / 20%.

### Row 9.4 — Where the generalisation gap actually is

The same data by median and p90 rather than mean, contact-point:

| bin (m) | MAPE val→test | **median** APE val→test | **p90** APE val→test |
|---|---|---|---|
| 0–10 | 22.7 → 24.1 | 10.9 → 11.6 | 62.5 → 50.2 |
| 10–20 | 10.3 → 18.4 | **7.5 → 10.3** | **19.5 → 35.3** |
| 20–30 | 12.2 → 20.3 | **7.7 → 8.8** | **22.5 → 30.8** |
| 30–50 | 18.7 → 20.6 | 12.0 → 10.0 | 28.7 → 35.9 |

**The central behaviour transfers; the tail does not.** At 20–30 m the median
moves 1.1 points while the mean moves 8.1. Reporting MAPE alone described a
collapse that did not happen to the typical object, and hid one that did happen
to the worst 10%.

### Row 9.5 — The tail is 10 detections, not a distribution

`drive_0009` carries the entire degradation; `drive_0015` is the best sequence
in the project. Contact-point, 10–30 m:

| drive | split | N | MAPE | median | >30% APE | bias |
|---|---|---|---|---|---|---|
| `0015` road | test | 444 | **7.7%** | 4.7% | 3.2% | +0.20 |
| `0091` city | val | 1143 | 8.5% | 5.4% | 2.6% | +0.80 |
| `0059` city | val | 1497 | 12.2% | 9.8% | 3.4% | +0.39 |
| `0084` city | val | 858 | 12.4% | 9.4% | 4.1% | −0.61 |
| `0056` city | val | 406 | 11.7% | 4.3% | 7.1% | +0.80 |
| **`0009` city** | **test** | **1287** | **23.3%** | **11.7%** | **14.1%** | **+1.70** |

Within `drive_0009`, by 50-frame block:

| frames | N | MAPE | >30% APE | bias |
|---|---|---|---|---|
| 50–99 | 199 | **55.5%** | 23.6% | **+9.08** |
| 150–199 | 168 | 26.8% | 30.4% | +0.16 |
| 300–349 | 162 | **6.6%** | 1.9% | +0.25 |

In the worst block the **median error is −1.54 m** and **10 detections of 199
carry 81% of the summed bias**, the worst estimating 253 m for an object at
25.8 m.

### Row 9.6 — Three hypotheses killed by measurement

| hypothesis | test | verdict |
|---|---|---|
| the ruler degraded | test ruler MAE 0.27 m vs val 0.25 | **no** (Row 9.1) |
| occlusion: a nearer object hides the lower body | nearer object present for 87.9% of outliers vs **85.7%** of the rest | **no discrimination** |
| road non-flatness (D-020 mechanism) | worst block observed bias **+9.08 m**, road profile predicts **−0.65 m** | **wrong sign, 10× short** |
| box bottom near the horizon row | outlier denominators ~4.7 px vs 47.7 px expected | **confirmed** |

The refined occlusion form — a nearer detection covering >50% of *this* box's
bottom edge — does show +8.29 m bias, but reaches only 8 of 199 boxes in the
worst block. Real, secondary.

### Row 9.7 — The defect: an ungated pole, and what closing it costs

`D = f·h/(v_bottom − v_horizon)`. The only guard was
`min_pixels_below_horizon: 3`, a division-by-zero check that permits a **398 m**
answer and a **33% range error per pixel** of box-edge error. Fixed by abstaining
beyond `max_range_m = 50 m`, the envelope D-015 already declared ([D-025](decisions.md#d-025)).

Boxes inside the pole, by drive:

| drive | split | N | inside pole | their med \|e\| | worst estimate |
|---|---|---|---|---|---|
| `0091` | val | 1545 | 0 (0.0%) | — | — |
| `0059` | val | 2249 | 58 (0.9%) | 23.9 m | 99 m |
| `0084` | val | 1674 | 93 (1.5%) | 9.9 m | 88 m |
| `0015` | test | 883 | 43 (1.3%) | 8.2 m | 109 m |
| `0009` | test | 2164 | **118 (3.7%)** | 9.9 m | **369 m** |

Post-fix, best estimator per bin:

| bin (m) | val pre | val post | test pre | test post |
|---|---|---|---|---|
| 0–10 | 22.7% | 21.0% | 24.1% | 22.5% |
| 10–20 | 10.3% | **10.1%** | 17.2% | **15.8%** |
| 20–30 | 11.0% | **10.7%** | 15.6% | **12.2%** |
| 30–50 | 15.3% | **11.8%** | 14.4% | **12.7%** |

**Medians are unchanged by the fix** (val 10–20 m: 7.5% → 7.5%; test 20–30 m:
8.8% → 8.6%). It deletes fabricated values; it does not improve the estimator.

Cost, paid in abstentions and reported as required: contact-point validity falls
to **95% (val) / 90% (test)**; in the 0–10 m bin it is 71% on test, and
contact-point is **no longer evaluable at all in the 50+ bin** — capped at 50 m,
every estimate there is an underestimate by construction (bias −14.27 m).

**The fix does not rescue the claim.** Test 10–20 m is 15.8%, still past 15%.
The honest post-fix envelope is **20–50 m at ≤13% on both splits**, with 10–20 m
marginal (10.1% val vs 15.8% test) and the near field failing on both.

⚠ **Provenance.** Rows 9.3–9.6 are clean held-out measurements. Row 9.7's
post-fix columns are **not** — the defect was found by inspecting test, so those
numbers were taken after the split was seen. The gate's *value* comes from
D-015, which predates it, and it repairs val too; but the sequencing stands and
the as-frozen result above remains the headline.

---

## Pending — nothing measured yet

These rows are deliberately empty. A value here that is not a measurement is the
failure mode this file exists to prevent.

| Row | Blocks on |
|---|---|
| 8.x — End-to-end latency p50/p95/p99 vs budget | Phase 8 |
