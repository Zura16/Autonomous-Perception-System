# History

Day-by-day journal of what landed. The narrative; `benchmarks.md` holds the
numbers and `decisions.md` holds the reasoning.

---

## 2026-09-01 — Phases 0 and 1

**Context.** The repository held a v1 demo: ~1.7k lines of YOLO detection, IoU
tracking, geometric depth, a lane solver, and a Flask HUD showing distances, TTC,
and a BRAKE indicator — with no ground truth anywhere in it. Two charter drafts
were sitting untracked in the working directory. The session began by adopting
the APS charter and rebuilding underneath it.

**Decided up front** (three questions, answered by the author):
Claude implements and the author reviews, overriding the charter's coach mode
([D-001](decisions.md)) · KITTI **raw** drives rather than the 50 GB tracking
benchmark, against 17 GB of free disk ([D-004](decisions.md)) · v1 archived to
`legacy/` rather than deleted ([D-008](decisions.md)).

### Phase 0 — repo, dataset, one frame

- Archived v1 to `legacy/`, excluded from lint and tests. Deleted
  `commit_state.json` and stopped tracking the fake daily-commit scheduler.
- Scaffolding: `pyproject.toml` (ruff + black, line length 100), `.gitignore`
  that keeps datasets and weights out of git, editable install.
- `tools/fetch_kitti.py` — resumable (HTTP Range), SHA-256 recorded to
  `MANIFEST.json`, unused sensor streams pruned on unpack. Hit
  `CERTIFICATE_VERIFY_FAILED` on this python.org build; fixed with certifi rather
  than by disabling verification.
- `aps/kitti/` — calibration parsing and the velo→rect0→image chain, lazy drive
  loading, OXTS ego state, tracklet XML parsing.
- `tools/render_frame.py` produced the standing calibration check
  (`docs/figures/phase0_projection_check.png`): LiDAR depth-coloured, 3D
  wireframes snug on the vehicles.

**Findings banked immediately:**
frame rate is **9.657 Hz**, not the documented 10 ([D-006](decisions.md)) ·
near-face vs centroid range differ by **2.03 m** ([D-003](decisions.md)) ·
ego pitch swings **1.05°** on a gentle drive ([D-009](decisions.md)) · only 35%
of forward LiDAR returns land in the image.

### Split revision, before spending bandwidth

The first val split (`0027` + `0056`) was chosen on scene labels and frame counts:
**809 boxes, 91% from one drive**, with `0027` contributing 69 boxes over 61 of
188 frames. Too thin to bin range error five ways.

Rather than download drives and look afterwards, tracklet zips (~1.5 MB against
~1.5 GB per drive) were fetched for nine candidates and audited for label density
and range distribution. `0059` surfaced with 3810 boxes spread evenly across all
five bins, and `0091` with the only real VRU coverage in the project (42
pedestrians, 8 cyclists). Val became `0059` + `0084` + `0091` + `0056` + `0027`,
~8.7k boxes. The procedure is committed as `tools/audit_labels.py --probe`
([D-005](decisions.md)).

### Phase 1 — the ruler

Six LiDAR-in-box estimators implemented and scored against the independent human
3D box annotations, per range bin.

**First result was alarming and correct to be alarmed by:** MAE 3.19 m with p95
error of 25 m, from a sensor accurate to centimetres. Stratifying by the
annotator's occlusion flag explained all of it — 0.10 m on visible objects,
**11.86 m on fully-occluded ones**, where the box contains the occluder and the
estimator confidently measures the wrong car. Worst case: a car labelled at 57.6 m
read as 26.6 m, five frames running.

That produced the validity tiers ([D-010](decisions.md)) and hard rule 14. It also
produced the caveat that now travels with every range number in the project:
**ground truth exists for 64% of labelled objects, so every range figure is a
lower bound.**

**A change I made and the evidence reversed.** An adaptive shrink fallback lifted
estimator validity from 86% to 100%. Measured, the recovered boxes carried **12×
the error** (1.56 m vs 0.13 m), and four of them degraded the 20–30 m bin by 4.5×.
Reverted, banked as [D-011](decisions.md), and locked down by a regression test.
Coverage is not a quality metric.

**Final ruler:** `shrink_p20`, MAE **0.13 m**, bias +0.09, p95 0.37 —
0.05 / 0.08 / 0.15 / 0.18 / 0.22 m across the five bins. Flat with range, best in
every bin rather than on average, and ~20× tighter than the monocular error
Phase 3 expects to find.

Also caught while writing tests: the `cam_offset` docstring described the
negation of what the function returns, and the synthetic test fixture had
`P_rect_02[0,3]` negative where KITTI's is positive. The real-data path was
always correct; the misunderstanding would have surfaced later as a sign error.

**Landed:** 38 tests, ruff + black clean, `docs/` spine written
(benchmarks, decisions, glossary, error-budget, claims, failure-modes, tradeoffs).

**Next:** Phase 2 — detection baseline. YOLOv8 mAP on val, latency on M2/MPS.
Note for that phase: no CUDA on this machine, so the charter's TensorRT ladder is
not reproducible here and must not be claimed.

---

## 2026-09-02 — Phase 1 re-verified on val; three dev figures overturned

**The session's one instruction to itself** was written into *Current status* the
day before: *re-run the ruler on val before quoting the dev number anywhere.*
That turned out to be the most valuable thing done in either session.

### What val changed

| | dev (2026-09-01) | val (now) |
|---|---|---|
| N | 385 | **8705** |
| GT coverage (usable tier) | 64% | **45%** |
| Ruler MAE | 0.13 m | **0.25 m** (gated) |
| 50+ m bin | 0.22 m, N=10 | bimodal, N=84 |
| Pitch peak-to-peak | 1.05° | up to **2.22°** |

All three headline dev figures were optimistic, and the far-range one was an
artefact of a ten-box sample.

### The bimodality, and the gate

Ungated on val beyond 50 m: **median |error| 0.12 m but MAE 4.75 m** — 83% of
boxes under 1 m, 14% over 5 m, worst 66 m. Two populations, not one distribution.
Cause: a 2D box containing the object *and* something in front of it, without the
annotator flagging occlusion.

Discriminators were measured rather than guessed. Point count — the intuitive
choice — turned out **backwards** (failures had *fewer* points, 14 vs 17 median).
Interquartile depth spread separated the populations **47×** (0.248 m correct vs
11.65 m failing). Threshold set at the p95 of the spread distribution for boxes
that are *correct* (1.313 m), rounded to **1.5 m**.

Result: MAE 0.41 → **0.25 m**, failure rate 1.06% → **0.27%**, and the 50+ bin
from MAE 4.75 / p95 33.79 to **0.78 / 0.53**. Cost: 5% of measurements, 27% of
them at long range ([D-013](decisions.md)).

The gate reads only LiDAR evidence, never the label — so it transfers to detector
boxes in Phase 3, and it avoids the circularity of gating on agreement with the
label and then reporting that subset's agreement as accuracy. Characterisation
therefore runs **ungated**, with the gate applied as a reported layer.

**A unit test disproved my own justification.** The first draft argued 1.5 m from
vehicle geometry — that an obliquely-viewed car legitimately spans that much
depth. A test with a uniform 4 m extent (IQR 2.0 m) failed, showing the geometric
story was wrong; the honest reason is the empirical percentile. The comment in
`groundtruth.py` now says so, and records that the geometric argument was post-hoc.

### Pitch stopped being hypothetical

Val drives reach **2.011°** and **2.221°** peak-to-peak — at or above the level
the charter states already costs >10% range error — on ordinary city driving with
**no hard braking**. `0084` also carries a **sustained +1.026° mean**: a standing
offset across a whole drive, i.e. a systematic range bias no temporal filter
removes, and roughly two orders of magnitude larger than the 0.25 m GT budget.

Phase 3 can no longer defer the pitch decision ([D-009](decisions.md)).

### Also landed

- Latency budget fixed at **103.56 ms/frame**; TensorRT ladder dropped as
  inherited scope from the superseded FuseTrack charter ([D-012](decisions.md)).
- YOLOv8n smoke-tested on a KITTI frame: **12.4 ms MPS / 28.0 ms CPU**, already
  inside budget before any optimisation — which is itself the argument for not
  optimising.
- 41 tests, ruff + black clean. Every doc carrying a Phase 1 number updated;
  dev figures retained in `benchmarks.md` marked superseded.

### Phase 2 — detection baseline, same day

The class-mapping question was settled first: **class-agnostic AP as the
headline, vehicle/VRU as the only finer split, no per-class mAP**
([D-014](decisions.md)). KITTI `Van` is COCO `car` or `truck` depending on the
vehicle; a `Cyclist` produces both a `person` and a `bicycle` box. A per-class
number built on that would measure the mapping, not the detector. Class-agnostic
NMS follows from the same reasoning, and ignore regions (Tram/Misc/Person-sitting,
matched on intersection-over-detection-area rather than IoU) discarded **556**
detections that would otherwise have been false positives for finding real
objects KITTI raw does not grade.

**Results, YOLOv8n zero-shot on val:** class-agnostic AP@0.5 **0.615** (vehicle
0.662, VRU 0.450); latency p50 17.1 / p95 24.8 / **p99 34.0 ms** against the
103.56 ms budget.

**Recall vs range turned out to matter far more than AP.** 86 / 76 / 65 / 46 /
24% across the bins — and **VRU recall collapses to 2% at 30–50 m and 0%
beyond**. The safety-critical class is bounded at ~30 m by detection alone. Row
2.3 gives the mechanism: recall tracks *pixel height*, not range (31% below
25 px, 81% above 80 px). The detector and the monocular estimator degrade for the
same `h = f·H/D` reason.

**The latency tail was worth reporting.** p99 is a comfortable 33% of budget, but
**five frames of 1450 exceeded it** at 180–553 ms, scattered through the run so
not a warmup artefact. The p99 hides them entirely; each is a dropped frame on a
real system.

### The structural finding

Phase 3 needs labels that are **both detected and groundtruthable**. Computed as
a true per-label join — not a product of marginals, which understated it by ten
points because the two are positively correlated:

| bin | 0–10 | 10–20 | 20–30 | 30–50 | 50+ |
|---|---|---|---|---|---|
| usable N | 430 | 1142 | 793 | 475 | **30** |

**Thirty objects beyond 50 m.** The same order as the ten-box dev sample that had
produced a false 0.22 m figure that morning. The credible evaluation envelope is
therefore **0–50 m**, and the limit is *evidence availability* rather than
estimator accuracy — two different claims, only one of them supported
([D-015](decisions.md)).

**Landed:** 68 tests (27 new, covering IoU/IoA, greedy association, ignore
regions, and AP bookkeeping — including that a duplicate detection is a false
positive and that AP with zero labels is NaN rather than 0). ruff + black clean.
`artifacts/detections_val.npz` caches detector boxes paired with matched GT
range, which Phase 3 consumes directly.

**Next:** Phase 3 — monocular range and the error-vs-range curve, the project's
headline artifact. **The pitch decision blocks it** and cannot be deferred: val
pitch reaches 2.01°/2.22° peak-to-peak with a sustained +1.03° offset on `0084`,
which is ~2 orders of magnitude larger than the 0.25 m ground-truth budget.

---

## 2026-09-02 (cont.) — Phase 3: the headline artifact

The pitch decision was settled first, and the answer changed on measurement.

### Pitch: the concern was right, the number was wrong

The choice was **characterise the flat-ground assumption, don't correct it** — a
horizon-row estimator would need validating against a reference, and OXTS is not
that reference.

But rather than sweep hypothetical pitch values, a road plane fitted to LiDAR
(RANSAC + least-squares refit on the consensus set) measures the *actual*
camera-to-road angle. It recovered KITTI's documented mounting independently
(velodyne 1.713 m against a documented 1.73 m; camera 1.655 m) — and then
overturned the previous day's alarm:

| | OXTS (wrong quantity) | LiDAR road plane |
|---|---|---|
| spread | 2.01–2.22° p2p | std **0.319°** |
| sustained offset | +1.026° | mean **+0.150°** |
| worst | — | \|p95\| **0.715°** |

OXTS reports navigation-frame vehicle attitude, which includes **road grade** —
driving uphill registers as pitch while the camera stays aligned with the road.
That confound was named in D-009 and then quoted past anyway. Corrected in
[D-016](decisions.md).

### Two errors caught by unit tests, both flattering the method

1. **The linearised pitch formula used as if exact.** `δD/D ≈ tan(θ)·D/h` is a
   first-order expansion; the exact relation `x/(1−x)` is superlinear and
   diverges where the contact point reaches the horizon. At 50 m the
   linearisation says 37.7% where the truth is **60.5%** — understated by a
   third. It also moved the predicted estimator crossover from 59.7 m to 49.8 m,
   i.e. from outside the evidence envelope to inside it.
2. **A config gate declared and never implemented.** `min_rows_from_bottom: 2`
   sat in `configs/camera.yaml` and was loaded by `CameraModel`, but no code read
   it. Boxes clipped at the image edge are 4.2% of detections and carry **46.4%
   MAPE against 13.6%** ([D-017](decisions.md)). A config value with no reader is
   not a conservative default; it is a silent lie about what the code does.

### The curve

**val, N=6122.** contact-point MAPE **22.7 / 10.3 / 12.2 / 18.7 / 24.0%**;
size-prior **29.5 / 14.9 / 11.0 / 15.3 / 16.5%**. Credible envelope **10–30 m at
≤15% MAPE**; no bin reaches 10%.

**It is U-shaped.** Worst in the near field, not at range — the opposite of what
the pinhole argument alone predicts, and the most uncomfortable result the
project has produced.

### Why: the detector, not the geometry

Implied object height (`box_h × range / f_y`, which must be range-independent)
is **1.400 m at 0–10 m against a true 1.595 m**, 5–8% low elsewhere. Detector
boxes are systematically undersized, worst near, and both estimators inherit it
as a range over-estimate.

> **⚠ Corrected 2026-09-09 — this magnitude was overstated 2.6×.** It compared
> implied heights against a *class mean*, absorbing the class's 20.1% spread.
> Measured per object it is −4.5% overall / −7.1% near ([D-018](decisions.md)).

The error budget predicted detector box error would dominate. It does — but as a
**bias, not jitter**, so nothing that averages over frames will remove it.

This also explains why the pitch stratification is **flat** (14.1 / 15.6 / 13.8 /
14.7%) where the geometry predicts an 8× span: box bias swamps the pitch term at
the magnitudes present. **Correcting pitch would have bought almost nothing** —
which retroactively validates the decision to characterise it. Pitch becomes the
binding constraint only once the box bias is addressed.

**Landed:** 95 tests, ruff + black clean. `aps/geometry.py`, `aps/groundplane.py`,
`eval/eval_range.py`, `configs/camera.yaml`.

**Next:** Phase 4 — tracking. IoU baseline then SORT, with MOTA / IDF1 /
ID-switches on identical data.

---

## 2026-09-09 — corrections, the road profile, and Phase 4

Resumed after a week. Session began by re-checking a Phase 3 claim and ended
three corrections and one phase later.

### Correction 1 — the detector box bias was overstated 2.6×

Phase 3 blamed the range error on detector boxes being ~12% short, measured by
comparing an *implied* object height against the **pooled class mean**. That
conflates the detector's error with the class's 20.1% height spread. Measured per
object against its own label box (5254 pairs): **−4.5% overall, −7.1% inside
10 m** ([D-018](decisions.md)). The method itself checked out — label box vs
pinhole prediction is 1.000–1.011 — so the approach was sound and the *reference*
was wrong. `claims.md` C-11 had shipped the wrong figure and was rewritten.

The weaker consequence mattered more than the magnitude: box bias accounts for
~80% of the mid-range bias but only **~17% of the near-field bias**, so "box bias
explains the U shape" was never supported.

### Correction 2 — the overhang hypothesis, and a confounded test

Proposed that the near-field bias was definitional: `range_m` is the bumper while
the contact point is the tyre patch, ~0.8 m back. The decisive test is that
pedestrians have no overhang. Pooled medians gave vehicle **−0.03 m** and VRU
**+0.38 m** — wrong groups, wrong direction ([D-019](decisions.md)).

**That pooled test was itself confounded**: vehicle bias changes *sign* with
range, so pooling averages a near over-estimate against a far under-estimate.
Redone within range bins, overhang stays rejected — the veh−VRU gap is +0.94 to
+0.17 near but −0.65 to −1.11 beyond 13 m, neither constant nor positive
throughout. The conclusion survived a flawed test, which is luck, not method.

### The far-range cause: road non-flatness

Predicted with **no free parameters** — a contact point at depth `D` on ground at
height `y_act` gives `D_est = D·h_cam/y_act` — then checked. The road falls ~10 cm
below the assumed 1.655 m plane by 50 m, and that alone predicts the observed
bias beyond 20 m to 0.1–0.6 m ([D-020](decisions.md)):

| depth | predicted | observed |
|---|---|---|
| 20–25 m | −0.59 m | −0.50 m |
| 30–40 m | −1.73 m | −2.26 m |
| 40–50 m | −2.46 m | −2.12 m |

**The flat-ground assumption is the dominant contact-point error at range**, and
"ground-plane non-flatness — unquantified" in the error budget became a measured
term. Inside 13 m it explains almost nothing; a flat **+0.7 m** residual remains
and is still open.

Three errors of *comparison* rather than computation now (wrong pitch quantity,
class mean, pooled sign change). Standing check added: **before believing a
summary statistic, ask what it is pooled over.**

### Phase 4 — tracking

IoU baseline, canonical SORT, and SORT-with-raw-output, all fed **identical
detections**. Val, 1450 frames:

| tracker | MOTA | MOTP | IDF1 | ID sw |
|---|---|---|---|---|
| `iou` | 0.268 | 0.768 | 0.488 | 507 |
| `sort` | 0.270 | 0.735 | 0.515 | 354 |
| `sort_det` | **0.284** | **0.769** | **0.522** | 358 |

The third row is the point. Canonical SORT bundles association and output
smoothing; separated, **association buys −29% ID switches** while **smoothing
costs MOTP 0.769 → 0.735** for nothing. Shipped configuration keeps the
prediction and discards the smoothing ([D-021](decisions.md)).

MOTA barely moves between trackers because ~3450 of 8705 objects are never
detected — MOTA is measuring Phase 2's 60% recall, not the association. ID
switches concentrate at **10–20 m** (259 of 507 over 64 objects), inside the
operating envelope, which matters because Phase 5 reads per-object box-height
histories.

**A test premise that was wrong taught something real:** a newborn track has zero
velocity, so an object moving further than its own width per frame never
associates once and SORT's filter never bootstraps. SORT cannot rescue what it
never caught. Locked in as a test.

**Dev pointed the wrong way for the third time** — a 60-frame dev run had `iou`
ahead of `sort` on every metric. Dev is for wiring, never conclusions.

**Landed:** 127 tests, ruff + black clean. `aps/tracking.py`, `aps/motmetrics.py`,
`eval/eval_tracking.py`, `eval/eval_box_quality.py`, `eval/eval_road_profile.py`.

**Next:** Phase 5 — closing speed and TTC. Scale-rate estimator with a KF over
the image-plane state, deadband, validated on synthetic constant-velocity
sequences before real data.

---

## 2026-09-14 — Phase 5: closing speed and TTC

**Design.** State is `u = 1/h`. Because `u ∝ D`, it is exactly linear under
constant relative velocity, so a constant-velocity Kalman filter is the exact
model rather than an approximation. `TTC = −u/u̇` — calibration- and prior-free.

**Measurement noise measured first.** The Phase 4 retrospective (D-021) had
flagged SORT's unfitted priors. So before writing this filter, detector box-height
error was measured on 5254 matched pairs: absolute spread swings 4× with range
while relative spread stays at 5.5–6.9% inside 30 m. Multiplicative noise, which
`u = 1/h` carries through unchanged. That number, not a guess, set the filter.

**Results, val:** TTC at GT TTC < 3 s has MAE **0.41 s** and median error
**+0.01 s**. Unbiased — the −4.5% box bias cancels in the ratio, as predicted.
Closing speed inherits range error (28% → 78% relative MAE with range).

**The noise model predicted reality.** A synthetic stationary-object test built on
the measured 6% gave 54.2% phantom closing with no deadband; real val tracks gave
**54.3%**. With a fixed deadband, 10.2% vs 11.9%.

**Three things corrected before commit:**

1. **"The adaptive deadband adapts to track quality"** — true, and useless at
   matched phantom rates, where fixed and sigma are equivalent (1.5% phantom,
   68% vs 66% detection).
2. **"6 and 12 m/s approaches are detected 100% at every setting"** — false at
   the strict end of my own sweep (fixed@0.30 catches 31% of 6 m/s approaches).
3. **A test that asserted nothing.** The matched-rate equivalence test built two
   estimators and never ran them, and passed. Rewritten to measure them.

**The selection trap, caught.** sigma@2.0 posted TTC MAE 0.41 s against fixed's
0.63 s. On the common subset every variant flags, all three are identical — a
deadband only decides whether a TTC is emitted. sigma was withholding 185 frames
with TTC MAE 3.00 s, and those were genuine threats (GT TTC median 2.15 s). It is
silent on 10.2% of imminent-threat frames; fixed on 1.6%. That is the Phase 7 trade.

This is the fourth comparison error the project has caught, and the first caught
*before* publishing: compare on a common subset before crediting a filter.

**Landed:** `aps/motion.py`, `configs/motion.yaml`, `eval/eval_ttc.py`, 22 new
tests (149 total). ruff + black clean.

**Next:** Phase 6 (lanes) per the milestone order, or Phase 7 (FCW), which now has
a validated TTC to consume.

---

## 2026-09-15 — Phase 6: lane detection and the departure metric

**Labelled data first.** KITTI raw has no lane labels. The KITTI road benchmark
(449 MB) has 95 `um_lane` frames with human ego-lane masks and a per-frame
camera-to-road fit. Projected ground-truth lane width came out at a median
3.45 m before any estimator existed, which validated the projection.

**Parameters frozen before evaluation.** The solver and `configs/lanes.yaml` were
committed (`27904fe`) before the labelled set was first scored. With no held-out
lane set, that commit is what makes "not tuned on the evaluation data" auditable.

**Results:** detection 97.9%, offset MAE 0.18 m excluding two straddling frames
(0.26 m with them), width bias +0.17 m — predicted beforehand from the
marking-centre vs lane-edge convention. Fixed nominal geometry costs a lateral
bias growing from +0.08 to +0.26 m with distance.

**The gross errors were not errors.** The two ~4 m frames are the only two where
the camera centreline lies outside the labelled lane: the car was straddling the
line, the solver found that line and warned correctly.

**The departure metric needed re-reading.** 13 frames meet the warning rule, but
8 graze its threshold by under 9 cm. Reported by condition instead: 3/5
body-over-line frames caught (CI 23–88%), 2/82 false warnings.

**Three hypotheses refuted, in order.** (1) "The failed edges are unpainted kerbs"
— the marking filter responds along 94–95 of 95 labelled edges. (2) "A yaw in the
road fit rotates the ground truth" — yaw is −0.44° everywhere; the diagonal that
prompted it was my diagnostic reading a quadratic fit at 25 m, past the label's
~17 m extent. (3) "The ±0.4 m search deadzone hides a departing car's line" — 1
of 13 frames, and it was caught. Nothing in the frozen config was changed.

**New lesson:** evaluate a fitted curve only where it has support. The harness
did; an ad-hoc diagnostic did not.

**Landed:** `aps/kitti/road.py`, `aps/lanes.py`, `configs/lanes.yaml`,
`eval/eval_lane.py`, 8 failure images, 19 new tests (168 total).

**Next:** Phase 7 — the FCW / AEB-request decision layer, evaluated as a detector
with TPR and FP/hour, consuming Phase 5's TTC.

---

## 2026-09-15 (cont.) — Phase 7: the metric that could not be produced

Built the decision layer (`aps/fcw.py`): warn on an in-path, closing object whose
TTC stays below a threshold for k of the last n frames; request AEB below a
stricter threshold. Pure logic, 16 tests. The harness collects perception once
and replays every decision configuration over identical rows.

Two design constraints were set before any result existed: exposure is minutes,
so false alarms get exact Poisson intervals rather than a bare per-hour figure;
and choosing an operating point on val makes val a selection set, so the unbiased
number would have to come from the held-out test drives.

**Neither constraint ended up mattering, because the data has no threats.** Val
holds **2 in-path threat events** below a 2 s TTC — 11 frames out of 8383
annotated samples — across 21 distinct in-path closing objects, with a minimum
in-path TTC of **1.17 s** over 2.72 minutes. Hard rule 7's TPR and FP/hour are
not producible, and that is Phase 7's result ([D-024](decisions.md)).

**Checked my own framing before publishing it.** The obvious objection is that
the ±1.5 m straight corridor is too narrow. Widening it produces events — 2 → 8
→ 84 as the half-width goes 1.5 → 2.5 → 5.0 m — but a 5 m half-corridor spans
oncoming and adjacent lanes, which an FCW must not warn about. So the threat
count is governed by the in-path definition rather than the data, and a TPR would
describe the corridor.

**What the sweep does show:** persistence cuts onsets 13 → 3 and false alarms
12 → 2 at a 0.2 s latency cost; half of all false alarms are ghost tracks with no
annotated object behind them; and the Phase 5 `sigma` deadband catches 0 of 2
events against 1 of 2 with no deadband — the suppression D-022 predicted, landing
on the threats.

Also moved ground-truth kinematics out of `eval_ttc.py` into `aps/groundtruth.py`
so Phase 5 and Phase 7 score against one definition, and verified the refactor
against the saved Phase 5 artifact before using it.

**Landed:** `aps/fcw.py`, `configs/fcw.yaml`, `eval/eval_fcw.py`, 16 tests
(184 total).

**Next:** Phase 8 — HUD and top-down view, which presents results that already
exist, then Phase 9's writeup.
