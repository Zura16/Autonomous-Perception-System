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

---

## 2026-09-15 (cont.) — Phase 8: HUD and top-down

Built last on purpose. `app/replay.py` composes the camera view with boxes,
per-track range and TTC, the lane overlay and a metric top-down panel, and adds
no estimate of its own.

Three display rules, each a hard rule made visual: abstentions render as `--`
rather than a filled-in guess; objects beyond the measured 10–30 m envelope are
drawn dimmed and **without numbers**, so the HUD cannot launder an unsupported
value into a screenshot; and the scope strip — monocular, offline, open-loop,
nothing actuated, LiDAR is ground truth only — is stamped on every frame.

**A still exposed a real defect in the still itself.** Rendering one frame leaves
the scale-rate filter with a single observation, so every TTC correctly read
`--`, which misrepresents the running system. Still mode now replays the
preceding frames to warm the tracker and filter before writing.

The replay also refuses to touch the held-out test drives: it loads dev+val only,
so a viewer cannot be the thing that opens the test split before Phase 9.

**Landed:** `app/replay.py`, `docs/figures/hud_still.jpg`.

**Next:** Phase 9 — the writeup.

---

## 2026-09-16 · Phase 9 — the held-out split, and what it broke

The test drives (`0009`, `0015`) had been on disk since Phase 2 and untouched by
design. Phase 9 ran the frozen pipeline on them once.

**Before anything could run, a crash.** Both ground-truth harnesses died on
`drive_0009` frame 177 with `FileNotFoundError`. The drive ships 447 images and
**443 velodyne scans** — KITTI does not guarantee one scan per image. Added
`Frame.has_velodyne`, made both harnesses skip *and report* those frames, and
taught `fetch_kitti.py --check` to count every kept stream rather than images
alone, so the gap is visible on disk before an evaluation trips over it. A
silently skipped frame is a silently shrunken denominator.

**Then the result.** The credible operating envelope declared in Phase 3 —
10–30 m at ≤15% MAPE — **did not hold**. Best-per-bin on test: 24.1 / 17.2 /
15.6 / 14.4 / 14.3 %, against val's 22.7 / 10.3 / 11.0 / 15.3 / 16.5. Both bins
the claim rested on missed; the far bins improved.

The ruler transferred (0.27 m MAE vs 0.25) and the detector *improved*
(AP 0.737 vs 0.615), so the estimator was handed more and better boxes and still
did worse.

**Four hypotheses, three dead.** Degraded ruler: no. Occlusion: a nearer object
was present for 87.9% of gross outliers and 85.7% of everything else — no
discrimination, and the first version of that test was too coarse to say
anything, which is itself the lesson. Road non-flatness, the mechanism that
explained the far-range bias in Phase 3: predicted −0.65 m where +9.08 m was
observed. Wrong sign, an order of magnitude short.

**What it was.** Ten detections of 199 carried 81% of the worst block's bias,
against a block median of −1.54 m. All had box bottoms ~4.7 px below the horizon,
where `D = f·h/(v_bottom − v_horizon)` divides by almost nothing; the worst
returned 253 m for an object at 25.8 m. `min_pixels_below_horizon: 3` had always
been a division-by-zero check wearing the name of a validity check — it permits a
398 m answer at 33% range error per pixel. Fixed by abstaining past the 50 m
envelope D-015 had already declared.

**The fix does not rescue the claim** (test 10–20 m is still 15.8%) and it is not
clean evidence, because the defect was found by looking at test. Both facts are
recorded next to the numbers, and the as-frozen result stays the headline. It
*does* repair val, whose 30–50 m figure was 18.7% and is 11.8% gated — the pole
had been inflating validation all along, invisible because it never grew large
enough to look wrong.

**The finding that outlives the fix.** Median APE moved 1–3 points where mean
MAPE moved 8. This project had reported the mean as its headline for six phases
and the two had always agreed, because val's tail was thin. Every range row now
carries its median. Related: five val sequences spanning 8.5–12.4% looked like
convergence, and blocks *within* one test sequence range from 6.6% to 55.5% —
scene variation exceeds every effect previously measured here.

**Also landed:** the range-curve figure regenerated from a committed artefact
(reading the ruler floor from the plotted split's own data, drawing median beside
mean, stamping provenance on any figure built from a non-default artefact), and
`claim_check.py`, which audits `claims.md` against benchmark rows and git. Two
subtleties it now gets right: a claim citing several rows pins to the **latest**
of their commits, not the earliest — otherwise a claim revised to rest on new
evidence gets backdated to before that evidence existed; and `Row 3.1–3.2, 9.3`
cites three rows, where the original regex saw one and left the rest unaudited.

**Landed:** `aps/kitti/dataset.py`, `aps/geometry.py`, `configs/camera.yaml`,
`tools/plot_range_curve.py`, `tools/claim_check.py`, `tools/fetch_kitti.py`,
`eval/eval_road_profile.py`, Rows 9.1–9.7, D-025, FM-21–FM-23, C-25–C-29,
`docs/figures/range_error_{val,test,test_gated}.png`.

**Next:** README scope pass complete; remaining Phase 9 work is the end-to-end
latency row (8.x) and a final `/quiz` against the four load-bearing components.

---

## 2026-09-17 · Phase 9 continued — latency, and a HUD that had outlived its claim

**The last open benchmark row.** `bench/run_bench.py` was listed in CLAUDE.md's
command table and had never been written, so only *detection* had ever been timed
and hard rule 10 was open at the pipeline level. Built it, and the pipeline fits:
**p99 61.91 ms against the 103.56 ms budget, 60% of it**, N=1548 val frames,
30 warmup excluded, device synchronized inside every timed region. By hard rule
11 no optimization follows, and that is recorded so it does not get reopened.

Detection (37.6%) and lanes (25.3%) are the whole cost. Tracking, range, TTC and
the decision layer together are **1.1%** — the geometry that took six phases to
characterise is free, and the two stages nobody had to think about are the entire
latency story.

Three things the harness is built to get right, and one it got wrong first:

- **the total is timed, not summed.** Summing per-stage p99s reads 66.31 ms
  against a measured 61.91 — percentiles are not additive, because stages do not
  peak on the same frames;
- **N changed the answer by 37%.** The first run used 300 frames of one drive and
  reported p99 45.28 ms. Across the full split it is 61.91–64.61. The small
  number was not miscalculated; it was a percentile estimated from too few frames
  to contain the stalls it purported to describe. That is Row 1.3's 50+ m bin and
  Row 9.5's ten detections again, in a third quantity;
- **the over-budget count does not reproduce.** Two runs of an identical config
  gave 3 and 7 frames over budget, and a *different stage* spiked each time —
  which is how they are identified as host scheduling stalls rather than pipeline
  work. p99 is stable to ±4%, so the count is published as a range;
- **the first warmup implementation was wrong.** It tried to undo a recording it
  had already made and failed on the first frame of every stage. Warmup is now a
  single slice across aligned series. The report also printed the unused
  `--drive` default while five drives were being measured — a latency row naming
  the wrong sequence fails hard rule 10 as surely as a missing p99.

**The HUD had outlived its own claim.** `app/replay.py` hardcoded
`ENVELOPE_M = (10.0, 30.0)` and a matching footer caption, both of which stayed
on screen after the held-out split withdrew that band. It now reads
`CredibleEnvelope` from `configs/camera.yaml` and generates the caption from it.

Two real defects surfaced in doing that:

- **the gate was one-sided.** The camera view checked only the upper bound while
  the top-down panel checked both, so the two panels disagreed about what was
  "inside" — and the near field, the *worst* range the project measures, was
  being printed as vouched while 30–50 m values, among the best, were dimmed away;
- **TTC was being withheld along with range.** The two are measured separately,
  and TTC is unbiased where it matters precisely because the scale-rate ratio
  cancels the box bias that drives range error. Gating it on a range criterion
  suppressed the most trustworthy output in the stack, hardest in the near field
  where a warning matters most. It is now exempt, declared as `gates_ttc: false`
  in config so the exemption is a stated decision rather than an omission.

The consequence is published rather than styled away: **the HUD now goes quiet on
range in the near field**, because that is where the estimator is least
trustworthy and where a collision warning most needs to be right.

**Landed:** `bench/run_bench.py`, `aps/geometry.py` (`CredibleEnvelope`),
`configs/camera.yaml`, `app/replay.py`, Rows 8.1–8.4, C-30–C-32, 189 tests.

**Next:** the `/quiz` cold reimplementation of the four load-bearing components
(range geometry, scale-rate/KF, tracker association, FCW thresholds) — the last
thing D-001 leaves outstanding before any of this goes on a résumé.

---

## 2026-09-19 · The near-field residual, closed by geometry

The last unexplained error source in the project, open since Phase 3 and
recorded in the charter for six phases as "UNIDENTIFIED — flat +0.7 m residual."

**A false start worth recording.** The first diagnostic compared the observed
box-bottom "deficit" against the pixel shift "needed" to explain the bias, and
showed a 6–13 px gap in the near field. There is no gap: the two are
algebraically identical per object (both equal `f·h·(1/t − 1/p)`), verified to
1e-14. The apparent gap came from comparing a median-of-differences with a
difference-of-medians — the project's own standing warning, walked into while
using it.

**What the real decomposition showed.** Label boxes sit near the flat-ground
prediction (+5.8 px val, −3.3 px test — sign flips, so not systematic), and the
LiDAR ruler agrees with the independent 3D labels to 0.13–0.21 m in *every* bin,
which kills the "shrink_p20 breaks on very large boxes" suspect outright. But the
worst near-field detections had box bottoms at row **371 of 375** and a true
range of **~4 m**, and were over-estimates **100%** of the time.

**The cause.** A contact point at range D lands at `cy + f·h/D`, which leaves a
375-row image below `D_min = f·h/(H_img − cy)` = **5.91 m**. Closer than that the
contact point is not in the picture, the box bottom saturates at the frame edge,
and range cannot read below ~5.9 m. Predicted 5.91 with nothing fitted; measured
**6.03 m (val) / 6.02 m (test)**.

**Size-prior fails there too, and worse** (58.8% vs 49.2%), which I had expected
to be immune since it contains no ground plane. The same edge truncates box
*height*. One boundary, two mechanisms.

So it is a **field-of-view limit, not an estimator defect** — and a concrete
answer to why production AEB fuses sensors.

**The fix and its limits.** The guard must read the box's distance from the edge,
not the estimate: unlike D-025's pole, saturation produces no absurd value to
catch. `min_rows_from_bottom` 2 → 10, swept on **val only** (sub-D_min boxes sit
4 px from the edge, valid ones 140 px; 10 px catches 83% for 0.34% of valid
boxes). val 0–10 m MAPE **21.0 → 12.4%**, size-prior 29.5 → 19.9%, **zero boxes
dropped at 10–50 m**.

Two things reported rather than smoothed:

- **the prediction generalised; the fix only partly did.** test gains 2.2 points
  against val's 8.6, because the 17% of boxes that leak past the margin are much
  worse there — 19 survivors at 82% MAPE carrying 99% of the bin's bias. The
  margin was **not** retuned on test. Having made that error once in D-025,
  making it again knowingly would be worse;
- **coverage is the price.** Near-field range is now declined on 32% of val
  objects and 48% of test. The band where a warning matters most is the band the
  estimator most often refuses to measure, and whether that beats a saturated
  answer is a decision-layer question this does not settle.

**Still open:** +0.60 m (val) / +0.54 m (test) between 5.91 and 13 m — the
original residual, now isolated and a third of its apparent size.

**Landed:** `aps/geometry.py` (`min_supportable_range_m`), `configs/camera.yaml`,
Rows 9.8–9.10, D-026, C-33–C-35, 193 tests.

**Next:** the `/quiz` cold reimplementation — the last item D-001 leaves open.

---

## 2026-09-21 · Closing the loop: CI, a re-measured budget, pinned versions

Three gaps from the end-to-end audit, all closed.

**CI.** Every guard this project has — documented commands executed against
argparse, config keys required to be read or declared inert, claims and
cross-references audited — was only ever run by hand. `.github/workflows/ci.yml`
runs lint, tests, `claim_check` and a scope-language grep on every push. It
installs `.[dev]` and deliberately **not** `.[detect]`: the detector is ~2 GB of
torch and every check here is static or LiDAR-only, so the job finishes in under
a minute. One non-obvious setting: `fetch-depth: 0`, because `claim_check`
resolves each claim to the commit that introduced its benchmark row and a shallow
clone would make every claim look unpinnable.

**Row 8.1 re-measured on the shipped configuration.** The latency row predated
D-026, so the benchmark and the code disagreed about what was being timed. Re-run:
**p99 59.67 ms, 58% of the 103.56 ms budget**, against 61.91 pre-gate — a 3.6%
difference, well inside the ±8% run-to-run spread. The gates only decide whether
a range value is returned, so this is the expected null result, now on the record
rather than assumed.

It also gives Row 8.3 a third repeat, which sharpens the finding: across three
runs p99 spans 8%, the over-budget count goes 3 / 7 / 3, and `max` varies by 39%.
The run that differs by a *configuration change* moves less than two identical
runs differ from each other — which is the cleanest possible statement that the
spread is host scheduling noise and not the pipeline.

**Versions pinned.** `pyproject.toml` used `>=` with no upper bound on
everything, so `pip install -U` could silently invalidate every number in
`benchmarks.md` with no commit to blame. `ultralytics` is the sharp edge: it
ships the weights *and* the NMS defaults, so a minor release moves AP and
therefore every range, TTC and FCW figure derived from those boxes. Added upper
bounds, a `requirements-lock.txt`, and Row 0.2 recording the exact versions every
row was measured on — a measurement whose software version is unrecorded is no
more reproducible than one whose split is unrecorded.

**Landed:** `.github/workflows/ci.yml`, `requirements-lock.txt`, Row 0.2,
Rows 8.1–8.4 re-measured, C-36, bounded `pyproject.toml`.

**Next:** the `/quiz` cold reimplementation ([D-001](decisions.md)) — the last
outstanding item in the project.
