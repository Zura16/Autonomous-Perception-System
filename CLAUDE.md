# CLAUDE.md — Autonomous Perception System (APS)

Auto-loaded every session. This file is the operative context: scope, hard rules, commands, current status. Deep detail lives in the docs — read them on demand, don't guess:

| Doc | What it holds | Read it when… |
|---|---|---|
| [docs/benchmarks.md](docs/benchmarks.md) | **The only source of numbers.** Every measurement with its full context | A number is about to be stated anywhere |
| [docs/error-budget.md](docs/error-budget.md) | How error propagates: pixel → range → velocity → TTC → decision | Any change touching geometry, filtering, or thresholds |
| [docs/decisions.md](docs/decisions.md) | Every non-obvious choice + *why* + what breaks it | Before re-deciding anything; "why did we…" |
| [docs/failure-modes.md](docs/failure-modes.md) | Documented conditions where each stage breaks, with frames | README, interview prep, scoping the next phase |
| [docs/claims.md](docs/claims.md) | Every external claim → benchmark row + commit hash proving it | Writing resume / README / portfolio copy |
| [docs/glossary.md](docs/glossary.md) | Terms as used here + canonical constants | A term or project constant is needed |
| [docs/history.md](docs/history.md) | Day-by-day journal of what landed | How/when something was built |
| [docs/tradeoffs.md](docs/tradeoffs.md) | Honest steelmen per component: what it buys, costs, **when the alternative wins** | "Defend your design"; `/quiz` source |

---

## What APS is

A **monocular vision perception and forward-collision-warning stack**, evaluated against LiDAR ground truth before anything is claimed about it.

```
KITTI frame ─► YOLOv8 detection ─► 2D tracker (IoU → SORT) ─► monocular range ─► scale-rate velocity + KF ─► TTC ─► FCW / AEB-request logic
            └► OpenCV lane solver ──────────────────────────► lane-departure metric ────────────────────────────┘
Ground truth: projected LiDAR range  ·  Output: HUD + top-down view (Phase 8, deliberately last)
```

**Scope, stated identically everywhere:** a single forward camera. No LiDAR, no radar, no fusion, no vehicle actuation in the estimation path — LiDAR appears **only** as ground truth for evaluation. The system emits **warnings and an AEB _request_** in an offline, open-loop evaluation. It does not brake anything.

**The purpose is a defensible error characterization, not a demo.** A dashboard nobody can falsify is worth less than an error-vs-range curve on a plain white background. Every session must leave a row in `benchmarks.md` that survives *"on what data, at what range, with what spread?"*

**Vocabulary that must be used correctly.** Production AEB is a safety-critical function under **ISO 26262** (functional safety) and **ISO 21448 / SOTIF** (hazards arising from performance limitations of a system working as designed), and production stacks fuse radar with camera *because* monocular range is unreliable. Knowing precisely why this project is not that is the differentiator. Implying it is that is the disqualifier.

## Teaching contract

1. **Teach before/while doing.** Every non-trivial choice: the pattern, the tradeoff, the alternative rejected, what would change in a production ADAS.
2. **Flag production-grade vs. simplified** explicitly (offline replay vs live, single camera, no functional safety, no HIL). Articulating the gap *is* the skill.
3. **No silent shortcuts.** Skipped calibration checks, unfiltered derivatives, hand-tuned thresholds — logged as conscious decisions, never quietly omitted.
4. **Small, reviewable changes.** One concept per commit, imperative mood (`add scale-rate closing-speed estimator with deadband`).
5. **Plan mode for anything crossing stages.** The stages are coupled through error propagation — a change in detection changes TTC.
6. **Bank the finding.** *"Range error exceeded 20% beyond 40 m, so FCW is only credible inside 40 m"* is the strongest sentence this project can produce. Suppressing it is the weakest thing it can do.
7. **End every explanation with the interview version:** one or two sentences carrying the number **and its caveat**.

### Who writes the code

**Coach mode is currently OVERRIDDEN — see [D-001](docs/decisions.md).** The author elected to have Claude implement and to review diffs instead. The charter's original terms are preserved below and resume at any stage the author asks to take over, with no renegotiation.

> **Coach mode (suspended).** Set up, don't solve. Hint ladder, one level at a time, only on request: (1) concept + where to look → (2) shape (pseudocode / signature) → (3) targeted snippet → (4) full solution, **only on explicit request**, logged as "solved for me — revisit." Debugging is a rep: ask for the observation and the hypothesis first. In this stack the bug is almost always a coordinate frame, a units mismatch, or an unfiltered derivative — rarely the model.

While the override stands, `/quiz` is **load-bearing rather than optional**, and the four pieces the charter singles out — range geometry, the scale-rate/KF velocity estimator, tracker association, FCW thresholds — each need a cold reimplementation before this goes on a résumé. Evaluation and the error-budget update stay mandatory **regardless of who typed.**

## Session protocol

- **Start:** read *Current status*. Confirm the dataset split and calibration in use — a split change invalidates every row in `benchmarks.md`.
- **During:** follow the active `docs/daily/day-NN.md`; one commit per step; `/eval` before calling anything done.
- **End:** run `/end-session` — reruns eval, lints, updates status/history/decisions, runs `/claim-check` and `/scope-check`.

## Hard rules (non-negotiable)

1. **Ground truth before estimation.** The LiDAR-derived range harness (Phase 1) exists before the monocular estimator (Phase 3). An estimator with no ruler is a decoration. This ordering is the project's entire thesis.
2. **Never report a single accuracy figure for monocular range.** Report **error vs. range**, binned (0–10 / 10–20 / 20–30 / 30–50 / 50+ m), with MAE, MAPE, and spread. A lone "±2 m" headline conceals that pinhole sensitivity is worst at distance — which is exactly what a good interviewer is probing for.
3. **Never differentiate distance to obtain velocity.** Use the scale relation `Ḋ = −D·ḣ/h` with a Kalman filter over the image-plane state, so range, closing speed, and TTC derive from one consistent state and can never contradict each other.
4. **Phantom closing speed is a bug, not noise.** A parked vehicle must produce ≈0 closing speed. A deadband on the closing rate is mandatory, and its threshold is a logged decision with a measured false-trigger rate.
5. **State the pitch assumption everywhere it bites.** Flat-ground contact-point geometry assumes zero camera pitch. **RESOLVED by measurement ([D-016](docs/decisions.md)):** true camera-to-road pitch, from a LiDAR road-plane fit, is mean +0.150° with \|p95\| **0.715°** — the earlier OXTS figures (2.0–2.2°) were the wrong quantity and overstated it 2–3×. Decision: **characterise, do not correct**, with a measured sensitivity curve (Row 3.4). Measured error turns out **flat across pitch**, masked by larger error — partly detector box bias, and partly a near-field cause that is still **unidentified** ([D-017](docs/decisions.md), [D-018](docs/decisions.md)). Pitch becomes the binding constraint only once those are resolved.
6. **Object-size priors carry their variance.** A prior without a variance is a guess wearing a lab coat.
7. **The decision layer is evaluated as a detector, never demoed.** FCW/AEB-request logic reports true-positive rate **and false positives per hour of video**, at a named TTC threshold. FP rate is the number that matters: a phantom brake at highway speed is itself the crash.
8. **Trackers are measured, not eyeballed.** Report MOTA / IDF1 / ID-switches, and land a SORT comparison. "The boxes look stable" is not a result.
9. **Lane detection ships with its failure set.** Those frames go into `failure-modes.md` with images; the departure metric is scored on a labeled set, never one hand-picked clip.
10. **No number without its context, and latency is always against the budget.** Every latency row carries hardware, resolution, batch size, N frames, warmup excluded, and **p50/p95/p99** — reported against the named budget of **103.56 ms/frame** (the measured 9.657 Hz sensor rate). "Real-time" with no denominator is marketing. Every accuracy row carries dataset, split, and N.
11. **Optimization is conditional on missing the budget, and every speedup names its baseline.** No optimization work begins until a stage is *measured* to exceed its share of 103.56 ms — optimizing a stage that already fits is theatre ([D-012](docs/decisions.md)). If it does begin: the full ladder is recorded, every rung with a **paired accuracy delta** on the same split, and a speedup reported without its accuracy cost is an unfinished experiment. **Async dispatch must be synchronized before the clock stops** (`torch.mps.synchronize()` here, `cudaEventSynchronize` elsewhere) — timing an async launch measures the dispatch, not the work.
12. **Claims are generated, never authored.** `docs/claims.md` rows are written *from* `benchmarks.md`, each pinned to a commit hash.
13. **Scope language is fixed.** "Monocular." "Warning and AEB request." "Offline, open-loop." Never "sensor fusion," never "commands braking," never "real-time" without a named budget.
14. **Ground truth is never assumed valid.** Every GT value carries a tier and a coverage figure. A box you cannot see into has **no** ground truth — abstaining is required, filling it in is a fabricated measurement. *(Measured: a naive estimator on fully-occluded boxes reports the occluder, MAE 11.9 m — [D-010](docs/decisions.md).)*

## Conventions

- **Python:** `ruff` + `black`; type hints on signatures; no bare `except`. Pure geometry/filter functions separated from I/O and drawing so they stay unit-testable — the geometry has closed-form test cases, so use them.
- **Config over constants.** Intrinsics, mounting height, class priors, TTC thresholds, deadband: all in `configs/`, never inline. A magic number in a `.py` file is a bug.
- **Units and frames live in every name.** `range_m`, `closing_speed_mps`, `ttc_s`, `bbox_px`. Mixed units is the most common silent failure in this stack.
- **Tests:** `pytest`. Closed-form cases for projection and range; synthetic constant-velocity sequences for the KF and TTC; a golden-frame regression fixture for detector output.
- **Plots are artifacts.** Every figure in `docs/figures/` is regenerated by a committed script from a committed benchmark row. No notebook screenshots.
- **`legacy/` is excluded from lint and tests.** It is a museum piece, not code.

## Reference values (canonical — verify against these)

> Each row is filled by a measurement, never an estimate. `TBD` here is honest; a guessed value here is the exact failure mode this file exists to prevent.

| Thing | Value |
|---|---|
| Dataset | KITTI **raw**, date `2011_09_26` (not the tracking benchmark — [D-004](docs/decisions.md)) |
| Split (by sequence — [D-002](docs/decisions.md)) | **dev** `0013` · **val** `0059` `0084` `0091` `0056` `0027` · **test** `0009` `0015` (held out) |
| Labelled boxes | dev 385 · val **8705** · test unaudited by design |
| Intrinsics (cam2, rectified) | `fx = fy = 721.5377 px` · `cx = 609.5593` · `cy = 172.8540` · `1242 × 375` · HFOV **81.4°** |
| rect0 → cam2 offset | `[+0.0598, −0.0004, +0.0027] m` (z ≈ 0 ⇒ depths interchangeable) |
| Frame rate | **9.657 Hz** (dt = 103.56 ms ± 0.06), measured — *not* the documented 10 Hz ([D-006](docs/decisions.md)) |
| `range_m` convention | Longitudinal, **near face** ([D-003](docs/decisions.md)). Centroid runs **+2.03 m** farther |
| Hardware | Apple M2 (4 P-core + 4 E-core), macOS 26.5.1, Python 3.12.4. **No CUDA** — MPS or CPU |
| **Latency budget (sensor rate)** | **103.56 ms/frame** = 9.657 Hz. The denominator for every latency row ([D-012](docs/decisions.md)) |
| **GT ruler** (`shrink_p20` + spread gate, **val**) | MAE **0.25 m**, med 0.17, p95 0.63 · by bin **0.21 / 0.24 / 0.25 / 0.27 / 0.78 m** · failure rate 0.27% |
| GT coverage (val) | **38.1%** of labelled boxes end with a ground-truth range (45% usable tier → 89% measured → 95% pass spread gate) |
| Spread gate | abstain when interquartile depth spread > **1.5 m** ([D-013](docs/decisions.md)) |
| Detection AP@0.5 (YOLOv8n 640px, val, class-agnostic) | **0.615** · vehicle 0.662 · VRU 0.450 |
| Detection recall vs range (conf 0.25) | **86 / 76 / 65 / 46 / 24 %** · VRU **75 / 69 / 36 / 2 / 0 %** |
| Detection latency (M2 MPS, 640px, batch 1) | p50 **17.12** · p95 24.75 · p99 **34.01 ms** = 33% of budget · **5/1450 frames over budget**, max 553 ms |
| **Evaluation envelope** | **0–50 m** — bounded by evidence, not accuracy. Usable N/bin: 430/1142/793/475/**30** ([D-015](docs/decisions.md)) |
| **Range MAPE by bin** — contact-point (val, post-D-025) | **21.0 / 10.1 / 10.7 / 11.8 %** (MAE 1.15/1.48/2.65/4.36 m) |
| **Range MAPE by bin** — size-prior (val) | **29.5 / 14.9 / 11.0 / 15.3 %** |
| **HELD-OUT test, as frozen** | best-per-bin **24.1 / 17.2 / 15.6 / 14.4 %** — **the 10–30 m envelope FAILED** ([D-025](docs/decisions.md)) |
| **Credible operating envelope** | **20–50 m at ≤13%, on both splits.** 10–20 m is marginal (val 10.1%, test 15.8%); the near field fails on both. The former "10–30 m at ≤15%" did **not** survive held-out data |
| **Median vs mean APE** (val → test) | 7.5→10.3 / 7.7→8.8 median against 10.3→18.4 / 12.2→20.3 mean. **The centre transfers; the tail does not** |
| Camera geometry (LiDAR-measured) | height **1.655 m** (std 0.027) · camera-to-road pitch mean +0.150°, \|p95\| **0.715°** |
| Detector box height bias (val, 5254 pairs) | **−4.5% overall · −7.1% at 0–10 m** · bottom edge −4.8 px near. A bias, not jitter ([D-018](docs/decisions.md)) |
| Far-range error cause | **road non-flatness** — road 10 cm below the assumed plane by 50 m; explains most of the bias beyond 20 m ([D-020](docs/decisions.md)) |
| Near-field error cause | **UNIDENTIFIED** — flat +0.7 m residual inside 13 m; ~0.3 m vehicle-specific, ~0.4 m common |
| Closing-speed error vs GT (val, no deadband) | rel MAE **28 / 32 / 51 / 78 %** by bin 0–10…30–50 m — inherits range error ([D-022](docs/decisions.md)) |
| **TTC error where it matters (GT TTC < 3 s)** | MAE **0.41 s**, median **+0.01 s** (unbiased), rel 30%, p90 1.02 s — common subset N=1930 |
| Phantom closing (relatively stationary) | **54.3%** no deadband · 11.9% fixed@0.05 · 2.6% sigma@2.0 — synthetic model predicted 54.2 / 10.2 |
| Imminent threats silenced (GT TTC < 3 s) | none 1.2% · fixed 1.6% · **sigma 10.2%** — deadband operating point deferred to Phase 7 |
| **Tracking (val, identical detections)** | `iou` MOTA 0.268 / IDF1 0.488 / **507 IDsw** · `sort` 0.270 / 0.515 / 354 · **`sort_det` 0.284 / 0.522 / 358** ← shipped |
| Tracking ablation | association **−29% ID switches**; Kalman smoothing of the OUTPUT costs MOTP 0.769→0.735 and is discarded ([D-021](docs/decisions.md)) |
| **Lanes (95 labelled KITTI road frames, params frozen first)** | detection **97.9%** · offset MAE **0.18 m** excl. 2 straddling frames (0.26 m all) · width bias +0.17 m (predicted) |
| Lane departure warning | body-over-line **3/5** (95% CI 23–88%) · false warnings **2/82 (2.4%)** · 8 of 13 rule frames graze the threshold ([D-023](docs/decisions.md)) |
| **FCW: TPR and FP/hour** | **NOT PUBLISHABLE on this data** — val holds **2 threat events** below TTC 2 s in **0.045 h** of exposure; min in-path TTC 1.17 s ([D-024](docs/decisions.md)) |
| FCW decision layer, what is measured | persistence 3/3 cuts onsets 13→3 and false alarms 12→2 at TTC 2.0 s for 0.2 s latency · **half of false alarms are ghost tracks** · `sigma` deadband can suppress the threat |
| **End-to-end latency (M2 MPS, val, N=1548, 30 warmup excl.)** | TOTAL p50 **30.13** · p95 43.47 · p99 **61.91 ms = 60% of budget** · 3–7 frames over budget across runs |
| Per-stage p99 (same run) | detect **38.91** (37.6%) · lanes **26.24** (25.3%) · track 0.65 · motion+TTC 0.32 · range 0.11 · FCW 0.08 ms. Decode 15.88 ms measured, **not charged** |
| Latency caveat | summing stage p99s overstates the total by **7.1%** — percentiles are not additive. N=300 understated total p99 by **37%** (Rows 8.2, 8.4) |
| GT ruler on **held-out test** | MAE **0.27 m**, fail 0.15% — against val's 0.25 m / 0.27%. The instrument transferred |
| Detection on **held-out test** | AP **0.737** class-agnostic (val 0.615) · recall 93/89/74/61/51% · VRU AP **0.127** on 250 labels |
| Test suite count | **186** |

## Common commands (run from repo root)

```bash
# ── Data
python tools/fetch_kitti.py --split dev val        # fetch + unpack + prune
python tools/fetch_kitti.py --check                # verify what is on disk
python tools/audit_labels.py                       # label density per drive
python tools/audit_labels.py --probe 0059 0084     # 1.5 MB probe before a 1.5 GB commit

# ── Ground truth (Phase 1 — the ruler)
python tools/build_range_gt.py --split val         # characterise the ruler, write the table
python tools/render_frame.py --drive 2011_09_26_drive_0013 --frame 20 \
    --out docs/figures/phase0_projection_check.png # standing calibration check

# ── Evaluate (always before performance or visual work)
python eval/eval_detection.py --split val                      # AP, recall-vs-range, latency
python eval/eval_box_quality.py --split val                   # detector box vs label box, per object
python eval/eval_road_profile.py --split val                  # real road vs the assumed flat plane
python eval/eval_range.py    --split val --bins 10,20,30,50   # → error-vs-range curve
python eval/eval_tracking.py --split val                       # iou vs sort vs sort_det
python eval/eval_ttc.py      --split val                      # TTC, closing speed, phantom rate
python eval/eval_fcw.py      --split val --ttc-threshold 2.0  # TPR + FP/hour
python eval/eval_lane.py --render 8                          # lanes on KITTI road um_lane + failure set

# ── Benchmark (latency; writes a fully-contextualized row)
python bench/run_bench.py --warmup 50 --iters 500 --report p50,p95,p99

# ── Replay / visualize (Phase 8 — the demo, not the evidence)
python app/replay.py --drive 2011_09_26_drive_0059 --frames 0 200 --lanes --out artifacts/replay.mp4
python app/replay.py --drive 2011_09_26_drive_0059 --still 96 --lanes --out docs/figures/hud_still.jpg

# ── Tests & lint
python -m pytest tests/ && ruff check . && black --check .
```

## Milestones

Ordering is deliberate: **the ruler is built first, the dashboard last.**

- [x] **Phase 0 — Repo + dataset + one frame.** KITTI loader, calibration parsed, one frame rendered with boxes. Environment fingerprinted.
- [x] **Phase 1 — Ground truth harness.** Project LiDAR into the image; per-detection GT range, with the ruler's own error and coverage characterised by range bin.
- [x] **Phase 2 — Detection baseline.** YOLOv8n zero-shot: AP 0.615 class-agnostic, recall-vs-range measured, latency at 33% of budget at p99. No optimization — nothing missed the budget except five host stalls ([D-012](docs/decisions.md)).
- [x] **Phase 3 — Monocular range + error-vs-range curve.** ✅ Two estimators, pitch characterised not corrected. Declared envelope **10–30 m at ≤15% MAPE** — later **refuted on held-out test** and replaced by **20–50 m at ≤13%** ([D-025](docs/decisions.md)).
- [x] **Phase 4 — Tracking.** ✅ IoU baseline vs SORT vs SORT-with-raw-output. Association buys −29% ID switches; canonical SORT's smoothing costs MOTP and is discarded ([D-021](docs/decisions.md)).
- [x] **Phase 5 — Closing speed + TTC.** ✅ KF over `u = 1/h` (exact under constant velocity), measured noise model, TTC unbiased (MAE 0.41 s < 3 s). Deadband measured as a gate; operating point deferred to Phase 7 ([D-022](docs/decisions.md)).
- [x] **Phase 6 — Lane detection + departure metric.** ✅ Metric BEV + top-hat + sliding windows on 95 labelled KITTI road frames, parameters frozen before evaluation. Offset MAE 0.18 m; failure set with frames ([D-023](docs/decisions.md)).
- [x] **Phase 7 — FCW / AEB-request decision layer.** ✅ Built and swept. **TPR/FP-per-hour declined as unpublishable**: 2 threat events in 2.7 minutes ([D-024](docs/decisions.md)). No operating point selected.
- [x] **Phase 8 — HUD + top-down view.** ✅ `app/replay.py`. Presents measured results only: abstentions stay blank, values outside the credible envelope are dimmed and their range withheld, scope stated on every frame. The envelope is now **read from `configs/camera.yaml`**, so the HUD follows the measurement instead of a hardcoded band; **TTC is deliberately exempt** from that gate ([D-022](docs/decisions.md)).
- [ ] **Phase 9 — Writeup.** README leading with the error curve and the operating envelope; `claims.md` generated from `benchmarks.md`; failure modes published, not buried. **Held-out evaluation done — and it broke the headline claim ([D-025](docs/decisions.md)).**

## Current status

> Keep SHORT (≤ 15 lines). `/end-session` updates it; the narrative goes to `docs/history.md`.

- **Phase:** **Phase 9 in progress** (2026-09-17). Phases 0–8 ✅. Held-out evaluation, README, `claims.md` (29 backed + pinned) and `/scope-check` **done**. Remaining: the `/quiz` cold reimplementation of the four load-bearing components.
- **⚠⚠ THE HELD-OUT SPLIT BROKE THE HEADLINE CLAIM ([D-025](docs/decisions.md), Rows 9.1–9.7).** The declared envelope — **10–30 m at ≤15% MAPE** — **does not hold on test**: best-per-bin **24.1 / 17.2 / 15.6 / 14.4 %**, both envelope bins missing. Far bins *improved*. This is reported as the result, not repaired into one.
- **The ruler and the detector both transferred** — test ruler MAE **0.27 m** (val 0.25), detection AP **0.737** (val 0.615, test is *easier*). So the range estimator was handed more and better boxes and still did worse. The failure is the estimator's.
- **The centre transferred; the tail did not.** Median APE moved 7.5→10.3 and 7.7→8.8, while the mean moved 10.3→18.4 and 12.2→20.3. **MAPE alone described a collapse that never happened to the typical object.** Every range row now carries its median.
- **Root cause: an ungated pole.** 10 detections of 199 carried **81%** of the worst block's bias; all had box bottoms ~4.7 px below the horizon, where `D = f·h/(v_bottom − v_horizon)` diverges — worst estimate **253 m for an object at 25.8 m**. `min_pixels_below_horizon: 3` was a divide-by-zero check, never a validity check (it permits 398 m at 33% error per pixel). Fixed: abstain past `max_range_m = 50 m`, the envelope D-015 already declared.
- **Three rival explanations died on measurement** — degraded ruler (no), occlusion (87.9% vs 85.7%, no discrimination), road non-flatness (predicted −0.65 m against an observed +9.08 m). Recorded because the dead ones are the evidence.
- **The fix does not rescue the claim.** Post-fix test 10–20 m is still 15.8%. **Honest envelope: 20–50 m at ≤13% on both splits**, 10–20 m marginal, near field failing on both. Post-fix numbers are labelled as *not* clean held-out evidence, since the defect was found by inspecting test.
- **⚠ PHASE 7'S RESULT IS THAT THE METRIC CANNOT BE PRODUCED ([D-024](docs/decisions.md)).** Val is **0.045 h** of exposure holding **2 threat events** below TTC 2 s (11 frames of 8383 annotated), over 21 in-path closing objects, minimum in-path TTC **1.17 s**. A TPR over 2 events is not a rate; 12 false alarms carry a 137–462/h interval. **No TPR, no FP/hour, and no operating point is published.** The held-out test split (1.3 min) is smaller still — this needs staged NCAP-style scenarios or hours of naturalistic driving.
- **Threat count is set by the corridor definition, not the data**: 2 events at ±1.5 m, 8 at 2.5 m, 84 at 5.0 m — but a 5 m corridor counts oncoming traffic an FCW must *not* warn about.
- **What Phase 7 does show:** persistence is the strongest lever (3/3 cuts onsets 13→3, false alarms 12→2, for 0.2 s latency); **half of all false alarms are ghost tracks** with no annotated object; the Phase 5 `sigma` deadband can suppress the threat itself (0 of 2 caught).
- **Lanes ([D-023](docs/decisions.md)).** KITTI raw has no lane labels, so scored on the road benchmark's 95 `um_lane` frames; parameters committed before first evaluation (`27904fe`). Detection 97.9 %; offset MAE **0.18 m** excluding 2 lane-straddling frames (0.26 m with them); width bias +0.17 m, predicted beforehand from marking-centre vs lane-edge convention. Fixed nominal camera costs a distance-growing lateral bias (+0.08 → +0.26 m).
- **Two ~4 m lane errors were lane-identity disagreements**: the car straddled the line, the solver found the line being crossed and warned correctly. **Departure warning: 3/5 body-over-line frames (CI 23–88 %), 2.4 % false warnings**; 8 of 13 rule frames graze the threshold by < 9 cm, so a per-frame hit rate there is a coin flip.
- **TTC is the trustworthy output ([D-022](docs/decisions.md)).** Filter over `u = 1/h` (proportional to range, so exact under constant velocity). `TTC = −u/u̇` is calibration- and prior-free. On val at GT TTC < 3 s: **MAE 0.41 s, median +0.01 s** — unbiased, because the −4.5% box bias cancels in the ratio. Closing speed inherits range error (rel MAE 28 → 78 % by range).
- **Deadband is a gate, not a smoother.** Without one **54.3 %** of stationary objects fake closing. It never changes a TTC value — only whether one is emitted. sigma@2.0's better headline error was selection: it goes silent on **10.2 %** of imminent threats (GT TTC median 2.15 s). Fixed and sigma are equivalent at matched phantom rates. **Operating point deferred to Phase 7.**
- **Tracking (Phase 4).** `sort_det` shipped: MOTA 0.284 / IDF1 0.522 / 358 ID switches; association −29 % ID switches, output smoothing discarded ([D-021](docs/decisions.md)).
- **Phase 3 on val, post-D-025 (N=6122).** Contact-point **21.0 / 10.1 / 10.7 / 11.8 %**; size-prior **29.5 / 14.9 / 11.0 / 15.3 %**. The val 30–50 m figure was **18.7% before the gate** — the same pole had been inflating val all along, unnoticed because it never grew large enough to look wrong.
- **Far-range range error explained ([D-020](docs/decisions.md)): road non-flatness.** Road falls ~10 cm below the assumed 1.655 m plane by 50 m; predicted from geometry with no free parameters, matches observation beyond 20 m to 0.1–0.6 m.
- **⚠ STILL OPEN: a flat +0.7 m near-field range residual** inside 13 m, ~0.3 m vehicle-specific. Suspects untested: effective horizon row ≠ `cy` · `shrink_p20` on very large boxes · detector bottom-edge placement on close cars.
- **Verified:** 186 tests, ruff + black clean. Ground-truth ruler MAE **0.25 m** (val) / **0.27 m** (test); detection AP@0.5 **0.615** / **0.737**; detection latency p99 34.01 ms = 33% of the 103.56 ms budget.
- **Known issues:** dev has now pointed the wrong way **three times** (50+ ruler bin, box-bias class mean, tracker ordering) — **dev is for wiring, never conclusions**, and D-025 adds that **val was not enough either** · abstention costs coverage: contact-point validity 95% val / 90% test, only **71%** in the 0–10 m test bin, and it is no longer evaluable at 50+ · tracking metrics self-implemented and raw-tracklet labels, so not leaderboard-comparable · 50+ bins thin throughout.
- **Standing warning:** draft resume bullets describing **Camera-LiDAR fusion in C++/CUDA with TensorRT and ROS/Gazebo** are not this system. Claims get generated from `benchmarks.md` in Phase 9.

## Project skills (slash commands)

| Skill | Use when |
|---|---|
| `/plan-day` | Generate the next `docs/daily/day-NN.md` in the house format |
| `/eval` | Run the full evaluation suite; append contextualized rows to `benchmarks.md` |
| `/error-budget` | Repropagate pixel → range → velocity → TTC error after any geometry or filter change |
| `/bench` | Latency harness; refuses to write a row with missing context |
| `/failure-hunt` | Sweep the split for worst-case frames per stage; add to `failure-modes.md` with images |
| `/claim-check` | Audit `claims.md` against benchmark rows + commit hashes. **Any unbacked claim is a failure** |
| `/scope-check` | Scan repo, README, and docs for overstated language (fusion · commands braking · real-time · high-fidelity) |
| `/teach <topic>` | Deep-dive a concept the APS way (tradeoffs, production gap, interview framing) |
| `/quiz` | Hostile-interviewer mode over everything built so far |
| `/end-session` | Rerun eval, lint, update status/history/decisions, run `/claim-check` + `/scope-check` |
