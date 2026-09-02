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
5. **State the pitch assumption everywhere it bites.** Flat-ground contact-point geometry assumes zero camera pitch; a 2° error alone already exceeds 10% range error — and the ego vehicle pitches precisely when braking. Either estimate pitch or document the degradation with numbers. Silence is not an option. *(Measured: 1.05° peak-to-peak on a gentle city drive with no hard braking — [D-009](docs/decisions.md).)*
6. **Object-size priors carry their variance.** A prior without a variance is a guess wearing a lab coat.
7. **The decision layer is evaluated as a detector, never demoed.** FCW/AEB-request logic reports true-positive rate **and false positives per hour of video**, at a named TTC threshold. FP rate is the number that matters: a phantom brake at highway speed is itself the crash.
8. **Trackers are measured, not eyeballed.** Report MOTA / IDF1 / ID-switches, and land a SORT comparison. "The boxes look stable" is not a result.
9. **Lane detection ships with its failure set.** Those frames go into `failure-modes.md` with images; the departure metric is scored on a labeled set, never one hand-picked clip.
10. **No number without its context.** Every latency row carries hardware, resolution, batch size, N frames, warmup excluded, and **p50/p95/p99**. Every accuracy row carries dataset, split, and N.
11. **Every speedup names its baseline** and records the full ladder with a **paired accuracy delta** on the same split.
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
| Labelled boxes | dev 385 · val ~8.7k · test unaudited by design |
| Intrinsics (cam2, rectified) | `fx = fy = 721.5377 px` · `cx = 609.5593` · `cy = 172.8540` · `1242 × 375` · HFOV **81.4°** |
| rect0 → cam2 offset | `[+0.0598, −0.0004, +0.0027] m` (z ≈ 0 ⇒ depths interchangeable) |
| Frame rate | **9.657 Hz** (dt = 103.56 ms ± 0.06), measured — *not* the documented 10 Hz ([D-006](docs/decisions.md)) |
| `range_m` convention | Longitudinal, **near face** ([D-003](docs/decisions.md)). Centroid runs **+2.03 m** farther |
| Hardware | Apple M2, macOS (darwin 25.5), Python 3.12.4. **No CUDA** — MPS or CPU |
| **GT ruler** (`shrink_p20`, usable tier) | MAE **0.13 m**, bias +0.09, p95 0.37 · by bin **0.05 / 0.08 / 0.15 / 0.18 / 0.22 m** |
| GT coverage | **64%** of labelled boxes are groundtruthable; **85%** of those yield a measurement |
| Detection mAP (model + input size) | TBD |
| **Range error by bin** (0–10 / 10–20 / 20–30 / 30–50 / 50+ m) | TBD / TBD / TBD / TBD / TBD |
| **Credible operating envelope** (range where error < X%) | TBD ← *the project's headline number* |
| Closing-speed error vs GT | TBD |
| TTC error where it matters (TTC < 3 s) | TBD |
| Tracking MOTA / IDF1 / ID-switches — IoU vs SORT | TBD |
| Lane departure detection rate / FP rate (labeled set) | TBD |
| FCW: TPR and **FP per hour**, at threshold TTC = TBD | TBD |
| Per-stage latency p50/p95/p99 | TBD |
| Test suite count | **38** |

## Common commands (run from repo root)

```bash
# ── Data
python tools/fetch_kitti.py --split dev val        # fetch + unpack + prune
python tools/fetch_kitti.py --check                # verify what is on disk
python tools/audit_labels.py                       # label density per drive
python tools/audit_labels.py --probe 0059 0084     # 1.5 MB probe before a 1.5 GB commit

# ── Ground truth (Phase 1 — the ruler)
python tools/build_range_gt.py --split dev         # characterise the ruler, write the table
python tools/render_frame.py --drive 2011_09_26_drive_0013 --frame 20 \
    --out docs/figures/phase0_projection_check.png # standing calibration check

# ── Evaluate (always before performance or visual work)
python eval/eval_range.py    --split val --bins 10,20,30,50   # → error-vs-range curve
python eval/eval_tracking.py --split val --tracker iou|sort
python eval/eval_ttc.py      --split val
python eval/eval_fcw.py      --split val --ttc-threshold 2.0  # TPR + FP/hour
python eval/eval_lane.py     --split val

# ── Benchmark (latency; writes a fully-contextualized row)
python bench/run_bench.py --warmup 50 --iters 500 --report p50,p95,p99

# ── Replay / visualize (Phase 8 — the demo, not the evidence)
python app/replay.py --seq 0000 --hud --topdown

# ── Tests & lint
python -m pytest tests/ && ruff check . && black --check .
```

## Milestones

Ordering is deliberate: **the ruler is built first, the dashboard last.**

- [x] **Phase 0 — Repo + dataset + one frame.** KITTI loader, calibration parsed, one frame rendered with boxes. Environment fingerprinted.
- [x] **Phase 1 — Ground truth harness.** Project LiDAR into the image; per-detection GT range, with the ruler's own error and coverage characterised by range bin.
- [ ] **Phase 2 — Detection baseline.** YOLOv8, mAP on the split, latency ladder. No optimization yet.
- [ ] **Phase 3 — Monocular range + error-vs-range curve.** Pitch handling decided and justified. **This curve is the headline artifact of the whole project.**
- [ ] **Phase 4 — Tracking.** IoU baseline → SORT. MOTA/IDF1/ID-switches for both, honestly compared.
- [ ] **Phase 5 — Closing speed + TTC.** Scale-rate estimator, KF over image-plane state, deadband. Validate on synthetic constant-velocity sequences first, then real data.
- [ ] **Phase 6 — Lane detection + departure metric.** Scored on a labeled set. Failure set documented with frames.
- [ ] **Phase 7 — FCW / AEB-request decision layer.** Evaluated as a detector: TPR **and FP/hour**.
- [ ] **Phase 8 — HUD + top-down view.** Deliberately last. It presents results that already exist.
- [ ] **Phase 9 — Writeup.** README leading with the error curve and the operating envelope; `claims.md` generated from `benchmarks.md`; failure modes published, not buried.

## Current status

> Keep SHORT (≤ 15 lines). `/end-session` updates it; the narrative goes to `docs/history.md`.

- **Phase:** **Phase 1 COMPLETE** (2026-09-01). Phase 0 ✅. **Next: Phase 2 — detection baseline** (YOLOv8 mAP on val + latency ladder on M2). The v1 demo pipeline is archived in `legacy/` and is *not* a baseline ([D-008](docs/decisions.md)).
- **Verified:** projection chain confirmed by overlay + 11 closed-form tests · **38 tests, ruff + black clean** · dev split ruler characterised end to end.
- **The ruler (Phase 1 headline).** `shrink_p20`, usable tier: MAE **0.13 m**, bias +0.09 m, p95 0.37 m; by bin **0.05 / 0.08 / 0.15 / 0.18 / 0.22 m** across 0–10 → 50+ m. Coverage: **64%** of labelled boxes are groundtruthable at all, **85%** of those yield a measurement. That 0.13 m is the floor under every range number this project will ever report — roughly **20× tighter** than the monocular error Phase 3 expects to find, which is what makes it usable as a ruler.
- **Negative result banked** ([D-011](docs/decisions.md)): an adaptive shrink fallback raised coverage 86% → 100% and the recovered boxes carried **12× the error** (1.56 m vs 0.13 m), poisoning a whole range bin. Reverted; abstention is now enforced by a test.
- **Known issues:** pitch handling still undecided (blocks Phase 3 — [D-009](docs/decisions.md)) · `0027` has only 69 labelled boxes and cannot carry a binned number alone · no CUDA on this machine, so the TensorRT-style latency ladder in the charter is not reproducible here · test split deliberately unaudited.
- **Open decisions:** see the table at the foot of [decisions.md](docs/decisions.md) — pitch estimation vs characterisation, contact-point vs class-size prior, TTC threshold and deadband.
- **Standing warning:** draft resume bullets exist describing **Camera-LiDAR fusion in C++/CUDA with TensorRT and ROS/Gazebo**. That is not this system and shares no component with it. Claims get generated from `benchmarks.md` in Phase 9.

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
