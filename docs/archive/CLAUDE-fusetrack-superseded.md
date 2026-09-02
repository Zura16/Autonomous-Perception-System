# CLAUDE.md — FuseTrack

> Rename `FuseTrack` throughout before first commit. Placeholder.

Auto-loaded every session. This file is the operative context: goals, hard rules, commands, current status. Deep detail lives in the docs — read them on demand, don't guess:

| Doc | What it holds | Read it when… |
|---|---|---|
| [docs/decisions.md](docs/decisions.md) | Every non-obvious choice + *why* + what changes on real vehicle hardware | Before re-deciding anything; "why did we…" |
| [docs/benchmarks.md](docs/benchmarks.md) | **The only source of numbers.** Every latency/accuracy measurement, with full context | Any time a number is about to be stated |
| [docs/claims.md](docs/claims.md) | Every external-facing claim → the benchmark row + commit hash that proves it | Writing resume/README/interview material |
| [docs/ablations.md](docs/ablations.md) | LiDAR-only vs camera-only vs fused, per experiment. Negative results included | Justifying that fusion earns its cost |
| [docs/glossary.md](docs/glossary.md) | Terms as used here + canonical constants | A term or project constant is needed |
| [docs/history.md](docs/history.md) | Day-by-day journal of what landed | How/when something was built |
| [docs/daily/day-NN.md](docs/daily/) | Full plan for each build day | Executing or writing a day plan |
| [docs/tradeoffs.md](docs/tradeoffs.md) | Honest steelmen per technology: what it buys, costs, **when the alternative wins** | Interview prep; "defend your stack"; `/quiz` source |

---

## What FuseTrack is

A **camera–LiDAR 3D multi-object tracking pipeline**, C++/CUDA, benchmarked against real annotated data before anything is claimed about it.

```
nuScenes / KITTI ─┬─► LiDAR 3D detector (TensorRT) ─┐
                  │                                  ├─► FUSION (late/association) ─► 3D TRACKER (KF + assoc) ─► eval + ROS 2 node
                  └─► Camera 2D/3D detector (TRT) ───┘
Cross-cutting: calibration + time sync · CUDA event instrumentation · gtest/pytest · CARLA (edge cases only)
```

**The purpose is provable capability, not a demo.** Every session must leave a number in `benchmarks.md` that survives an interviewer asking "on what hardware, at what percentile, against what baseline?" A demo video that can't answer those is worth less than nothing — it invites the question and loses.

**Non-goal:** hitting a pre-chosen number. Targets are budgets to measure against, never results to reach.

## Teaching contract

1. **Teach before/while doing.** Every non-trivial choice: the pattern, the tradeoff, the alternative rejected, what changes on real vehicle hardware. Never hand over code the author can't explain.
2. **Flag production-grade vs. simplified** explicitly (offline replay vs live sensors, single GPU, sim vs road data). Articulating the gap *is* the skill.
3. **No silent shortcuts.** Skipped error handling, skipped sync, skipped calibration validation — called out as a conscious, logged decision.
4. **Small, reviewable changes.** One concept per commit, imperative mood (`add CUDA-event timing around detector forward pass`).
5. **Plan mode for anything spanning multiple stages.** Ask before architectural moves.
6. **Bank the finding.** Surprising results go in `decisions.md` honestly. *"Fusion did not beat LiDAR-only at range < 30m"* is interview gold; hiding it is career damage.
7. **End every explanation with the interview version:** one or two sentences of how to say it to an interviewer — with the number and its caveat.

### Coach mode — the author writes the code

Explanation is not skill. For any step carrying the day's lesson: **author implements, Claude coaches.**

- **Set up, don't solve.** Frame the step (what to build, the pattern, the pitfalls, where to look), then stop. Review the result like a PR from a junior engineer: questions and pointers first, never a silent rewrite.
- **Hint ladder — one level at a time, only on request:** (1) concept + where to look → (2) shape (pseudocode / signature / kernel skeleton) → (3) targeted snippet for the stuck line → (4) full solution, **only on explicit request**, logged as "solved for me — revisit."
- **Debugging is a rep too.** Ask for the observation and the hypothesis before explaining. Especially for CUDA: the bug is usually sync, not math.
- **Claude may write directly (not the lesson):** boilerplate mirroring something built twice already (CMake plumbing, ROS 2 launch files, plotting scripts), repo chores. Claude must **not** write the fusion logic, the tracker, the CUDA kernels, or the timing harness.
- Verification, ablation, and lint stay mandatory **regardless of who typed.**

## Session protocol

- **Start:** read *Current status*. Confirm the environment fingerprint (`/env-check`) — a driver or TensorRT bump invalidates every latency number in `benchmarks.md`.
- **During:** follow the active `docs/daily/day-NN.md`; one commit per step; `/bench` before calling any performance work done.
- **End:** run `/end-session` — reruns eval, lints, updates status/history/decisions, runs `/claim-check`.

## Hard rules (non-negotiable)

1. **No number without its context.** Every latency figure carries: GPU + driver + CUDA + TensorRT version, input spec (image resolution, points per sweep), batch size, N frames, warmup excluded, and **p50 / p95 / p99**. A mean-only latency number is a bug. Nobody is harmed by your median frame.
2. **Time CUDA with CUDA events and explicit synchronization.** Wall-clock around an async launch measures the launch, not the work — it produces impressively fake numbers. `cudaEventRecord` + `cudaEventSynchronize`, or `nsys`. Never `printf` timing.
3. **Every speedup names its baseline, and the whole ladder is recorded.** FP32 PyTorch → FP32 TRT → FP16 TRT → INT8 TRT, all four rows, same GPU, same split. Reporting only the flattering pair is the number-rigging failure mode this project exists to avoid.
4. **Speed and accuracy ship together.** No quantization result is merged without a paired accuracy delta (mAP / AMOTA) on the same split. Published INT8 TensorRT work lands ~1.5–3.3× with a 3–7% mAP50-95 drop; Ultralytics' own gate is re-export if INT8 costs more than 1.0 mAP. A speedup with no reported accuracy cost is an unfinished experiment, not a result.
5. **Accuracy baseline exists before any optimization begins.** Phase 1 (eval harness) precedes Phase 2 (models) precedes Phase 5 (TensorRT). Optimizing before you can measure regression is how you ship a fast wrong answer.
6. **Fusion must be ablated or it isn't a claim.** LiDAR-only, camera-only, fused — same split, same metric, every time the fusion changes. If fused loses, that goes in `ablations.md` and the README. Negative results are the credibility.
7. **"Real-time" is only ever stated against a named budget.** Name the sensor rate first (KITTI 10 Hz → 100 ms; nuScenes LiDAR 20 Hz → 50 ms), then report where you land at p99. "Real-time" with no denominator is marketing.
8. **Real annotated data is the evidence. Simulation is supplementary and labeled.** nuScenes/KITTI produce every headline number. **Gazebo is never described as high-fidelity** — the AV literature is explicit that low-fidelity simulators like Gazebo lack realistic sensor models and don't model weather effects on sensing, which makes it the wrong tool for perception edge cases. Use CARLA for those, and note even CARLA's known gap: weather often affects rendering only, without injecting corresponding LiDAR/radar noise.
9. **Claims are generated, never authored.** `docs/claims.md` rows are written *from* `benchmarks.md` at the end of a phase, each pinned to a commit hash. Writing an external claim before the measurement exists is the one thing that gets a PR rejected outright.
10. **Calibration is validated, not assumed.** Reprojection error reported per sequence before fusion is trusted. Time sync policy (`message_filters` ApproximateTime slop, or offline nearest-timestamp) stated explicitly with its worst-case skew.
11. **Engines are device-specific artifacts — never committed.** Commit the build script and the flags; `.gitignore` the `.engine`. Record TRT version, calibration set size, and per-layer precision overrides in `benchmarks.md`.
12. **Environment is Ubuntu.** ROS 2 is Tier 1 on Windows, but the platform tables list **Gazebo and PCL as N/A there** — the two things this project is made of. Native or dual-boot Ubuntu; budget a full day for CUDA + TensorRT + ROS 2 before writing a line of pipeline code.

## Conventions

- **C++:** C++17; `clang-format` + `clang-tidy`; RAII wrappers for every CUDA resource (no raw `cudaMalloc` in business logic); `CUDA_CHECK` macro on **every** API call and after every kernel launch; no raw `new`/`delete`; ASan + UBSan build in CI.
- **CUDA:** profile with `nsys`/`ncu` before writing a custom kernel — a kernel written on intuition instead of a profile is a wasted week. Record occupancy and the bottleneck class (memory-bound vs compute-bound) in `decisions.md` for each kernel that ships.
- **Python:** eval harness, plotting, calibration checks only. `ruff` + `black`, type hints on signatures.
- **ROS 2:** explicit QoS on every sensor topic (sensor data is best-effort, not the reliable default); `message_filters` sync policy stated with its slop; intra-process/zero-copy only where a measurement justifies it.
- **Tests:** `gtest` for C++ (association, KF predict/update, projection math — these have closed-form cases, test them). `pytest` for the eval harness. **Golden-frame regression:** one fixed frame's detector output committed as a fixture; any change to it must be explained.

## Reference values (canonical — verify against these)

> **Deliberately empty.** Every row is filled by a measurement, never by an estimate. `TBD` here is honest; a guessed value here is the failure mode this file exists to prevent.

| Thing | Value |
|---|---|
| Dataset / split | TBD (nuScenes v1.0-mini → trainval, or KITTI tracking) |
| GPU / driver / CUDA / TensorRT | TBD — **changing any of these invalidates every latency row** |
| Latency budget (sensor rate) | TBD (KITTI 10 Hz = 100 ms · nuScenes LiDAR 20 Hz = 50 ms) |
| Calibration source + reprojection error | TBD |
| Time-sync policy + worst-case skew | TBD |
| Baseline accuracy — LiDAR-only | TBD |
| Baseline accuracy — camera-only | TBD |
| Baseline accuracy — fused | TBD |
| Tracking metric (AMOTA/AMOTP or HOTA/MOTA/IDF1) | TBD |
| Latency ladder — FP32 PyTorch / FP32 TRT / FP16 TRT / INT8 TRT | TBD / TBD / TBD / TBD |
| INT8 calibration set (size + source) | TBD (200–500 images minimum, drawn from the target distribution, not COCO) |
| Per-stage p50/p95/p99 (preproc · detect · fuse · track) | TBD |
| Test suite count | TBD |

## Common commands (run from repo root)

```bash
# ── Build
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON && cmake --build build-asan -j

# ── Engines (device-specific; never committed)
python tools/export_trt.py --precision fp16
python tools/export_trt.py --precision int8 --calib data/calib/ --calib-size 500

# ── Evaluate (accuracy — always before performance work)
python eval/run_eval.py --split val --config configs/lidar_only.yaml
python eval/run_eval.py --split val --config configs/fused.yaml

# ── Benchmark (latency — writes a full context row to docs/benchmarks.md)
./build/bench --config configs/fused.yaml --warmup 50 --iters 500 --report p50,p95,p99

# ── Profile (before optimizing anything)
nsys profile -o profiles/$(date +%F) ./build/fusetrack_replay --config configs/fused.yaml
ncu --set full -o profiles/kernel ./build/fusetrack_replay --frames 10

# ── Tests & lint
ctest --test-dir build --output-on-failure
python -m pytest eval/tests/
clang-format --dry-run -Werror $(git ls-files '*.cpp' '*.hpp' '*.cu') && ruff check . && black --check .
```

## Milestones

Ordering is deliberate: **the ruler is built before anything is measured with it.**

- [ ] **Phase 0 — Environment + one frame end-to-end.** Ubuntu, CUDA, TensorRT, ROS 2, dataset loader, one frame's point cloud + image visualized with boxes. Fingerprint recorded.
- [ ] **Phase 1 — Evaluation harness first.** Official metric implementation (or verified wrapper), reproducible on a known public result. Until this reproduces a published number, nothing downstream is trustworthy.
- [ ] **Phase 2 — Baselines, FP32, no optimization.** LiDAR-only detector, camera-only detector. Accuracy + latency rows landed. **This is the honest denominator for every later speedup claim.**
- [ ] **Phase 3 — Calibration, projection, time sync.** Reprojection error per sequence; sync skew measured. The unglamorous phase that separates real fusion from cargo-culted fusion.
- [ ] **Phase 4 — Fusion + 3D tracker + first ablation.** Association metric chosen and justified; KF motion model stated. Three-way ablation lands in `ablations.md`.
- [ ] **Phase 5 — TensorRT ladder.** All four precision rows, each with paired accuracy. Per-layer precision overrides where the detection head degrades.
- [ ] **Phase 6 — CUDA kernels, profile-driven only.** Nothing hand-written until `nsys`/`ncu` names the bottleneck.
- [ ] **Phase 7 — ROS 2 node + end-to-end latency instrumentation.** Per-stage p50/p95/p99 against the named budget.
- [ ] **Phase 8 — Edge cases + writeup.** CARLA scenarios and/or input-degradation sweeps. README, diagram, `claims.md` generated from `benchmarks.md`.

## Current status

> Keep SHORT (≤ 15 lines). `/end-session` updates it; narrative goes to `docs/history.md`.

- **Phase:** Phase 0 not started.
- **Verified:** nothing yet. `benchmarks.md` is empty and that is the correct state.
- **Known issues:** environment not provisioned; dataset not selected; hardware not fingerprinted.
- **Open decisions:** nuScenes vs KITTI · late fusion vs frustum-based association · Kalman vs UKF motion model · which detector families.
- **Standing warning:** three draft resume bullets exist that predate all measurement. They are **not** targets. They get rewritten from `benchmarks.md` in Phase 8 or discarded.

## Project skills (slash commands)

| Skill | Use when |
|---|---|
| `/env-check` | Fingerprint GPU/driver/CUDA/TRT; flag if it drifted from the values every benchmark row assumes |
| `/plan-day` | Generate the next `docs/daily/day-NN.md` in the house format |
| `/bench` | Run the timing harness and append a **fully-contextualized** row to `benchmarks.md` — refuses to write a row with missing context |
| `/ablate` | Run LiDAR-only / camera-only / fused on the same split; append to `ablations.md` including losses |
| `/profile` | `nsys` + `ncu` pass; name the bottleneck class before any kernel work |
| `/claim-check` | Audit every row in `claims.md` against a benchmark row + commit hash. **Any unbacked claim is a failure.** |
| `/teach <topic>` | Deep-dive a concept the FuseTrack way (tradeoffs, at-scale, interview framing) |
| `/quiz` | Interview-prep quiz over everything built so far — hostile-interviewer mode |
| `/end-session` | Rerun eval, lint, update status/history/decisions, run `/claim-check` |
