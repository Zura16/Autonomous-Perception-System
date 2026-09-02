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
