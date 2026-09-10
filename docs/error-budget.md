# Error budget

How error propagates: **pixel → range → velocity → TTC → decision.**

Read this before any change touching geometry, filtering, or thresholds. The
stages are coupled: a one-pixel change in box height at 50 m is a metre of range,
which is a phantom closing speed, which is a false brake request.

Sections marked **MEASURED** carry a number from [benchmarks.md](benchmarks.md).
Sections marked **DERIVED** are analysis with no measurement behind them yet, and
say so — the whole point of this file is that the two are never confused.

---

## Stage 0 — The ruler's own error · **MEASURED**

Nothing downstream can be measured more precisely than the ruler measuring it.

| | |
|---|---|
| Estimator | `shrink_p20` + spread gate, on projected LiDAR |
| MAE | **0.25 m** (median 0.17, p95 0.63) — **val**, N=3315 |
| By bin | 0.21 / 0.24 / 0.25 / 0.27 / **0.78** m |
| Residual failure rate | 0.27% at \|e\| > 5 m |
| Coverage | **38.1%** of labelled objects end with a ground-truth range |

**Usable because it is ~10× tighter than the monocular error Phase 3 expects.**
If a monocular estimator ever reports errors approaching 0.25 m, that number is
measuring the ruler, not the estimator, and must be reported as a floor.

**Beyond 50 m the ruler is weakest exactly where it matters.** The 50+ bin
retains only 73% of measured boxes after gating and lands at 0.78 m — three times
the near-field figure, on 61 boxes. Any operating-envelope claim past 50 m rests
on that thin evidence and must say so.

**Three structural limits, not noise:**

- **Occlusion.** No ground truth exists for occluded objects; the box contains
  the occluder (MAE 11.86 m if you try). Every range number is therefore measured
  on the *visible, untruncated* subset and is a **lower bound** on real-world
  error ([D-010](decisions.md)).
- **Support falls as ~1/range².** 1568 returns per box at 0–10 m, 22 at 50+ m
  (val). The ruler weakens exactly where the estimator it measures weakens.
- **Ambiguity.** A box holding two surfaces has no single answer; the spread gate
  abstains on 5% of measured boxes, rising to 27% at 50+ m ([D-013](decisions.md)).

---

## Stage 1 — Pixel → range · **DERIVED, awaiting Phase 3**

The pinhole relation for a contact-point or size-prior estimator:

```
D = f · H / h          D range (m) · f focal (px) · H true height (m) · h box height (px)
```

Differentiating for a box-height error `δh`:

```
δD/D = −δh/h        and        h = f·H/D        so        δD ∝ D² · δh / (f·H)
```

**Range error grows with the SQUARE of range for a fixed pixel error.** This is
the single most important sentence in this document, and the reason hard rule 2
forbids a single pooled accuracy figure.

With `f = 721.54 px` and a pedestrian-scale `H = 1.7 m`, a **1 px** box-height
error costs approximately:

| range | box height | δD from 1 px |
|---|---|---|
| 10 m | 123 px | 0.08 m (0.8%) |
| 20 m | 61 px | 0.33 m (1.6%) |
| 30 m | 41 px | 0.73 m (2.4%) |
| 50 m | 25 px | 2.0 m (4.1%) |
| 80 m | 15 px | 5.2 m (6.5%) |

**DERIVED — this is arithmetic, not a measurement.** Real detector box jitter is
several pixels and is *not* independent of range; Phase 2 measures it, Phase 3
replaces this table with observed error.

### Error sources feeding `δh`

| Source | Status | Note |
|---|---|---|
| Detector box jitter | Phase 2 | The dominant term. Multiply the table above by the measured jitter in px |
| **Camera pitch** | Partly measured | See Stage 1b |
| Class-height prior variance | Phase 3 | Vehicle width alone spans >400 mm compact→SUV. A prior without a variance is a guess ([hard rule 6](../CLAUDE.md)) |
| **Ground-plane non-flatness** | **MEASURED** | The dominant far-range term. Road falls ~10 cm below the assumed plane by 50 m; predicts −0.6 m at 20–25 m and −2.5 m at 40–50 m, matching observation to ~0.1–0.4 m ([D-020](decisions.md)) |
| Intrinsics error | Negligible | KITTI calibration; not re-estimated here |

### Stage 1a — Ground-plane non-flatness · **MEASURED, and it dominates at range**

The flat-ground assumption is not a rounding error; it is the **largest single
term in the contact-point estimator beyond 20 m**.

A contact point at depth `D` on ground at real height `y_act` gives
`D_est = D·h_cam/y_act`, so the fractional bias is `h_cam/y_act − 1` — computable
outright from a measured road profile, with no fitting.

| depth (m) | measured road height | vs assumed 1.655 m | predicted bias |
|---|---|---|---|
| 5–8 | 1.668 | +0.013 | −0.05 m |
| 20–25 | 1.699 | +0.044 | −0.59 m |
| 30–40 | 1.741 | +0.086 | −1.73 m |
| 40–50 | 1.751 | +0.096 | −2.46 m |

Observed vehicle bias at those ranges is −0.50, −2.26 and −2.12 m: the road
profile accounts for most of it beyond 20 m ([D-020](decisions.md)).

**Consequence for the estimator.** Contact-point range at distance is limited by
the *road*, not by the camera or the detector. Improving it needs a road-profile
estimate from the image — a research problem, not a tuning exercise — which is
why the flat-ground assumption is *characterised* rather than corrected.

**Inside ~13 m the road explains almost nothing**, and a flat **+0.7 m** residual
remains, ~0.3 m of it vehicle-specific. That is the project's open question.

### Stage 1b — Pitch · **PARTLY MEASURED**

Flat-ground contact-point geometry assumes zero camera pitch. Assuming zero when
the truth is `θ` gives, **exactly**:

```
D_est = D / (1 − x)        x = tan(θ) · D / h_cam
δD/D  = x / (1 − x)
```

**This is superlinear in range, and it diverges** at `D = h_cam/tan(θ)` — the
range at which the contact point reaches the assumed horizon, where the estimator
abstains rather than returning a huge number.

The commonly quoted `δD/D ≈ x` is only the first-order expansion and understates
the cost badly where it matters. A unit test against a synthetically pitched
projection caught this being used as if it were exact:

| range | x | linearised | **exact** |
|---|---|---|---|
| 10 m | 0.075 | 7.5% | **8.2%** |
| 20 m | 0.151 | 15.1% | **17.8%** |
| 30 m | 0.226 | 22.6% | **29.2%** |
| 50 m | 0.377 | 37.7% | **60.5%** |

*(at the measured val p95 pitch of 0.715°, h_cam = 1.655 m)*

### The pitch magnitude, now measured correctly · **MEASURED**

Earlier drafts of this file used **OXTS vehicle pitch** and reported 2.01–2.22°
peak-to-peak with a sustained +1.03° offset. **That was the wrong quantity and it
overstated the real error by 2–3×.** OXTS reports vehicle attitude in the
*navigation* frame, which includes **road grade**: driving up a hill tilts the
vehicle while the camera stays aligned with the road surface it is looking at.

Fitting the road plane to LiDAR returns (`aps/groundplane.py`) measures the angle
the geometry actually needs — camera relative to the local road — in the right
frame ([D-016](decisions.md)):

| | camera-to-road pitch (val, N=134) |
|---|---|
| mean | **+0.150°** |
| std | 0.319° |
| \|pitch\| p95 | **0.715°** |
| \|pitch\| max | 0.937° |

The same fits give **camera height 1.655 m** (std 0.027, range 1.583–1.745),
which is the nominal the estimator now uses.

**The concern was right; the magnitude was wrong.** Even at 0.715°, the exact
relation above costs **29% at 30 m and 60% at 50 m** — so pitch remains a first-
order term, and the flat-ground assumption is expensive at range. It is simply
not the 2°-scale catastrophe the OXTS figures implied.

**Caveat that must travel with these numbers:** measured on drives containing no
hard braking. The ego vehicle pitches most under exactly the deceleration an FCW
system exists for, so this distribution is a **floor**, not the operational range.

**Decision: characterise, do not correct** ([D-016](decisions.md)). A horizon-row
pitch estimator would need validating against a reference, and OXTS is not that
reference. The measured distribution above turns the sensitivity analysis from a
hypothetical sweep into a statement about what the assumption actually costs on
this data.

---

## Stage 2 — Range → closing speed · **DERIVED, awaiting Phase 5**

**Never by differencing range.** With frame interval `Δt ≈ 0.1036 s`, finite
differencing turns a range error `δD` into a velocity error:

```
δv ≈ √2 · δD / Δt ≈ 13.7 · δD
```

A **0.5 m** range wobble becomes **6.9 m/s** — 25 km/h of phantom closing speed
on a parked car. This is why hard rule 3 mandates the scale relation
`Ḋ = −D·ḣ/h` with a Kalman filter over the image-plane state instead: the filter
smooths in the domain where the noise actually lives (pixels), and range, closing
speed, and TTC then derive from one consistent state and cannot contradict
each other.

**`Δt` must come from the timestamps.** Measured 9.657 Hz vs the documented
10 Hz is a **3.4% systematic** error in every velocity and TTC figure, and would
be indistinguishable afterwards from estimator bias ([D-006](decisions.md)).

---

## Stage 3 — TTC · **DERIVED, awaiting Phase 5**

```
TTC = −D / Ḋ
```

Two error terms, and the second dominates:

```
δTTC/TTC ≈ δD/D  ⊕  δḊ/Ḋ
```

Because `Ḋ` is the noisy quantity, **TTC error is dominated by closing-speed
error**, and it blows up as `Ḋ → 0` — i.e. TTC is least reliable exactly when an
object is *not* closing, which is the false-positive regime. Hence the mandatory
deadband ([hard rule 4](../CLAUDE.md)).

Constant-relative-velocity TTC is itself an approximation: it assumes neither
vehicle accelerates during the prediction horizon, which is false the moment
either one brakes. To be stated wherever a TTC number appears.

---

## Stage 4 — Decision · **DERIVED, awaiting Phase 7**

The FCW threshold is not a free parameter — it must be justified against the
measured error budget above:

- **TTC threshold too low** → warnings arrive too late to be actionable.
- **TTC threshold too high** → false positives, and a phantom brake at highway
  speed is itself the crash.

Evaluated as a detector: **TPR and FP/hour** at a named threshold, never demoed.
The threshold is set *after* Stage 1–3 are measured, and the ordering is the
argument: a threshold chosen first is a threshold chosen from vibes.

---

## Summary — where the error actually comes from

| Stage | Growth with range | Status |
|---|---|---|
| Ruler (LiDAR GT) | ~flat, 0.05 → 0.22 m | **MEASURED** |
| Pixel → range | **D²** for fixed pixel error | DERIVED |
| Pitch → range | **D** for fixed pitch error | pitch excursion MEASURED, coupling DERIVED |
| Range → velocity | ×13.7 if finite-differenced | DERIVED — avoided by design |
| Velocity → TTC | diverges as Ḋ → 0 | DERIVED |

**The expected headline:** monocular range error grows roughly quadratically, so
there is some range beyond which FCW is not credible. Finding that number
honestly — and stating it — is the project's deliverable.
