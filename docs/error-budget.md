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
| Ground-plane non-flatness | Unquantified | Contact-point geometry assumes a flat road |
| Intrinsics error | Negligible | KITTI calibration; not re-estimated here |

### Stage 1b — Pitch · **PARTLY MEASURED**

Flat-ground contact-point geometry assumes zero camera pitch. A pitch error `θ`
displaces the apparent contact row, and for a camera at height `H_cam` the range
error is approximately `δD/D ≈ D·θ/H_cam` for small `θ` — i.e. **pitch error also
grows with range.**

**Measured on val ([D-009](decisions.md)) — this exceeds the threshold:**

| drive | mean | peak-to-peak |
|---|---|---|
| 0059 | −0.257° | 1.117° |
| **0084** | **+1.026°** | **2.011°** |
| **0091** | +0.423° | **2.221°** |
| 0056 | +0.425° | 1.126° |
| 0027 | +0.390° | 0.471° |

The charter's stated sensitivity is that **2° alone exceeds 10% range error**.
Two of five val drives *reach or exceed* that, on ordinary city driving with **no
hard braking** — and the ego vehicle pitches most during exactly the braking
manoeuvre an FCW system exists for.

`0084`'s **sustained +1.026° mean** is the more dangerous number: a standing
offset across a whole drive is a systematic range bias, not an excursion, and no
temporal filter removes it. At 30 m with a 1.65 m camera height, ~1° of
uncorrected pitch is on the order of a 30% range error — **two orders of
magnitude larger than the 0.25 m ground-truth budget.** Dev's 1.05° understated
this.

**Not yet corrected, and deliberately so.** OXTS reports *vehicle* pitch in the
navigation frame; the geometry needs *camera* pitch relative to the local road
plane, which differs by suspension travel, road grade, and mounting error.
Subtracting one for the other would look like a correction while correcting the
wrong angle. Phase 3 decides: estimate pitch from the horizon row, or characterise
the degradation with a measured sensitivity curve. **Silence is not an option.**

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
