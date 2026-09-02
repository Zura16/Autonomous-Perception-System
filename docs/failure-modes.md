# Failure modes

Documented conditions under which each stage breaks, with frames. A stage
without a failure set has not been characterised — it has only been demoed.

Populated by `/failure-hunt`, which sweeps the split for worst-case frames rather
than waiting for one to be noticed.

---

## Ground truth (Phase 1)

### FM-1 · Occlusion: the estimator measures the occluder · **CONFIRMED**

**Condition:** a labelled object is partly or fully occluded by a nearer object
overlapping its 2D box.

**Behaviour:** the depth estimator returns a precise, confident, *wrong* value —
the occluder's range. Measured MAE **11.86 m** on fully-occluded boxes
(N=79, dev split) against 0.10 m on visible ones.

**Worst observed:** `drive_0013`, frames 100–104, track 8 — a car labelled at
52.8–57.6 m read as 21.9–26.6 m. The system was measuring the parked car in
front of it, with no signal that anything was wrong.

**Handling:** abstention via the GT validity tier ([D-010](decisions.md)).
Not fixable by a better estimator — you cannot range-find what you cannot see.

### FM-2 · Truncation: amodal label vs modal sensor · **CONFIRMED**

**Condition:** an object is clipped by the image border.

**Behaviour:** the label's 3D box is amodal and its near face may lie outside the
image, while LiDAR only sees the visible portion. The estimate reads **long** —
measured bias **+2.2 m**.

**Handling:** abstention. Same gate, different mechanism.

### FM-3 · Sparse support at range · **CONFIRMED**

**Condition:** LiDAR returns per box fall as ~1/range².

**Behaviour:** median returns per box drop from 1887 (0–10 m) to 24 (50+ m). Below
8 supporting returns the estimator abstains, so ground-truth coverage itself
degrades with range.

**Handling:** abstention plus a reported coverage column. The attempted
workaround — widening the sampling window — made things 12× worse
([D-011](decisions.md)).

### FM-4 · Two surfaces in one box: bimodal error beyond 50 m · **CONFIRMED**

**Condition:** a 2D box contains returns from the object *and* from something in
front of or behind it, without the annotator flagging occlusion.

**Behaviour:** the percentile reduction returns one surface with full confidence
and no signal about which. Error becomes **bimodal**: on val beyond 50 m, median
|error| 0.12 m but MAE 4.75 m — 83% of boxes under 1 m, 14% over 5 m, p99 66 m.

**Detection:** the interquartile depth spread separates the populations 47×
(0.248 m on correct boxes, 11.65 m on failures). Point count does **not** —
failures had *fewer* points than successes, so the intuitive gate is backwards.

**Handling:** abstain above 1.5 m spread ([D-013](decisions.md)). Residual failure
rate 0.27%, and 27% of 50+ m boxes are given up to achieve it.

### FM-5 · Ego pitch excursion and standing offset · **CONFIRMED**

**Condition:** ordinary city driving. No hard braking required.

**Behaviour:** measured peak-to-peak pitch of **2.011°** (`0084`) and **2.221°**
(`0091`), at or above the level stated to cost >10% range error. `0084` also
carries a **sustained +1.026° mean** — a systematic range bias across an entire
drive that no temporal filter removes.

**Handling:** undecided, and blocking Phase 3 ([D-009](decisions.md)). Either
estimate camera pitch from the horizon row or publish a measured sensitivity
curve. Note the dev drive showed only 1.05° and would have understated this.

---

## Detection (Phase 2) — not yet characterised
## Monocular range (Phase 3) — not yet characterised

Expected, from the error budget and to be confirmed or refuted with frames:
pitch excursion under braking, non-flat road, class-height prior variance,
truncated and small-box regimes.

## Tracking (Phase 4) — not yet characterised
## Lane detection (Phase 6) — not yet characterised

The charter names the expected set: shadows, faded markings, night, rain, sharp
curves, construction. Naming them is not characterising them; frames are required.

## FCW (Phase 7) — not yet characterised
