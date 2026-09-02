# Tradeoffs

An honest steelman per component: what it buys, what it costs, and **when the
alternative wins**. Source material for `/quiz` and for any "defend your design"
question.

A component with no "when the alternative wins" row has not been thought about.

---

## Monocular camera vs stereo vs radar/LiDAR

**Buys:** one cheap sensor, no extrinsic calibration between sensors to maintain,
no time-sync problem. Rich semantics — class, lane markings, traffic lights.

**Costs:** range is *inferred*, not measured. Error grows ~D² with box-height
error, and the geometry rests on assumptions (flat ground, zero pitch, known
object height) that are all violated in exactly the situations that matter.

**When the alternative wins:** essentially always, for range. Production AEB
fuses radar precisely because radar measures range and range-rate directly and
degrades gracefully in weather. **Stereo was available in this dataset and was
deliberately discarded** ([D-007](decisions.md)) — keeping it would have given
better range and dissolved the question the project exists to answer.

**Interview version.** "Monocular range is the weakest link and I can quantify how
weak. Production stacks fuse radar because radar measures range-rate directly;
I discarded the stereo pair KITTI ships so the monocular limits would be the
thing under measurement."

---

## LiDAR-in-box ground truth vs 3D box labels vs stereo depth

**Buys:** ground truth for *any* box, including detector outputs that have no
label. Centimetre-class sensor accuracy. Measured here at 0.13 m MAE.

**Costs:** a 2D box is a poor container — it admits background past the
silhouette, returns through glass, and any occluder. Requires a validity gate,
which costs 36% of labels.

**When the alternative wins:** 3D box labels alone are better when only labelled
objects matter and occlusion handling is someone else's problem — they are amodal
and unaffected by occlusion. They cannot ground-truth an unlabelled detection,
which is why they are used here as a *cross-check* on the ruler, not as the ruler.

---

## Near-face vs centroid range convention

**Buys (near face):** matches the physics of collision (the gap closing to zero)
and matches what a bounding box actually generates (the visible face).

**Costs:** differs from most published monocular-distance work by **+2.03 m**
(measured), so external comparisons need explicit conversion.

**When the alternative wins:** comparing against published numbers, or when the
downstream consumer wants object position rather than gap. Both are computed;
neither is substituted silently ([D-003](decisions.md)).

---

## Abstention vs coverage

**Buys:** every reported value is one the evidence supports. Coverage becomes a
reported quantity instead of a hidden assumption.

**Costs:** gaps. 64% GT coverage, and a range curve measured only on the visible,
untruncated subset — a lower bound rather than the real number.

**When the alternative wins:** when a downstream consumer *must* have a value for
every object and can act on a confidence score. Then the right design is
value + variance, not value + silence. The wrong design — the one measured and
reverted here — is a value with the uncertainty discarded ([D-011](decisions.md)).

---

## KITTI raw vs KITTI tracking benchmark

**Buys (raw):** per-drive downloads that fit the disk, with synchronised images,
velodyne, calibration, OXTS ego state, and per-frame 3D boxes with track IDs.

**Costs:** tracking metrics are **not comparable to the KITTI tracking
leaderboard** — different sequences, different annotation packaging, no
`DontCare` regions.

**When the alternative wins:** any claim of leaderboard-relative standing. That
requires the official split and ~50 GB ([D-004](decisions.md)).

---

## Sequence-level vs frame-level splits

**Buys:** no leakage. Adjacent frames at ~10 Hz are near-duplicates.

**Costs:** coarse. Five drives means one unusual drive skews a split, and val is
city-dominated.

**When the alternative wins:** never, for this data. Frame-level leakage would be
invisible and unquantifiable; a coarse split is at least arguable
([D-002](decisions.md)).
