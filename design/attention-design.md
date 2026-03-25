# Visual Attention System Design

## 1. Overview

The visual attention system determines **where the agent looks** each step. It takes
raw visual features from all perception extractors, builds a saliency map, optionally
attenuates recently attended locations, finds peaks, and selects one focus point for
object resolution.

### Pipeline Position

```
FeatureExtractors (9 total)
  | emit VisualFeature + Settled on perception bus
  v
VisionAttention (roc/attention.py)
  | builds SaliencyMap from features
  | calls SaliencyMap.get_focus()
  | emits VisionAttentionData on attention bus
  v
ObjectResolver (roc/object.py)
  | takes focus_points.iloc[0] -- the single winning focus point
  | resolves features at that location to an Object
  v
Sequencer -> Transformer -> Predict -> Action
```

Only **one focus point** is resolved per step. The rest are discarded.

### Key Files

- `roc/attention.py` -- VisionAttention component, SaliencyMap, get_focus()
- `roc/saliency_attenuation.py` -- attenuation ExpMod implementations
- `roc/object.py` -- ObjectResolver.do_object_resolution() (consumer of focus points)

---

## 2. SaliencyMap

`SaliencyMap` (`roc/attention.py:73`) is a 2D numpy grid where each cell holds a
`list[VisualFeature]`. Feature extractors populate it: for each feature, every cell
the feature covers gets that feature appended to its list.

A feature extractor like `Flood` produces one feature per contiguous region, but that
feature gets added to **every cell in the region**. So floor cells, wall cells, and
corridor cells all get features from Flood, Line, Color, Shape, etc. -- not just cells
with unique characters.

### Strength Calculation

`get_strength(x, y)` computes per-cell saliency (`roc/attention.py:159`):

```
strength = len(feature_list)     # base: 1 per feature covering this cell
         + 10 per Single feature  # unique character at this cell
         + 15 per Delta feature   # cell content changed from last step
         + 20 per Motion feature  # changed AND was non-empty before
```

Typical values from a real starting screen (max_strength = 20):

| Cell type | Features present | Raw strength | Normalized |
|-----------|-----------------|-------------|------------|
| Floor (.) in a room | Flood, Line, Color, Shape (no Single) | 4 | 0.034 |
| Wall (-) segment | Line, Color, Shape | 3 | 0.041 |
| Player (@) | Single, Color, Shape | 13 | 1.000 |
| Pet (d) | Single, Color, Shape | 13 | 1.000 |
| Stairs (<) | Single, Color, Shape, Flood | 14+ | 1.000 |

Key observation: **every visible cell has non-zero strength** because area-covering
features (Flood, Line, Color, Shape) reach all visible cells. This creates a connected
non-zero "floor" across the entire visible map.

### Real Data (step 1 of a game)

From a real game run with all feature extractors:

```
Feature report: {'Flood': 4, 'Line': 123, 'Single': 8, 'Color': 8, 'Shape': 8}
Max strength: 20
Non-zero cells: 1659 / 1659     (every cell in the 21x79 grid)
fkimg.min(): 0.050000
Unique strengths: [0.05, 0.10, 0.15, 1.0]
  0.05:    8 cells   (cells with 1 feature)
  0.10:   49 cells   (cells with 2 features)
  0.15: 1594 cells   (floor/wall cells with ~3 features from Line/Color/Shape)
  1.00:    8 cells   (Single-detected entities: @, d, -, etc.)
```

The 0.15 "floor" of 1594 cells is the connected background that the morphological
reconstruction will flood through.

---

## 3. get_focus() Algorithm

`SaliencyMap.get_focus()` (`roc/attention.py:201`) runs three stages:

### Stage 1: Build normalized strength image (fkimg)

```python
fkimg[x, y] = get_strength(x, y) / max_strength
```

This produces a 2D float array with values in [0, 1], indexed as `[x][y]` (width-first).
Note: this is transposed from numpy's usual [row][col] convention.

### Stage 2: Attenuation (inhibition of return)

```python
attenuation = SaliencyAttenuationExpMod.get(default="none")
pre_peak = np.unravel_index(np.argmax(fkimg), fkimg.shape)
fkimg = attenuation.attenuate(fkimg, saliency_map)
post_peak = np.unravel_index(np.argmax(fkimg), fkimg.shape)
```

The attenuation ExpMod modifies `fkimg` in place to reduce strength at recently
attended locations. `pre_peak` and `post_peak` record the argmax before and after
attenuation to detect whether attenuation shifted the brightest point.

**Important:** `post_peak` is logged to the dashboard but is NOT used for object
resolution. It is the argmax of the attenuated image -- a different value than
`focus_points.iloc[0]` which comes from Stage 3.

See Section 4 for attenuation ExpMod details.

### Stage 3: Morphological reconstruction peak-finding

```python
seed = np.copy(fkimg)
seed[1:-1, 1:-1] = fkimg.min()
rec = reconstruction(seed, fkimg, method="dilation")
peaks = fkimg - rec
```

Algorithm (from scikit-image):
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

1. **Seed**: Copy fkimg, set all interior pixels to fkimg.min(). Border pixels keep
   their original values.
2. **Reconstruction**: "Flood" inward from the borders by dilation. The flood rises
   from the border values and fills every cell it can reach without exceeding the
   original fkimg value. Think of it as pouring water from the edges -- it fills
   valleys and plateaus.
3. **Peaks**: Subtract the reconstruction from the original. Only cells that are
   **higher than the flood level** survive with non-zero values. These are isolated
   local maxima.

#### How it works on NetHack data

The floor cells at 0.15 form a connected background across the grid. The border
cells are also at 0.15 (visible map edges have Line/Color/Shape features). The
reconstruction floods from the border at 0.15 and propagates through all connected
cells at that level, reaching the entire floor.

Cells with Single features (strength 1.0) are above the flood. Subtracting:
`peaks = 1.0 - 0.15 = 0.85`. These survive as peaks.

Floor cells: `peaks = 0.15 - 0.15 = 0.0`. Eliminated.

#### Real data

```
Before reconstruction: 1659 non-zero cells
After reconstruction:  8 surviving peaks (the Single-detected entities)
```

All 8 survivors have the same strength (1.0), so the sort is an **arbitrary tiebreak**
among equals.

#### Connected component labeling

After finding peaks, adjacent non-zero cells are grouped:

```python
structure = np.ones((3, 3), dtype=int)  # 8-connectivity
labeled, ncomponents = label(peaks, structure)
```

Each group gets a unique label. The focus_points DataFrame includes this label.

#### DataFrame construction

```python
df = pd.DataFrame({"x": ..., "y": ..., "strength": ..., "label": ...})
    .sort_values("strength", ascending=False)
```

Sorted by strength descending. `iloc[0]` is the strongest peak. When multiple
peaks share the same strength (common -- all Single-detected cells are 1.0),
the tiebreak is arbitrary (depends on numpy's nonzero() ordering).

---

## 4. Attenuation ExpMods

All attenuation operates on fkimg (the normalized strength image) before
morphological reconstruction. Three implementations:

### 4a. NoAttenuation (name="none")

Returns fkimg unchanged. Default when no attenuation is configured.

### 4b. LinearDeclineAttenuation (name="linear-decline")

`roc/saliency_attenuation.py:115`

Implements inhibition of return inspired by Snyder & Kingstone (2000).

**State:** FIFO buffer of last N attended locations (`_history`), each with (x, y, tick).

**Parameters** (from Config):
- `capacity`: max history entries (default 5)
- `radius`: Manhattan distance radius (default 3)
- `max_penalty`: peak penalty multiplier (default 1.0)
- `max_attenuation`: floor on attenuation, i.e. min multiplier (default 0.9)

**Algorithm:** For each non-zero cell in fkimg:
1. Sum penalties from all history entries:
   - `recency_weight = (N - i) / N` where i=0 is most recent
   - `spatial_weight = max(0, 1 - manhattan_distance / radius)`
   - `penalty += max_penalty * recency_weight * spatial_weight`
2. Apply: `fkimg[x, y] *= max(1 - max_attenuation, 1 - penalty)`

Effect: cells near recently attended locations get their strength reduced. The
most recent location gets the strongest penalty. Cells beyond `radius` Manhattan
distance are unaffected.

**notify_focus()**: Records `focus_points.iloc[0]` (the winning morphological peak)
in the history buffer.

#### Real data (linear-decline, steps 1-5)

```
Step 1: 0 cells changed (empty history). pre_peak = post_peak = '-' at (63,6)
Step 2: 12 cells changed. post_peak = '@' at (66,8)
Step 3: 25 cells changed. pre_peak = '.' at (66,8) str=1.0
        Attenuation reduced (66,8) from 1.0 to 0.1 (recent focus)
        post_peak shifted to '.' at (67,10) str=0.66
Step 4: 34 cells changed. post_peak = '@' at (65,8) str=0.56
Step 5: 35 cells changed. pre_peak = '@' at (66,9) str=1.0
        Attenuation reduced to 0.1. post_peak shifted to '-' at (63,11) str=0.34
```

Attenuation successfully shifts focus away from recently attended locations.

### 4c. ActiveInferenceAttenuation (name="active-inference")

`roc/saliency_attenuation.py:390`

Discrete-state active inference agent (Parr & Friston 2017). Maintains per-location
beliefs about hidden states and attenuates based on epistemic value (entropy).

**State:**
- `_beliefs`: dict mapping (x, y) -> LocationBelief (categorical distribution over states)
- `_vocab`: StateVocabulary mapping feature sets to state IDs
- `_omega`: global volatility estimate

**Algorithm:** For each tracked location:
1. Compute entropy of belief distribution
2. `epistemic_mult = (1 - max_attenuation) + max_attenuation * (entropy / max_entropy)`
3. `effective_mult = (1 - saliency_weight) * epistemic_mult + saliency_weight`
4. `fkimg[x, y] *= effective_mult`

Low entropy (known state) -> strong attenuation. High entropy (uncertain) -> weak
attenuation. Unobserved locations gain entropy over time via volatility-weighted
propagation, naturally drawing attention back to stale observations.

---

## 5. post_peak vs focus_points[0] Divergence

### The problem

`post_peak` (argmax of attenuated fkimg) and `focus_points.iloc[0]` (top
morphological peak) can be at different locations. ObjectResolver uses
`focus_points.iloc[0]`. The dashboard displays `post_peak`.

### When they diverge

**Tiebreak among equals:** When multiple cells share the maximum strength after
attenuation (common at step 1 when all Single-detected cells are 1.0),
`post_peak` = argmax (first in memory order) while `focus_points.iloc[0]` =
first in the sorted DataFrame (arbitrary among equals after nonzero() ordering).

Real data from step 1:
```
post_peak:  (63, 6) '-' strength=1.0
focus[0]:   (68,11) '-' strength=1.0
Same? False   <-- both '-' at strength 1.0, different arbitrary picks
```

**Attenuation + multiple tied peaks:** After attenuation suppresses the previous
focus, multiple cells may tie for the new maximum.

Real data from step 5:
```
pre_peak:  (66, 9) '@' strength=1.0
  -> attenuation reduced to 0.1
post_peak: (63,11) '-' strength=0.34
focus[0]:  (68, 6) '-' strength=0.34
Same? False   <-- both '-' at 0.34, different tiebreak
```

### When they agree

When one cell has a uniquely higher strength than all others (e.g., player or
monster with Delta/Motion bonuses), both argmax and the morphological peak point
to the same cell.

Real data from step 2:
```
post_peak: (66, 8) '@' strength=1.0   (uniquely brightest after attenuation)
focus[0]:  (66, 8) '@' strength=1.0
Same? True
```

---

## 6. Origin and Suitability of Morphological Reconstruction

### Origin

The morphological reconstruction peak-finding algorithm was adopted from a
scikit-image tutorial for finding peaks in continuous grayscale images:
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

It was prototyped in the Jupyter notebook:
`experiments/2024.06.25-18.35.36-EXP-attention-debug/` (commit `ad62775`,
November 2024). The notebook experimented with `peak_local_max`, `maximum_filter`,
and `threshold_rel` before settling on morphological reconstruction.

### Why it worked in the notebook

The notebook tested on a real ROC saliency map where all feature extractors were
running. The non-zero "floor" of feature-bearing cells (strength ~0.07 from
Flood/Line/Color/Shape features) connected the room interior to the grid borders.
Reconstruction flooded through this connected floor and subtracted it, leaving
only elevated peaks (Single-detected entities with bonuses).

### Current behavior

With the full ROC pipeline, reconstruction works correctly: the 0.15 floor connects
to the borders, gets subtracted, and the 8 Single-detected entities survive.

The problem is the **tiebreak**: all Single-detected cells often have the same
strength (1.0 normalized), so `focus_points.iloc[0]` is an arbitrary pick among
them. The argmax (`post_peak`) is a different arbitrary pick.

### Potential concern

If a room is surrounded entirely by unexplored (dark) cells with zero strength,
the border of fkimg would be zero, the seed interior would also be zero, and
reconstruction would flood at zero -- meaning it could never reach the room interior.
All room cells would survive as "peaks," and reconstruction would be a no-op.

In practice this doesn't happen because Line features span the full grid width/height,
giving border cells non-zero strength even when unexplored. But it's worth noting
that the algorithm depends on this connected floor reaching the borders.

---

## 7. Attention Spread Metric

The dashboard's Visual Attention tab shows a "Spread" metric tracking how much of
the game world the attention system has examined.

**Numerator:** Cumulative count of unique NLE glyph IDs that have been at the top
focus point across all steps. Grows when attention focuses on a glyph type it hasn't
attended before.

**Denominator:** Cumulative count of unique NLE glyph IDs (excluding background
S_stone = glyph 2359) that have appeared anywhere on screen across all steps. Grows
when new glyph types appear.

**Display:** `Spread: x/y (z%)`

Computed in `roc/gymnasium.py:_inject_attention_spread()` using two module-level
sets (`_attended_glyphs`, `_seen_glyphs`). The glyph at the top focus point is
looked up via `obs["glyphs"][fy, fx]` where (fx, fy) are the focus point coordinates
in map space (21x79), which is the same coordinate system as `obs["glyphs"]`.

### Real data progression (game 1, 4648 steps)

```
Step     1:  1/12  (8.3%)   first glyph attended, 12 types on starting screen
Step    10:  7/14  (50.0%)  rapid early growth
Step    50: 11/16  (68.8%)
Step   100: 12/16  (75.0%)
Step  1000: 15/18  (83.3%)
Step  4000: 21/23  (91.3%)
Step  4648: 25/29  (86.2%)  dip: 6 new glyphs appeared near end, 4 unattended
```

---

## 8. Known Issues and Future Work

1. **Tiebreak:** When multiple cells share the same max strength, the winning
   focus point is arbitrary. A spatial or temporal tiebreaker (prefer closest to
   previous focus, prefer most recently changed) would make attention more
   deterministic.

2. **post_peak vs focus_points[0]:** The dashboard displays `post_peak` as if it's
   where attention went, but ObjectResolver uses `focus_points.iloc[0]`. These
   diverge when strengths are tied. Either the dashboard should show the actual
   winning focus point, or the resolution should use `post_peak`.

3. **Single focus point:** Only one focus point is resolved per step. The TODO in
   `do_object_resolution()` notes that resolving multiple focus points would improve
   coverage. The morphological peak-finding already produces N peaks -- they're just
   not used.

4. **notify_focus() uses focus_points.iloc[0]:** Both LinearDeclineAttenuation and
   ActiveInferenceAttenuation record `focus_points.iloc[0]` in their history, not
   `post_peak`. If the resolution switches to use `post_peak`, the attenuation
   history should follow.
