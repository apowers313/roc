# Saliency Attenuation ExpMod Design

## 1. Overview

This document describes the design for a `saliency-attenuation` ExpMod that implements inhibition of return (IOR) in the ROC attention pipeline. The ExpMod attenuates the strength image inside `SaliencyMap.get_focus()` before peak-finding, so that recently attended locations have reduced saliency and may naturally fall below the peak detection threshold.

### Design Principles

1. **Attenuation, not suppression.** The ExpMod reduces strength values in the normalized strength image. Peaks are not explicitly removed -- they may disappear naturally from the morphological dilation peak-finding step as an emergent property of reduced strength. Heavily attenuated locations will have lower (or zero) peaks; lightly attenuated locations will simply rank lower in the output.
2. **Integration inside `SaliencyMap.get_focus()`.** The ExpMod is called after the strength image is built but before the dilation-based peak detection. `VisionAttention`, `ObjectResolver`, and the rest of the pipeline are unchanged.
3. **Three flavors** selected via the standard ExpMod mechanism:
   - `none` (default) -- current behavior, no attenuation
   - `linear-decline` -- Snyder & Kingstone (2000) inspired linear decline over N locations
   - `active-inference` -- Parr & Friston (2017) discrete-state epistemic agent
4. **Game-agnostic.** The ExpMod operates on abstract numpy strength arrays and saliency grids. No game-specific knowledge.

---

## 2. Integration Point

### Where in the Pipeline

```
Perception -> VisionAttention -> SaliencyMap.get_focus()
                                       |
                                  build strength image (fkimg)
                                       |
                              >>> ATTENUATE fkimg HERE <<<   <-- ExpMod integration point
                                       |
                                  morphological dilation peak-finding
                                       |
                                  connected component labeling
                                       |
                                  focus_points DataFrame
                                       |
                                  AttentionEvent -> ObjectResolver (unchanged)
```

### Current Flow (attention.py:200-249)

```python
def get_focus(self) -> DataSet[VisionAttentionSchema]:
    max_str = self.get_max_strength()
    if max_str == 0:
        max_str = 1
    fkimg = np.array(
        [[self.get_strength(x, y) / max_str for y in range(self.height)]
         for x in range(self.width)]
    )
    # peak-finding via dilation reconstruction
    seed = np.copy(fkimg)
    seed[1:-1, 1:-1] = fkimg.min()
    rec = reconstruction(seed, fkimg, method="dilation")
    peaks = fkimg - rec
    # ... connected component labeling, DataFrame creation ...
```

### Proposed Flow

```python
def get_focus(self) -> DataSet[VisionAttentionSchema]:
    max_str = self.get_max_strength()
    if max_str == 0:
        max_str = 1
    fkimg = np.array(
        [[self.get_strength(x, y) / max_str for y in range(self.height)]
         for x in range(self.width)]
    )

    # Apply IOR attenuation before peak-finding
    from .saliency_attenuation import SaliencyAttenuationExpMod
    attenuation = SaliencyAttenuationExpMod.get(default="none")
    fkimg = attenuation.attenuate(fkimg, self)

    # peak-finding via dilation reconstruction (unchanged)
    seed = np.copy(fkimg)
    seed[1:-1, 1:-1] = fkimg.min()
    rec = reconstruction(seed, fkimg, method="dilation")
    peaks = fkimg - rec
    # ... connected component labeling, DataFrame creation ...
```

The ExpMod receives the normalized strength image (`fkimg`, a 2D numpy array with values in [0, 1]) and the `SaliencyMap` instance (for accessing grid metadata like width, height, and cell features). It returns a modified strength image of the same shape. The downstream peak-finding operates on the attenuated image, so heavily attenuated locations naturally produce smaller (or no) peaks.

---

## 3. ExpMod Interface

### Base Class

```python
class SaliencyAttenuationExpMod(ExpMod):
    modtype = "saliency-attenuation"

    def attenuate(self, strength_image: np.ndarray, saliency_map: SaliencyMap) -> np.ndarray:
        """Attenuate the strength image before peak-finding.

        Args:
            strength_image: 2D numpy array of normalized saliency strengths,
                shape (width, height), values in [0, 1]. This is the `fkimg`
                computed inside `SaliencyMap.get_focus()`.
            saliency_map: The SaliencyMap instance, providing access to grid
                dimensions, cell features via `get_val(x, y)`, and other
                grid metadata.

        Returns:
            A 2D numpy array of the same shape with attenuated strength values.
            Values should remain in [0, 1]. The returned array is used for
            downstream peak-finding -- reduced values produce smaller peaks.
        """
        raise NotImplementedError

    def notify_focus(self, focus_points: DataSet[VisionAttentionSchema]) -> None:
        """Called after peak-finding with the resulting focus points.

        Allows the ExpMod to record which location was ultimately selected
        (the top-ranked peak after attenuation) for use in future attenuation
        decisions. Called by `get_focus()` before returning.

        Args:
            focus_points: The final DataFrame of detected peaks after
                attenuation, sorted by descending strength.
        """
        pass  # Default: no-op. Override in stateful flavors.
```

### Lifecycle

- `attenuate()` is called once per tick by `SaliencyMap.get_focus()`, after building the strength image but before dilation-based peak detection
- `notify_focus()` is called once per tick after peak-finding completes, so the ExpMod knows which location(s) emerged as peaks. The top-ranked peak (iloc[0]) is what ObjectResolver will attend.
- The ExpMod is stateful -- it maintains a history of previously attended locations across ticks
- State is maintained on the ExpMod singleton instance (same as `DirichletCategoricalResolution._alphas`)
- The current tick is obtained via `from .sequencer import tick as current_tick` inside the method

---

## 4. Flavor 1: `none` (Default)

### Behavior

Returns the strength image unchanged. This is exactly the current behavior -- peak-finding operates on the raw normalized strengths.

### Implementation

```python
class NoAttenuation(SaliencyAttenuationExpMod):
    name = "none"

    def attenuate(self, strength_image: np.ndarray, saliency_map: SaliencyMap) -> np.ndarray:
        return strength_image
```

No state. No parameters. Direct passthrough.

---

## 5. Flavor 2: `linear-decline`

### Theory Basis

Snyder & Kingstone (2000) showed IOR can be measured at a minimum of 5 previously attended locations, with inhibition magnitude largest at the most recently attended location and declining approximately linearly for each earlier location. Wang & Klein (2010) found IOR persists for ~4 previously inspected items during search.

### Mechanism

Maintain a fixed-size buffer of the last N attended locations (default N=5). On each tick:

1. Receive the normalized strength image (2D array, values in [0, 1])
2. For each cell (x, y) in the image, compute an attenuation multiplier based on proximity to previously attended locations
3. Multiply each cell's strength by its attenuation multiplier
4. Return the attenuated strength image for peak-finding
5. In `notify_focus()`, record the top-ranked peak location in the history buffer (FIFO, capped at N)

### Attenuation Multiplier

For cell (x, y), the penalty from the i-th most recent attended location (x_i, y_i) is:

```
penalty_i = max_penalty * recency_weight_i * spatial_weight_i

recency_weight_i = (N - i) / N       # linear decline: most recent = 1.0, oldest = 1/N
spatial_weight_i  = max(0, 1 - d_i / radius)  # linear spatial falloff
d_i = |x - x_i| + |y - y_i|         # Manhattan distance (matches game grid)
```

Total penalty at (x, y):

```
penalty(x, y) = sum(penalty_i for i in 0..len(history)-1)
```

The attenuation multiplier is:

```
multiplier(x, y) = max(1 - max_attenuation, 1 - penalty(x, y))
attenuated_strength(x, y) = raw_strength(x, y) * multiplier(x, y)
```

Where `max_attenuation` caps how much IOR can reduce saliency (default 0.9 = can reduce to 10% of original, never fully zero out). Because this operates on the strength image before peak-finding, heavily attenuated cells may no longer produce peaks in the dilation reconstruction step -- this is the emergent "suppression" effect.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `capacity` | 5 | Number of previous locations to track (N) |
| `max_penalty` | 1.0 | Maximum penalty at exact location of most recent fixation |
| `radius` | 3 | Manhattan distance at which spatial attenuation reaches zero |
| `max_attenuation` | 0.9 | Maximum fraction of saliency that can be reduced (0.0-1.0) |

### Data Structures

```python
@dataclass
class AttendedLocation:
    x: int
    y: int
    tick: int

class LinearDeclineAttenuation(SaliencyAttenuationExpMod):
    name = "linear-decline"

    capacity: int = 5
    max_penalty: float = 1.0
    radius: int = 3
    max_attenuation: float = 0.9

    def __init__(self) -> None:
        super().__init__()
        self._history: deque[AttendedLocation] = deque(maxlen=self.capacity)
```

### Telemetry

See Section 12 (Observability) for the full observability plan. Linear-decline-specific metrics:
- `roc.saliency_attenuation.max_penalty` (histogram) -- largest penalty applied to any cell
- `roc.saliency_attenuation.history_size` (histogram) -- current history buffer fill level

---

## 6. Flavor 3: `active-inference`

### Theory Basis

Parr & Friston (2017) "Uncertainty, Epistemics and Active Inference" demonstrates that IOR-like behavior emerges naturally from an active inference agent that selects actions to minimize expected free energy. The agent attends to locations with highest expected information gain. Recently observed locations have lower epistemic value (their hidden states are already known), creating IOR. As the environment changes (volatility), uncertainty about previously observed locations grows, and the agent revisits them -- producing adaptive, context-sensitive IOR.

### Architecture Overview

The active inference agent maintains beliefs about:
- **Hidden states s** at each spatial location (what is there?)
- **Sensory precision zeta** per location (how reliable is the observation?)
- **Transition volatility omega** (how fast do things change?)

Each tick, it:
1. Receives the strength image in `attenuate()`
2. Propagates uncertainty at all tracked locations (unobserved locations drift toward uniform)
3. Computes per-cell attenuation based on epistemic value: low-entropy (well-known) cells are attenuated, high-entropy (uncertain) cells are boosted or left unchanged
4. Returns the attenuated strength image for peak-finding
5. In `notify_focus()`, observes features at the top-ranked peak and updates beliefs

### State Space Design

The agent's state space maps onto ROC's saliency map grid. However, maintaining full active inference over every cell in the ~80x21 grid would be too expensive. Instead, the agent tracks beliefs only at **previously attended locations** (capped at `max_locations`). Cells without tracked beliefs are assumed to have maximum entropy (uniform) and receive no attenuation.

**Hidden states per focus point:** Each focus point location (x, y) has a categorical belief distribution Q(s) over a small discrete state space. The states encode an abstract "identity" of what occupies that location.

**State encoding:** Feature vectors at each focus point are hashed to a discrete vocabulary. Each unique feature-set combination maps to one state. The vocabulary grows as new feature combinations are encountered (up to a maximum, after which least-recently-used states are evicted).

```
state_id = hash(frozenset(str(f) for f in features_at_location))
```

### Generative Model

Following Parr & Friston (2017), the generative model is a discrete-state Markov Decision Process:

```
P(o_tau, s_tau | s_{tau-1}, a) = P(o_tau | s_tau) * P(s_tau | s_{tau-1}, a)
```

**Likelihood A:** Maps hidden states to observations.
- `A[o, s]` = probability of observing feature-set o given hidden state s at a location
- Initialized as identity (each state produces its characteristic observation)
- Modulated by sensory precision zeta: `P(o|s) ~ exp(zeta * ln A[o,s])`

**Transition B:** Models how states change over time.
- `B[s', s]` = probability of transitioning from state s to state s' in one tick
- Initialized with high self-transition probability (states are usually stable)
- Modulated by volatility omega: `P(s'|s) ~ exp(omega * ln B[s',s])`

### Belief Update (Perception Step)

When the agent observes location (x, y) at tick t:

```python
# 1. Get current observation (feature hash -> state_id)
o = encode_observation(features_at_location)

# 2. Update state belief Q(s) at this location via Bayes rule
log_likelihood = zeta * ln(A[o, :])          # precision-weighted
log_prior = omega * ln(B[:, :] @ Q_prev(s))  # volatility-weighted transition
Q(s) = softmax(log_likelihood + log_prior)

# 3. Update precision zeta from prediction error
prediction_error = -ln(A[o, :] @ Q_prev(s))   # surprise
zeta = update_gamma(zeta_alpha, zeta_beta + prediction_error)

# 4. Update volatility omega from state prediction error
state_pe = KL[Q(s) || B @ Q_prev(s)]          # state transition surprise
omega = update_gamma(omega_alpha, omega_beta + state_pe)
```

### Attenuation via Epistemic Value

For each cell (x, y) in the strength image, compute an epistemic attenuation factor:

```
epistemic_value(x, y) = H[Q(s_{x,y})]   # entropy of belief at this location
                                          # high entropy = uncertain = worth visiting
max_entropy = ln(max_states)              # entropy of uniform distribution
```

The attenuation multiplier for cell (x, y):

```
# Cells with low entropy (well-known) are attenuated; high entropy (unknown) are preserved
normalized_entropy = epistemic_value(x, y) / max_entropy   # in [0, 1]
multiplier(x, y) = (1 - max_attenuation) + max_attenuation * normalized_entropy
```

- A cell with maximum entropy (uniform belief, never observed or fully recovered) gets `multiplier = 1.0` -- no attenuation.
- A cell with zero entropy (perfectly known, just observed) gets `multiplier = 1 - max_attenuation` -- heavily attenuated.
- Cells without tracked beliefs are assumed to have maximum entropy: `multiplier = 1.0`.

The `saliency_weight` parameter blends epistemic attenuation with raw saliency:

```
effective_multiplier(x, y) = (1 - saliency_weight) * multiplier(x, y) + saliency_weight * 1.0
```

With `saliency_weight=0.0`, attenuation is fully epistemic. With `saliency_weight=1.0`, no attenuation occurs (raw saliency dominates).

### Why This Creates IOR

1. Agent attends to location A (high uncertainty -> high entropy -> high epistemic value)
2. After observing A, Q(s_A) becomes peaked (low entropy) -- epistemic value drops
3. Other locations B, C remain uncertain -> higher epistemic value -> agent attends there
4. Over time, state at A may change (governed by volatility omega) -> Q(s_A) entropy slowly increases
5. Eventually A becomes uncertain enough to be worth revisiting

The "inhibition duration" is directly controlled by volatility omega:
- **High volatility** (fast-changing environment): states change quickly, entropy recovers fast, IOR is short
- **Low volatility** (stable environment): states persist, entropy stays low, IOR is long

### Per-Location State

```python
@dataclass
class LocationBelief:
    """Beliefs about a single spatial location."""
    q_s: np.ndarray           # Categorical distribution over hidden states, shape (n_states,)
    last_observed_tick: int    # When this location was last attended
    last_observation: int      # State ID of last observation
    zeta: float               # Sensory precision (Gamma posterior: alpha/beta)
    zeta_alpha: float          # Gamma shape parameter for zeta
    zeta_beta: float           # Gamma rate parameter for zeta

@dataclass
class GlobalBelief:
    """Global beliefs shared across locations."""
    omega: float              # Transition volatility (Gamma posterior: alpha/beta)
    omega_alpha: float        # Gamma shape parameter for omega
    omega_beta: float         # Gamma rate parameter for omega
```

### State Vocabulary

```python
class StateVocabulary:
    """Maps feature-set hashes to discrete state IDs."""

    def __init__(self, max_states: int = 64) -> None:
        self._hash_to_id: dict[int, int] = {}
        self._next_id: int = 0
        self._max_states: int = max_states

    def encode(self, features: list[VisualFeature]) -> int:
        """Returns a stable integer state ID for a feature set."""
        h = hash(frozenset(str(f) for f in features))
        if h not in self._hash_to_id:
            if self._next_id >= self._max_states:
                # Evict least recently used or wrap around
                self._next_id = self._next_id % self._max_states
            self._hash_to_id[h] = self._next_id
            self._next_id += 1
        return self._hash_to_id[h]
```

### Entropy Propagation (Unobserved Locations)

Locations that haven't been observed recently need their entropy to increase over time (modeling the possibility that their state has changed). On each tick, for every tracked location that was NOT observed this tick:

```python
def propagate_uncertainty(belief: LocationBelief, B: np.ndarray, omega: float) -> None:
    """Increase uncertainty at unobserved locations via transition model."""
    ticks_elapsed = current_tick - belief.last_observed_tick
    if ticks_elapsed <= 0:
        return

    # Apply transition matrix to spread belief (states may have changed)
    # Weighted by volatility: higher omega = faster spread toward uniform
    transition_rate = 1 - exp(-omega * ticks_elapsed)
    uniform = np.ones_like(belief.q_s) / len(belief.q_s)
    belief.q_s = (1 - transition_rate) * belief.q_s + transition_rate * uniform
```

This is the key mechanism: unobserved locations drift toward uniform (maximum entropy), making them increasingly attractive to the epistemic agent. The rate is controlled by omega.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_states` | 64 | Maximum discrete states in vocabulary |
| `max_locations` | 32 | Maximum tracked location beliefs (LRU eviction) |
| `max_attenuation` | 0.9 | Maximum fraction of saliency that can be reduced (0.0-1.0) |
| `zeta_alpha_prior` | 2.0 | Gamma shape prior for sensory precision |
| `zeta_beta_prior` | 1.0 | Gamma rate prior for sensory precision |
| `omega_alpha_prior` | 2.0 | Gamma shape prior for volatility |
| `omega_beta_prior` | 1.0 | Gamma rate prior for volatility |
| `B_self_transition` | 0.9 | Prior probability that a location's state stays the same |
| `saliency_weight` | 0.5 | Weight for raw saliency in final score (0 = pure epistemic, 1 = pure saliency) |

### Saliency Integration

Pure epistemic attenuation could suppress all recently-observed locations even if they have high raw saliency, which would defeat the purpose of the saliency map. The `saliency_weight` parameter controls this trade-off:

```
effective_multiplier(x, y) = (1 - saliency_weight) * epistemic_multiplier(x, y)
                           +      saliency_weight  * 1.0
```

This ensures highly salient novel events still produce peaks even if a nearby location was recently observed. The default `saliency_weight=0.5` provides a balanced blend.

### Full Class Outline

```python
class ActiveInferenceAttenuation(SaliencyAttenuationExpMod):
    name = "active-inference"

    max_states: int = 64
    max_locations: int = 32
    max_attenuation: float = 0.9
    zeta_alpha_prior: float = 2.0
    zeta_beta_prior: float = 1.0
    omega_alpha_prior: float = 2.0
    omega_beta_prior: float = 1.0
    B_self_transition: float = 0.9
    saliency_weight: float = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._vocab = StateVocabulary(max_states=self.max_states)
        self._beliefs: dict[tuple[int, int], LocationBelief] = {}
        self._global = GlobalBelief(
            omega=self.omega_alpha_prior / self.omega_beta_prior,
            omega_alpha=self.omega_alpha_prior,
            omega_beta=self.omega_beta_prior,
        )
        self._A: np.ndarray = np.eye(self.max_states)  # likelihood matrix
        self._B: np.ndarray = self._init_transition_matrix()

    def attenuate(self, strength_image: np.ndarray, saliency_map: SaliencyMap) -> np.ndarray:
        """Attenuate strength image based on epistemic value at each cell."""
        from .sequencer import tick as current_tick

        # 1. Propagate uncertainty at all tracked locations
        self._propagate_all(current_tick)

        # 2. Build attenuation mask
        max_entropy = np.log(self.max_states)
        result = np.copy(strength_image)
        width, height = strength_image.shape

        for (bx, by), belief in self._beliefs.items():
            if 0 <= bx < width and 0 <= by < height:
                normalized_entropy = belief.entropy() / max_entropy
                epistemic_mult = (1 - self.max_attenuation) + self.max_attenuation * normalized_entropy
                effective_mult = (1 - self.saliency_weight) * epistemic_mult + self.saliency_weight
                result[bx, by] *= effective_mult

        # Cells without tracked beliefs: no attenuation (multiplier = 1.0)
        return result

    def notify_focus(self, focus_points: DataSet[VisionAttentionSchema]) -> None:
        """Observe features at the top-ranked peak and update beliefs."""
        from .sequencer import tick as current_tick

        if len(focus_points) == 0:
            return
        top = focus_points.iloc[0]
        sx, sy = int(top["x"]), int(top["y"])
        features = self._last_saliency_map.get_val(sx, sy)
        self._observe(sx, sy, features, current_tick)

    def _epistemic_value(self, x: int, y: int) -> float:
        """Entropy of belief at (x, y). Higher = more uncertain = more worth visiting."""
        ...

    def _observe(self, x: int, y: int, features: list, tick: int) -> None:
        """Update beliefs after observing location (x, y)."""
        ...

    def _propagate_all(self, current_tick: int) -> None:
        """Increase uncertainty at all unobserved locations."""
        ...

    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize B with high self-transition prior."""
        ...
```

### Telemetry

See Section 12 (Observability) for the full observability plan. Active-inference-specific metrics:
- `roc.saliency_attenuation.entropy_at_focus` (histogram) -- entropy at the attended location
- `roc.saliency_attenuation.entropy_range` (histogram) -- max minus min entropy across tracked beliefs
- `roc.saliency_attenuation.omega` (histogram) -- current volatility estimate
- `roc.saliency_attenuation.vocab_size` (counter) -- unique states in vocabulary
- `roc.saliency_attenuation.beliefs_tracked` (histogram) -- number of active location beliefs

---

## 7. SaliencyMap.get_focus() Changes

The only change to existing code is in `SaliencyMap.get_focus()` (attention.py):

```python
# After building fkimg, before peak-finding:
from .saliency_attenuation import SaliencyAttenuationExpMod
attenuation = SaliencyAttenuationExpMod.get(default="none")
fkimg = attenuation.attenuate(fkimg, self)

# After building the focus_points DataFrame, before returning:
attenuation.notify_focus(focus_points_dataset)
```

This is a small addition to `get_focus()`. ObjectResolver and the rest of the pipeline are completely unchanged.

---

## 8. File Organization

```
roc/
  attention.py          # MINIMAL CHANGE - get_focus() calls ExpMod for attenuation
  object.py             # UNCHANGED - ObjectResolver still takes focus_points.iloc[0]
  saliency_attenuation.py   # NEW - ExpMod base class + "none" implementation
                             #     + "linear-decline" implementation
                             #     + "active-inference" implementation
```

All three flavors live in `roc/saliency_attenuation.py` since they are core implementations (not experiment-specific), similar to how `SymmetricDifferenceResolution` and `DirichletCategoricalResolution` both live in `roc/object.py`.

---

## 9. Configuration

### Default (no change from current behavior)

No config changes needed. The `"none"` flavor is the default.

### Enabling linear-decline

```bash
roc_expmods_use='[("saliency-attenuation", "linear-decline")]'
```

### Enabling active-inference

```bash
roc_expmods_use='[("saliency-attenuation", "active-inference")]'
```

### Combined with other ExpMods

```bash
roc_expmods_use='[("action", "weighted"), ("saliency-attenuation", "linear-decline")]'
```

---

## 10. Testing Strategy

### Unit Tests (`tests/unit/test_saliency_attenuation.py`)

**NoAttenuation:**
- Returns strength image unchanged
- Works with various image sizes
- Works with all-zero image (edge case)

**LinearDeclineAttenuation:**
- With empty history, returns strength image unchanged (same as none)
- After notify_focus records location A, next tick attenuates cells near A
- Attenuation declines linearly: most recent location gets strongest attenuation
- Spatial falloff: cells far from history entries receive no attenuation
- History buffer respects capacity limit (FIFO eviction)
- max_attenuation prevents total zeroing of cells
- Multiple locations in history produce cumulative attenuation
- Attenuated cells produce fewer/smaller peaks in downstream peak-finding

**ActiveInferenceAttenuation:**
- With no prior observations, returns strength image unchanged (uniform entropy everywhere)
- After observing location A, cells near A are attenuated (low entropy)
- Unobserved locations gain entropy over ticks (uncertainty propagation)
- After enough ticks, A's entropy recovers and attenuation decreases
- Higher omega (volatility) -> faster entropy recovery -> shorter IOR
- saliency_weight=1.0 applies no attenuation (behaves like NoAttenuation)
- saliency_weight=0.0 is pure epistemic attenuation
- State vocabulary encodes and retrieves feature sets correctly

### Integration Tests (`tests/integration/test_saliency_attenuation.py`)

- Build a SaliencyMap, call get_focus() with each flavor, and verify peak differences
- Run a short sequence (5-10 ticks) with linear-decline: verify that repeatedly salient locations get attenuated and different peaks emerge
- Run with active-inference: verify entropy dynamics over a sequence cause peak rotation

---

## 11. Design Decisions and Rationale

### Why modify SaliencyMap.get_focus() instead of ObjectResolver?

Attenuation modifies the *strength landscape* that peak-finding operates on. This is fundamentally a saliency-level operation:
- Attenuating before peak-finding means peaks naturally appear/disappear based on the modified strength values -- no need to explicitly suppress or re-rank focus points
- The boundary is clean: `get_focus()` produces the best available peaks given IOR history, and ObjectResolver simply takes the top one without knowing about attenuation
- ObjectResolver's job is resolving focus points to objects, not deciding which focus points to consider

### Why not a separate Component?

A separate Component on the EventBus would add latency and complexity. The ExpMod pattern is simpler: it's a stateful function called synchronously within `get_focus()`. This matches how ObjectResolutionExpMod works -- called within the same Component's event handler.

### Why include saliency_weight in active-inference?

Pure epistemic selection would ignore saliency entirely, meaning a highly novel, salient event (e.g., a monster appearing) could be ignored in favor of revisiting a location just because it's been a while. The saliency_weight parameter preserves the "capture" effect of strong saliency signals while still biasing toward uncertainty reduction.

### Why does active-inference only track beliefs at previously attended locations?

Active inference over the full 80x21 grid (1680 cells) each tick would be expensive and unnecessary. The agent only needs beliefs at locations it has previously observed. Untracked cells are assumed to have maximum entropy (unknown) and receive no attenuation -- which is the correct behavior since there is nothing to inhibit at a never-visited location.

### Why Manhattan distance for spatial falloff?

ROC operates on a discrete grid (NetHack). Manhattan distance is the natural metric for grid adjacency and matches the movement model. Euclidean distance would work too but adds unnecessary floating-point computation for a grid-based game.

### Why attenuation instead of suppression?

Attenuation (reducing strength values) is more principled than suppression (removing peaks from the list):
- It operates at the right level of abstraction: the strength image, not the output DataFrame
- The "suppression" effect emerges naturally -- heavily attenuated cells fall below the peak detection threshold
- Partially attenuated locations still appear as peaks but rank lower, providing a graceful gradient
- It avoids the edge case of an empty focus_points DataFrame (there is always at least one peak if there is any saliency)

---

## 12. Observability

Saliency attenuation uses the same observability stack as the rest of ROC: OTel metrics for dashboards, OTel structured logs for per-tick decision records, tracing spans for performance, and loguru for terminal output. The guiding principle: **metrics** answer "is IOR working across runs?", **structured logs** answer "why did IOR pick *this* location on tick 47?", **traces** answer "how long did attenuation take?", and **terminal logs** are just sanity checks.

### 12.1 OTel Metrics

Created via `Observability.meter` following the `roc.` naming convention.

**All flavors:**

| Metric | Type | Unit | Description |
|---|---|---|---|
| `roc.saliency_attenuation.peak_count` | histogram | peaks | Number of peaks after attenuation (compare to pre-attenuation to measure collapse) |
| `roc.saliency_attenuation.top_peak_strength` | histogram | strength | Strength of the #1 peak post-attenuation |
| `roc.saliency_attenuation.top_peak_shifted` | counter | decision | Whether the #1 peak changed identity vs. unattenuated image (attributes: `shifted=true/false`) |

**linear-decline only:**

| Metric | Type | Unit | Description |
|---|---|---|---|
| `roc.saliency_attenuation.max_penalty` | histogram | penalty | Largest penalty applied to any cell this tick |
| `roc.saliency_attenuation.history_size` | histogram | locations | Current history buffer fill level |

**active-inference only:**

| Metric | Type | Unit | Description |
|---|---|---|---|
| `roc.saliency_attenuation.entropy_at_focus` | histogram | nats | Entropy at the attended location (low = revisiting known, high = exploring) |
| `roc.saliency_attenuation.entropy_range` | histogram | nats | Max minus min entropy across tracked beliefs |
| `roc.saliency_attenuation.omega` | histogram | rate | Current volatility estimate |
| `roc.saliency_attenuation.vocab_size` | counter | states | Unique states in vocabulary |
| `roc.saliency_attenuation.beliefs_tracked` | histogram | locations | Number of active location beliefs |

### 12.2 OTel Structured Logs

One structured log record per tick via `Observability.get_logger("roc.saliency_attenuation")`, following the `_log_decision()` pattern in `object.py`. This record flows to the JSONL debug log (when `roc_debug_log=true`), the Remote Logger MCP (default on), and Loki.

```python
record = {
    "event": "saliency_attenuation",
    "tick": current_tick,
    "flavor": "linear-decline",           # or "none", "active-inference"
    "peaks_before": 8,                     # peak count before attenuation
    "peaks_after": 6,                      # peak count after attenuation
    "top_peak_before": {"x": 10, "y": 5, "strength": 0.92},
    "top_peak_after": {"x": 20, "y": 15, "strength": 0.78},
    "top_peak_shifted": True,              # did the winner change?
}
```

**linear-decline additions:**
```python
{
    "history": [(10, 5), (20, 15), (30, 10)],   # current history buffer
    "max_penalty_applied": 0.73,                  # largest penalty on any cell
}
```

**active-inference additions:**
```python
{
    "entropy_at_focus": 0.12,          # entropy at attended location
    "entropy_max": 4.16,               # max entropy across tracked beliefs
    "entropy_min": 0.12,               # min entropy across tracked beliefs
    "omega": 2.1,                      # current volatility estimate
    "beliefs_tracked": 12,             # number of active location beliefs
    "vocab_size": 23,                  # unique states in vocabulary
}
```

### 12.3 Tracing Spans

Wrap `attenuate()` with a span so it appears in traces alongside `do_attention` and `do_object_resolution`:

```python
@Observability.tracer.start_as_current_span("saliency_attenuation")
def attenuate(self, strength_image: np.ndarray, saliency_map: SaliencyMap) -> np.ndarray:
    ...
```

The span context is automatically attached to the structured log record, correlating attenuation decisions with the enclosing attention span.

### 12.4 Loguru Terminal Output

Minimal -- only for development sanity checks. High-frequency per-tick info at `debug` level, significant events at `info`:

```python
logger.debug("attenuation: {} peaks -> {}, top shifted: {}", before, after, shifted)
logger.info("IOR history full, evicting oldest: ({}, {})", x, y)
```

Filterable via `roc_log_modules="saliency_attenuation:DEBUG"`.

### 12.5 What NOT to Collect

- **Per-cell attenuation values**: The ~1680 cells per tick are too noisy. Use the structured log's summary stats instead.
- **Raw belief vectors**: High-dimensional, only useful for deep debugging. Use the DAP MCP debugger to inspect these live via `evaluate_expression`.
- **Separate metrics for `notify_focus()`**: It's part of the same tick as `attenuate()` -- log them together in one record.
