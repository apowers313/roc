# Dirichlet-Categorical Object Resolution: Design

## Overview

Replace the current `SymmetricDifferenceResolution` ExpMod with a Bayesian
approach using a Dirichlet-Categorical model. The new ExpMod computes posterior
probabilities over object identities rather than hard set-difference thresholds.

This design is game-agnostic. It operates on abstract feature nodes without
knowledge of what game or environment produces them.

## Design Principles

**Game-agnostic core.** ROC's architecture separates game-specific perception
(feature extractors) from game-agnostic downstream processing. Everything from
Attention onward -- including object resolution -- must work with abstract
features without assuming anything about the environment producing them. The
current `SymmetricDifferenceResolution` violates this by hardcoding
NetHack-specific feature type names. This design corrects that.

**Minimal assumptions.** The algorithm should assume as little as possible about
the structure of the feature space, the dynamics of the environment, or the
number and behavior of objects. Where assumptions are unavoidable (e.g.,
"objects have spatial positions"), they should be justified by the system
architecture (this is a visual perception system) rather than by properties of
a specific game.

**Principled over hand-tuned.** Where possible, let the math determine
thresholds and weights rather than requiring manual tuning. The current
distance <= 1 threshold is arbitrary and must be re-tuned for different feature
configurations. A Bayesian approach derives decisions from evidence
accumulation.

## Background

See `design/object-resolution-alternatives.md`, Alternative 6 for the full
Bayesian Object Identity proposal. This document narrows that proposal into a
concrete implementation plan for the ROC ExpMod system.

### Why Dirichlet-Categorical Over Other Alternatives

The alternatives document (see `design/object-resolution-alternatives.md`)
presents eight approaches. We chose Dirichlet-Categorical (Alternative 6) for
the following reasons:

1. **Jaccard similarity (Alt 1) and TF-IDF (Alt 2)** are incremental
   improvements to the distance metric. They fix specific problems (size
   normalization, feature weighting) but remain threshold-based: you still need
   an arbitrary cutoff to decide "match vs. new." Dirichlet-Categorical
   subsumes both -- feature frequency naturally gives TF-IDF-like weighting
   (frequently observed features dominate the alpha vector), and the posterior
   normalizes across candidates regardless of feature set size.

2. **Spatial priors (Alt 3)** are incorporated into this design as a prior
   term. They are not an alternative to Dirichlet-Categorical but a complement.

3. **Bipartite matching (Alt 4)** solves a different problem (resolving
   multiple objects per frame). It is orthogonal to the distance metric and can
   be layered on top of Dirichlet-Categorical later. Currently the system only
   resolves one object per frame; bipartite matching requires multi-object
   attention which is not yet implemented.

4. **Feature vectors with cosine similarity (Alt 5)** requires defining a
   fixed-length vector schema, which means knowing all feature dimensions up
   front. This conflicts with the game-agnostic principle -- different
   environments may produce different feature types. The Dirichlet model works
   with variable-length feature sets naturally.

5. **Kalman filter (Alt 7)** is a motion model. It predicts where objects will
   be based on velocity estimates. This is valuable but adds significant
   complexity (state estimation, process noise, measurement noise matrices).
   The exponential spatial decay in our design captures the core intuition
   ("nearby objects are more likely the same") without requiring a full motion
   model. Kalman filtering can be added later as a refinement to the spatial
   prior.

6. **Learned distance (Alt 8)** requires a reward signal for resolution
   quality, which depends on downstream components (Transformer, Action) being
   mature enough to provide meaningful feedback. The current system is not at
   that stage. Dirichlet-Categorical learns from observation counts alone --
   unsupervised, no reward signal needed.

**Summary of decision**: Dirichlet-Categorical provides the best ratio of
capability to complexity. It handles uncertainty, learns from data, subsumes
simpler metric improvements, and requires no game-specific knowledge or
downstream reward signals. It is also well-understood mathematically (conjugate
prior inference), making it straightforward to implement correctly and reason
about.

### Current System

`SymmetricDifferenceResolution` (in `roc/object.py`):

1. Walks the graph backwards from feature nodes to find candidate Objects.
2. Computes symmetric set difference over a hardcoded set of "physical" feature
   types (`SingleNode`, `ColorNode`, `ShapeNode`).
3. Matches if best candidate has distance <= 1; otherwise creates a new Object.

Problems:
- Binary match/no-match with an arbitrary threshold.
- All features weighted equally.
- Hardcoded feature type filter is game-specific.
- No spatial or temporal context.
- A feature occasionally absent causes a hard mismatch.

## Algorithm

Each frame, for one observation (the highest-saliency focus point):

### Step 1: Find Candidates

Same graph walk as today: from the observation's feature nodes, traverse
`FeatureNode -> FeatureGroup -> Object` to collect all Objects sharing at least
one feature. This step is unchanged.

### Step 2: Compute Priors -- P(identity)

For each candidate Object and a "new object" hypothesis, assign a prior
probability before examining features.

**Spatial prior.** ROC is a visual perception system. The data pipeline always
produces observations at (x, y) screen coordinates -- this is intrinsic to the
architecture, not a game-specific assumption. The `ResolvedObject` dataclass
already carries `x: XLoc` and `y: YLoc`, and the `ObjectCache` is keyed by
`(XLoc, YLoc)`. Any environment that ROC perceives visually will have spatial
coordinates.

For each candidate Object, compute the distance from the current observation's
position to the Object's last-seen position. Convert to a prior weight using an
exponential decay:

```
spatial_weight(obj) = exp(-manhattan_distance / spatial_scale)
```

`spatial_scale` is a configurable parameter on the ExpMod controlling how
quickly spatial influence decays. Objects with no prior position (never resolved
before) get a uniform spatial weight (1.0).

**Why Manhattan distance.** The visual system operates on a discrete grid (screen
coordinates). Manhattan distance is the natural metric for grid-based positions,
avoids the sqrt computation of Euclidean distance, and is adequate for the
purpose of "nearby vs. far." If ROC is ever applied to continuous-coordinate
environments, Euclidean distance could be substituted with no algorithmic change
-- only the distance computation differs.

**Why exponential decay.** The exponential `exp(-d / scale)` has desirable
properties: it's always positive, monotonically decreasing, and the `scale`
parameter has an interpretable meaning (the distance at which the weight drops to
~37%). Other decay functions (Gaussian, inverse-square) would also work; we chose
exponential for simplicity and because it penalizes large distances without
completely zeroing them out.

**Temporal prior.** Each Object tracks the tick at which it was last resolved.
Objects not seen recently are less likely to be the same entity:

```
temporal_weight(obj) = exp(-ticks_since_last_seen / temporal_scale)
```

`temporal_scale` is a configurable parameter on the ExpMod. This is generic --
every sequential environment has ticks (the `Frame.tick` counter in
`roc/sequencer.py`).

**Rationale.** The current system has no temporal awareness at all. An Object
last seen 500 ticks ago competes equally with one seen last tick. This is
clearly wrong -- the longer an entity has been absent, the less likely a new
observation corresponds to it (it may have been destroyed, moved off-screen,
etc.). Temporal decay captures this intuition without requiring any
game-specific knowledge about object lifetimes.

**Current state.** The `Object` node has `resolve_count` (total times matched)
but no `last_tick` or `last_seen` field. The `Frame` node has a `tick` field
incremented by `get_next_tick()` in `roc/sequencer.py`. The temporal prior
requires adding `last_tick` to the Object node (see Data Storage section).

**Interaction with spatial prior.** The spatial and temporal priors are
multiplied, not added. This means an Object that is both far away AND not seen
recently gets a very low prior -- the two signals reinforce each other. An
Object that is far away but was just seen last tick (e.g., a teleporting entity)
still gets a moderate prior from the temporal term. This multiplicative
combination is standard in Bayesian tracking literature (see SORT tracker,
Bewley et al. 2016).

**New object prior.** A base probability that the observation is something never
seen before. See "New Object Prior" section below for how this value is
determined.

**Combined prior.** Multiply spatial and temporal weights, normalize across all
candidates plus the new-object hypothesis:

```
unnormalized_prior(obj) = spatial_weight(obj) * temporal_weight(obj)
unnormalized_prior(new) = new_object_prior_weight
P(obj) = unnormalized_prior(obj) / sum(all unnormalized priors)
```

### Step 3: Compute Likelihoods -- P(features | identity)

Each Object maintains a Dirichlet parameter vector (alpha values), one entry per
feature it has ever encountered. When a feature is observed on an Object, that
feature's alpha is incremented.

The likelihood of observing a set of features given an Object is the product of
posterior predictive probabilities:

```
P(feature_j | obj) = alpha_j / sum(alpha)
```

For the full observation (assuming feature independence):

```
P(observation | obj) = product of P(feature_j | obj) for each observed feature_j
```

**Work in log-space** to avoid underflow: sum log-probabilities instead of
multiplying probabilities.

**Handling unseen features.** If the observation contains a feature that the
candidate Object has never encountered, that feature's alpha is the base
`prior_alpha` (not zero). This is the key advantage of the Dirichlet prior --
unseen features get a small but nonzero probability rather than causing a hard
mismatch. The more features the Object has seen, the lower the relative
probability of an unseen feature (because `sum(alpha)` grows), but it never
reaches zero.

**New object likelihood (Option C).** The "new object" hypothesis is modeled as
a Dirichlet model that has never been updated -- all alphas are at the base
`prior_alpha`. For an observation with K features, each feature has probability
`prior_alpha / (N * prior_alpha)` where N is the total number of distinct
features in the "new object" model's vocabulary. In practice, N is the number
of distinct features the system has ever encountered across all Objects (the
global feature vocabulary size). This means:

- The more features that exist in the system, the lower the "new object"
  likelihood for any particular observation (the probability mass is spread
  more thinly).
- An existing Object with strong evidence for the observed features will
  easily beat the "new object" model.
- An observation containing features that no existing Object has strong
  evidence for will favor the "new object" model.

This is the behavior we want: as the system learns more Objects and their
feature profiles become well-characterized, it becomes harder for an observation
to be classified as "new" when it genuinely matches something known.

### Step 4: Compute Posteriors

Apply Bayes' rule:

```
log P(obj | features) = log P(features | obj) + log P(obj) - log Z
```

where `log Z = logsumexp` over all candidates including "new."

### Step 5: Decision

Use MAP (maximum a posteriori): select the identity with the highest posterior.
If the "new object" hypothesis wins, return `None` (caller creates a new
Object). Otherwise return the matched Object.

Optionally, require that the winning posterior exceed a `confidence_threshold`
(configurable, default 0.5). If no candidate exceeds the threshold, return
`None`.

### Step 6: Update

After resolution, update the matched Object's alpha vector by incrementing
counts for each observed feature. Also update the Object's last-seen position
and tick. These updates happen in the ExpMod's resolve method (for alphas) or in
the caller `ObjectResolver.do_object_resolution` (for position/tick on the
Object node).

## Feature Filtering

### Problem

The current system hardcodes `{"SingleNode", "ColorNode", "ShapeNode"}` as the
feature types used for matching (in `SymmetricDifferenceResolution._distance()`).
This is game-specific -- it assumes NetHack's visual feature types. A different
environment might not have these feature types at all, or might have feature
types where motion or distance features are critical for identity.

### Options Considered

**Option A: Hardcode a different set.** Replace one hardcoded set with another.
Rejected -- same problem, different values.

**Option B: Features self-declare identity relevance.** Add a boolean attribute
(e.g., `identity_relevant: bool`) to the `FeatureNode` base class. Each feature
extractor sets this flag. The resolution algorithm only considers features where
the flag is True. This pushes the decision to the perception layer (which is
game-specific) and keeps the resolution layer generic.

Pros: Clean separation of concerns. Feature extractors know best whether their
output is relevant to object identity.

Cons: Requires modifying the `FeatureNode` base class and all existing feature
extractors. The flag is static -- a feature type is either always relevant or
never relevant, which may be too coarse.

**Option C: Configurable exclusion set on the ExpMod.** The ExpMod takes a
parameter `excluded_feature_labels` (default: empty set). Any feature node whose
labels intersect this set is excluded from likelihood computation. By default,
all features participate.

Pros: No changes to the perception layer. Configurable per experiment.
Straightforward to understand and implement.

Cons: Requires knowing feature type names to configure -- but this is a
deployment concern, not a design concern. The algorithm itself is generic.

**Option D: Learned feature relevance.** Let the Dirichlet model learn which
features matter by observing which features are consistently associated with
correctly-identified objects.

Pros: Fully automatic.

Cons: Requires a correctness signal (how do you know a resolution was correct?).
Adds significant complexity. Premature -- we don't yet have the infrastructure
to evaluate resolution quality.

### Decision

**Option C (configurable exclusion set).** It is the simplest approach that
achieves game-agnosticism. By default, all features participate -- this is the
correct generic behavior since the algorithm has no basis for excluding features
it knows nothing about. Environment-specific configurations can exclude noisy
feature types when empirically justified.

Option B is a good future enhancement if we find that feature extractors
consistently need to signal identity relevance. It can be added without changing
the resolution algorithm -- the exclusion set would simply be populated from
feature flags rather than from config.

## New Object Prior

### Problem

The `new_object_prior_weight` controls how readily the system creates new
Objects vs. matching to existing ones. Too high and it fragments identities
(the same entity gets split into many Objects). Too low and it merges distinct
objects (different entities get collapsed into one Object).

This is the most game-sensitive parameter. A fast-paced environment with many
entities appearing and disappearing needs a higher new-object rate than a
stable environment with persistent objects. A game-agnostic system cannot
hardcode this value.

### Options

**Option A: Fixed parameter.** A configurable constant on the ExpMod (e.g.,
0.1). Simple and predictable.

Pros: Easy to understand and tune. No hidden dynamics.

Cons: Requires manual tuning per environment. A single constant may not be
correct throughout a game -- early exploration may have a high rate of new
objects, while later gameplay in a familiar area may have very few.

**Option B: Adaptive rate.** Track the empirical rate of genuinely new objects
over a sliding window. If the system has been creating many new objects recently,
the prior that the next observation is also new should be higher. Conversely, in
a stable scene where objects persist, the prior should be low.

```
new_object_rate = new_objects_created_in_last_W_ticks / W
new_object_prior_weight = clamp(new_object_rate, min_prior, max_prior)
```

Where `W` (window size), `min_prior`, and `max_prior` are configurable.

Pros: Adapts to the current game state. Handles transitions between
high-novelty and low-novelty phases.

Cons: Introduces three new parameters (`W`, `min_prior`, `max_prior`). The
sliding window size is itself game-sensitive. Can create feedback loops: if
the system incorrectly creates too many new objects, the adaptive rate
increases, making it even more likely to create new objects.

**Option C: Likelihood-driven.** Don't use a separate prior weight at all.
Instead, model the "new object" hypothesis as a Dirichlet model with only the
base prior (all alphas = `prior_alpha`). This model has seen no evidence for
any feature, so it assigns uniform probability across the feature space. The
"new object" hypothesis competes with existing Objects purely on likelihood.

If the observation is poorly explained by all existing Objects (low likelihood
under every learned model), the uniform model wins naturally. If any existing
Object explains the observation well (high likelihood from accumulated alpha
counts), the uniform model loses.

Pros: Requires no tuning beyond `prior_alpha`, which has principled defaults
(1.0 = uniform/Laplace, 0.5 = Jeffreys). No game-specific parameters.
Naturally adapts: well-characterized Objects have strong likelihoods that beat
the uniform model; truly novel observations have low likelihood under all
existing models, so the uniform model wins. No feedback loop risk.

Cons: The behavior is entirely governed by `prior_alpha` and the number of
features in the space. In a very large feature space, the uniform model's
likelihood will be extremely low (spread across many features), making it hard
for the "new object" hypothesis to ever win. Conversely, in a very small
feature space, the uniform model is relatively competitive and may win too
often. The `confidence_threshold` parameter provides a safety valve but does not
fully address this.

### Decision

Start with **Option C** (likelihood-driven). The reasoning:

1. It is the most principled approach -- the decision emerges from Bayesian
   inference rather than a hand-tuned parameter.
2. It requires the fewest parameters to configure, which aligns with the
   game-agnostic design principle.
3. The `prior_alpha` parameter has well-established defaults from Bayesian
   statistics (Jeffreys prior, Laplace prior) rather than requiring empirical
   tuning per environment.
4. The potential failure mode (feature space size affecting the uniform model's
   competitiveness) can be monitored via the existing observability counters
   (`roc.objects_resolved` with `new` attribute).

### Downsides and Risks of Option C

While Option C is the most principled approach, it has specific failure modes
that must be understood and monitored:

1. **Cold start bias toward new objects.** In the first few ticks, all Objects
   have weak alpha profiles (close to the uniform prior). The "new object"
   model is almost as good as any existing Object model. This means early
   observations may create too many Objects before any Object has accumulated
   enough evidence to dominate. Mitigation: accept this as a warmup cost. The
   telemetry will show a high new-object rate in the first ~50 ticks that
   should decrease as alpha vectors accumulate evidence. If it does not
   decrease, the model has a problem.

2. **Feature vocabulary growth.** The "new object" model's likelihood depends
   on the global feature vocabulary size (N). As the system encounters more
   feature types, N grows, and the uniform model's per-feature probability
   (`prior_alpha / (N * prior_alpha) = 1/N`) shrinks. This means the "new
   object" hypothesis gets weaker over time -- which is generally desirable
   (established systems should be skeptical of novelty) but could become too
   weak in environments with very large feature vocabularies. If the system
   stops creating new Objects entirely even when genuinely new entities appear,
   this is the likely cause. Mitigation: monitor the `new_object_posterior`
   histogram. If it trends toward zero, consider switching to Option B or
   capping N.

3. **Sensitivity to prior_alpha.** The `prior_alpha` parameter affects both
   existing Object models (how much probability unseen features get) and the
   new Object model (how competitive it is). A very small `prior_alpha` (e.g.,
   0.01) makes the new Object model very weak and makes existing Objects
   intolerant of unseen features. A very large `prior_alpha` (e.g., 10.0)
   makes the new Object model too competitive and makes existing Objects too
   tolerant of unseen features. The sweet spot (0.5 to 1.0) is well-studied
   in Bayesian statistics, but it may need adjustment based on the actual
   feature space encountered.

4. **No spatial/temporal component in new-object hypothesis.** The "new object"
   model competes on likelihood alone -- it has no spatial or temporal prior
   (it has never been seen anywhere, at any time). This means it cannot
   benefit from spatial context. If a genuinely new entity appears right next
   to a well-characterized existing Object with similar features, the existing
   Object's spatial prior will boost it and may win even though the features
   don't match well. The confidence threshold partially addresses this (the
   match will be low-confidence), but it is an inherent limitation.

If Option C proves too eager or too reluctant to create new Objects in practice,
Option B (adaptive rate) is the recommended follow-up. It can be added as a
modification to the prior computation without changing the rest of the algorithm.
Option A (fixed parameter) is the fallback if adaptive rate introduces
instability.

All three options can be monitored using the same telemetry: the
`roc.resolution.decision` counter distinguishes `match`, `new_object`, and
`low_confidence` outcomes. The `roc.resolution.new_object_posterior` histogram
tracks how competitive the "new object" model is over time.

## Data Storage

### Alpha Vectors

Each Object needs a mapping from feature identity to alpha value. The key is a
string representation of the feature (e.g., `"SingleNode(type=399)"`), matching
how `SymmetricDifferenceResolution._distance()` currently uses `str(f)` for set
operations. The value is a float (the alpha count).

### Options Considered

**Option 1: Property on the Object node.** Store as a JSON-serialized dict on
the Object graph node (e.g., `alphas: dict[str, float]`). Persists in Memgraph.

Pros: Survives agent restarts. Visible in graph queries for debugging and
analysis. Consistent with how other Object properties are stored.

Cons: Requires serialization/deserialization on every access. Memgraph property
access is the performance bottleneck in the current system (the graph walk
dominates per-frame cost). Adding a dict property that must be read and written
every frame adds to this bottleneck. The dict grows over time as the Object
encounters more features.

**Option 2: Side dictionary in the ExpMod.** The ExpMod instance holds
`alphas: dict[NodeId, dict[str, float]]` mapping Object IDs to their alpha
vectors.

Pros: Fast access (pure Python dict lookup, no serialization). Zero graph I/O
overhead. Simple to implement.

Cons: Lost on restart. Not visible in graph queries. Creates a parallel state
store that must be kept in sync with the graph (if an Object is deleted from
the graph, its alpha entry becomes stale).

### Decision

**Side dictionary** (Option 2) for now. Reasoning:

1. Object resolution runs in-process and alpha vectors are only accessed during
   resolution. There is no need for other components to read them.
2. Graph persistence of alpha values is a nice-to-have for post-run analysis,
   not a functional requirement. We can add a dump/export mechanism later.
3. The agent currently runs episodes without restart, so persistence across
   restarts is not yet needed.
4. Avoiding graph I/O for alpha access keeps the performance profile identical
   to the current system -- the only graph operations are the candidate
   discovery walk (unchanged) and the position/tick update (small addition).
5. Stale entries (for deleted Objects) are harmless -- they are just never
   looked up. They can be cleaned up lazily or on a periodic sweep if memory
   becomes a concern (unlikely given ~200 bytes per Object).

### Last-Seen Position and Tick

Add three fields to the `Object` node:

```python
last_x: XLoc | None = Field(default=None)
last_y: YLoc | None = Field(default=None)
last_tick: int = Field(default=0)
```

These are updated by `ObjectResolver.do_object_resolution` after resolution,
regardless of which ExpMod is active. This data is useful to any resolution
strategy, not just Dirichlet-Categorical. The symmetric difference
implementation can ignore this data (it doesn't use spatial or temporal
information), but having it on the Object node means future ExpMods or analysis
tools can access it.

### Resolution Context (Interface Change)

The `resolve` method currently receives only `feature_nodes` and
`feature_group`. The Dirichlet-Categorical model also needs position and tick.

**Options considered:**

1. **Expand the signature** to pass `x`, `y`, `tick` as individual parameters.
   Simple but brittle -- if we later need more context (e.g., saliency
   strength, neighboring features), we add more parameters.

2. **Pass a context dataclass** containing all per-observation metadata.
   One parameter that can grow without changing the signature.

**Decision:** Option 2. The context will likely grow as we add more
sophisticated resolution strategies. A single context object keeps the
interface stable:

```python
@dataclass
class ResolutionContext:
    x: XLoc
    y: YLoc
    tick: int
```

The `resolve` signature becomes:

```python
def resolve(
    self,
    feature_nodes: Collection[FeatureNode],
    feature_group: FeatureGroup,
    context: ResolutionContext,
) -> Object | None:
```

This is a breaking change to the `ObjectResolutionExpMod` interface.
`SymmetricDifferenceResolution` must be updated to accept (and ignore) the
context parameter. This is a one-line change to its signature.

## Assumptions and Limitations

### Feature Independence

The Dirichlet-Categorical model assumes features are conditionally independent
given the Object identity. That is, `P(observation | obj)` is a product of
per-feature probabilities. This means the model does not capture feature
correlations -- e.g., that in NetHack, a red `d` glyph always co-occurs with
the red color. The model treats the glyph and color as independent evidence.

**Why this is acceptable for now:**

1. Feature sets in ROC are small (typically 3-5 features per observation).
   With so few features, the independence assumption has limited room to cause
   errors. The pathological cases for naive Bayes (highly correlated
   high-dimensional features) do not apply.

2. In the generic case, we cannot know which features correlate. Any
   correlation model would require game-specific knowledge or enough data to
   learn correlations, adding complexity with unclear payoff.

3. Empirically, naive Bayes classifiers often perform well despite violating
   the independence assumption (Zhang, 2004, "The optimality of Naive Bayes").
   The posteriors may be poorly calibrated (overconfident), but the MAP
   decision (which identity is most likely) is often correct because the
   ranking is preserved even when absolute probabilities are wrong.

4. If independence proves to be a problem, the natural extension is a
   Dirichlet-Multinomial model over feature pairs rather than individual
   features. This can be done without architectural changes -- only the
   likelihood computation changes.

### Single Object Per Frame

This design does not address the limitation of resolving only one object per
frame. That is an Attention system limitation (only the top-saliency focus
point is used), not a resolution algorithm limitation. The
Dirichlet-Categorical model will work correctly with bipartite matching (Alt 4)
when multi-object attention is implemented -- the likelihood scores become the
cost matrix entries.

## Object Creation: When and How

### Problem

The current system has a simple rule: if `resolve()` returns `None`, the caller
(`ObjectResolver.do_object_resolution`) creates a new Object via
`Object.with_features(fg)`. The resolution algorithm decides "match or not" and
the caller handles creation. This split is clean but raises questions for the
Dirichlet-Categorical design:

1. When exactly does the "new object" hypothesis win?
2. What happens to the new Object's alpha vector?
3. How do we prevent Object proliferation from compounding errors?

### Current Flow

```
resolve() returns Object  -->  increment resolve_count, send downstream
resolve() returns None     -->  Object.with_features(fg), send downstream
```

Both paths produce identical `ResolvedObject` output. Downstream components
(Sequencer, Transformer) do not know whether the Object is new or matched.

### Dirichlet-Categorical Object Creation

The "new object" hypothesis wins when its posterior exceeds all candidate
Objects. Under Option C (likelihood-driven), this happens when:

- No candidate Object's learned feature profile explains the observation well
  (all have low likelihood), OR
- The observation contains features that are rare or unseen across all
  candidates (the uniform model is competitive), OR
- All candidates are far away spatially and/or temporally stale (low priors
  suppress their posteriors even if likelihoods are moderate).

When a new Object is created, the ExpMod initializes its alpha vector from the
current observation: each observed feature gets `prior_alpha + 1` (the base
prior plus the first observation count), and the global feature vocabulary is
updated.

### Preventing Object Proliferation

Object proliferation (creating too many Objects for the same real-world entity)
is the primary failure mode. It compounds: more Objects means more candidates
per frame, which means more graph walking, which means slower resolution and
more opportunities for further mis-resolution.

**Safeguards in this design:**

1. **Spatial/temporal priors suppress stale candidates.** Objects not seen
   recently or far away get low priors, so they don't "steal" matches from
   nearby, recent candidates. This prevents the case where a stale Object
   competes with the correct one, causing a mismatch that creates a third.

2. **Alpha accumulation creates separation over time.** After a few
   observations, a correctly-matched Object has strong alpha values for its
   characteristic features. The uniform "new object" model cannot compete with
   a well-characterized Object for observations that match its profile.

3. **Confidence threshold as safety valve.** If the best posterior is below
   `confidence_threshold` (default 0.5), the system creates a new Object
   rather than making a low-confidence match. This trades Object proliferation
   risk for reduced mis-merge risk. The threshold is tunable.

4. **Telemetry monitoring.** The new-object rate is tracked via the
   `roc.objects_resolved` counter with `{"new": True/False}` attributes (see
   Telemetry section). A sustained high new-object rate is a signal that
   something is wrong.

**What this design does NOT do:**

- No post-hoc Object merging (detecting that two Objects are actually the same
  entity and unifying them). This is a valuable future capability but adds
  significant complexity. The current design is one-pass: resolution decisions
  are final.
- No Object splitting (detecting that one Object has been incorrectly matched
  to two different entities). Same reasoning.
- No maximum Object count or pruning. If proliferation occurs, it must be
  caught via telemetry and addressed by tuning parameters or fixing the
  underlying cause.

## Telemetry

### Design Principle

The Dirichlet-Categorical model introduces probabilistic decisions that are
harder to debug by inspection than the current hard-threshold approach. Good
telemetry is essential for understanding whether the model is working correctly,
diagnosing failure modes, and tuning parameters.

All telemetry uses the existing OpenTelemetry infrastructure in
`roc/reporting/observability.py`. Metrics are exported via OTLP to the
configured collector.

### Existing Metrics (Retained)

These metrics already exist and remain unchanged:

| Metric | Type | Description |
|---|---|---|
| `roc.candidate_objects` | Counter | Total candidate objects scanned per resolution |
| `roc.objects_resolved` | Counter | Total objects resolved, with `new` attribute (True/False) |

### New Metrics

| Metric | Type | Attributes | Description |
|---|---|---|---|
| `roc.resolution.posterior_max` | Histogram | | Highest posterior probability per resolution. Indicates decision confidence. Values near 1.0 mean confident matches; values near 0.5 mean ambiguous. |
| `roc.resolution.posterior_margin` | Histogram | | Difference between the top two posteriors. Large margins mean clear decisions; small margins mean the system is uncertain between two candidates. |
| `roc.resolution.candidates` | Histogram | | Number of candidate Objects per resolution. Tracks candidate set size over time. Growing candidate counts indicate Object proliferation. |
| `roc.resolution.new_object_posterior` | Histogram | | Posterior probability assigned to the "new object" hypothesis per resolution. Tracks how competitive the uniform model is against learned models. |
| `roc.resolution.spatial_distance` | Histogram | | Manhattan distance between observation and matched Object's last-seen position. Detects drift in spatial matching. |
| `roc.resolution.temporal_gap` | Histogram | | Ticks since the matched Object was last seen. Detects temporal staleness in matches. |
| `roc.resolution.alpha_sum` | Histogram | | Sum of alpha values for the matched Object. Proxy for how well-characterized the Object is. Low sums indicate Objects with little evidence. |
| `roc.resolution.decision` | Counter | `outcome`: `match`, `new_object`, `low_confidence` | Resolution outcome. `match` = matched existing Object, `new_object` = "new" hypothesis won, `low_confidence` = best posterior below confidence threshold (also creates new Object, but distinguished from genuine novelty). |

### New Spans

| Span | Description |
|---|---|
| `compute_priors` | Time spent computing spatial and temporal priors. |
| `compute_likelihoods` | Time spent computing Dirichlet-Categorical likelihoods. |
| `compute_posteriors` | Time spent computing and normalizing posteriors. |

These are child spans of the existing `do_object_resolution` span.

### Key Derived Metrics (for dashboards)

These are not emitted directly but can be computed from the raw metrics:

- **Match rate**: `decision{outcome=match} / (decision{outcome=match} + decision{outcome=new_object} + decision{outcome=low_confidence})`
- **Novelty rate**: `decision{outcome=new_object} / total`
- **Confidence distribution**: histogram of `posterior_max` values
- **Ambiguity rate**: fraction of resolutions where `posterior_margin < 0.1`
- **Object growth rate**: `decision{outcome=new_object}` per unit time (from counter derivative)

### Diagnostic Signals

| Signal | Healthy | Unhealthy | Action |
|---|---|---|---|
| Match rate | 80-95% after warmup | < 50% sustained | Objects are proliferating. Check spatial_scale, prior_alpha, excluded features. |
| Novelty rate | Decreasing over time | Constant or increasing | System is not learning. Check alpha update logic. |
| Posterior max (median) | > 0.7 | < 0.5 | Decisions are ambiguous. May need more discriminating features. |
| Posterior margin (median) | > 0.3 | < 0.1 | Top candidates are too similar. Spatial/temporal priors may need adjustment. |
| Candidates per resolution | Stable or slowly growing | Rapidly growing | Object proliferation. Check new-object rate. |
| Spatial distance (p95) | < 5 | > 15 | Matches are being made across large distances. Spatial scale may be too large. |
| Temporal gap (p95) | < 100 | > 500 | Matches are being made to very stale Objects. Temporal scale may be too large. |

## Verification and Acceptance Criteria

### Problem

The current system has no verification mechanism. Resolution decisions are
one-pass and final. There is no way to know if a resolution was correct without
manual inspection. The Dirichlet-Categorical model must be validated both at the
unit level (does the math work?) and at the integration level (does it produce
correct resolutions when running a real game?).

### Unit Test Verification

Unit tests verify the algorithm's behavior on controlled scenarios with known
correct answers. These tests use synthetic feature nodes and Objects, not a
running game.

**Scenarios to test:**

1. **Exact match.** Observation features exactly match a known Object's
   profile. Expected: match with high posterior (> 0.9).

2. **Partial match.** Observation has 2 of 3 features matching a known Object.
   Expected: match with moderate posterior, higher than "new object."

3. **No match.** Observation features are entirely unlike any known Object.
   Expected: "new object" hypothesis wins.

4. **Spatial disambiguation.** Two Objects have identical feature profiles but
   different last-seen positions. Observation is near one of them. Expected:
   match to the spatially closer Object.

5. **Temporal decay.** Object A was seen last tick, Object B was seen 200 ticks
   ago. Both have similar features. Expected: Object A is preferred.

6. **Alpha accumulation.** After resolving the same Object 20 times with
   consistent features, it should have high confidence and resist matching
   to observations with different features.

7. **Cold start.** First observation ever. No existing Objects. Expected: "new
   object" wins, new Object is created.

8. **Confidence threshold.** Ambiguous observation where the best posterior is
   below the confidence threshold. Expected: new Object created with
   `decision=low_confidence`.

### Integration Test Verification: Game Runs

The definitive test is running the agent against a real game and verifying that
object resolution produces sensible results. This requires:

1. **Run the agent** against NetHack (or another gym) for N steps (e.g., 200
   steps to get through initial exploration).

2. **Collect telemetry** from the run, specifically the metrics defined above.

3. **Verify key invariants:**

   - **Match rate stabilizes.** After an initial warmup period (first ~50
     ticks, where everything is new), the match rate should rise and stabilize.
     A system that keeps creating new Objects indefinitely is broken.

   - **Object count stabilizes.** The total number of unique Objects should
     plateau as the agent revisits familiar areas. Unbounded Object growth
     indicates proliferation.

   - **Known objects are re-identified.** When the agent returns to a
     previously visited location, objects at that location should match their
     previously-created Objects (same Object.uuid). This can be verified by
     recording (x, y, Object.uuid) tuples and checking for consistency across
     revisits.

   - **Distinct objects are not merged.** Two visually different entities at
     different positions should produce different Objects. This can be checked
     by ensuring that Objects with very different feature profiles are not
     assigned the same Object.uuid.

4. **Compare to baseline.** Run the same game with
   `SymmetricDifferenceResolution` and compare:
   - Total Objects created
   - Match rate over time
   - Qualitative inspection of resolution decisions (logged via observability
     events)

### Acceptance Criteria

| Criterion | Threshold | Rationale |
|---|---|---|
| Match rate after warmup (tick > 50) | >= 70% | Most observations should match known Objects once the scene is populated. |
| Object growth rate after warmup | < 1 new Object per 10 ticks | Object creation should slow dramatically after initial exploration. |
| Median posterior confidence | > 0.6 | The model should be reasonably confident in its decisions. |
| No regressions vs. symmetric difference | Object count within 2x | The new model should not create drastically more Objects than the baseline. If it does, the parameters need tuning. |
| Zero division-by-zero or NaN in posteriors | 0 occurrences | Log-space arithmetic must be numerically stable. |
| Per-frame resolution time | < 2x current | Bayesian computation should not significantly slow the pipeline. |

These thresholds are initial estimates. They will be refined based on the first
game runs. The telemetry infrastructure makes it easy to measure all of these.

### Game Run Monitoring Procedure

The implementation should include a monitoring script or notebook that:

1. Starts a game run with the Dirichlet-Categorical ExpMod active.
2. Queries the OpenTelemetry collector for resolution metrics.
3. Plots match rate, object count, and posterior confidence over time.
4. Flags any violations of acceptance criteria.
5. Produces a summary comparing to the symmetric-difference baseline.

This is not automated CI (game runs are slow and non-deterministic). It is a
manual verification step performed after implementation and after any parameter
changes.

## Configurable Parameters

All parameters live as class attributes on the ExpMod, overridable via config:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prior_alpha` | float | 1.0 | Base Dirichlet concentration. 1.0 = uniform, 0.5 = Jeffreys prior. |
| `spatial_scale` | float | 3.0 | Manhattan distance at which spatial prior decays to ~37% (1/e). |
| `temporal_scale` | float | 50.0 | Ticks since last seen at which temporal prior decays to ~37%. |
| `confidence_threshold` | float | 0.5 | Minimum posterior to accept a match. Below this, create new Object. |
| `excluded_feature_labels` | set[str] | {} | Feature node labels to exclude from likelihood computation. |

## Dependencies

No new dependencies are required. The Dirichlet-Categorical model is
analytically tractable (conjugate prior -- the posterior has a closed-form
solution), so the implementation is ~50 lines of arithmetic rather than a
library call.

### What we use

- `math.log`, `math.exp` -- stdlib. Log-space arithmetic for likelihood
  computation.
- `scipy.special.logsumexp` -- already a project dependency. Numerically
  stable normalization in log-space. Handles edge cases (very large/small
  exponents) that a naive `log(sum(exp(...)))` would not.
- `collections.defaultdict` -- stdlib. Alpha vector storage keyed by feature
  string.

### What we considered and rejected

**numpy.** Alpha vectors have 3-5 entries and candidate sets are in the tens.
At these sizes, Python dict lookups and `math.log` are faster than numpy array
creation overhead. Numpy is valuable for vectorized operations on large arrays;
our data is too small to benefit.

**scipy.stats.dirichlet.** The full Dirichlet distribution class supports
sampling, PDF evaluation, entropy, etc. We only need the posterior predictive
(`alpha_j / sum(alpha)`), which is a single division. Importing the full class
adds API surface for no benefit.

**scikit-learn CategoricalNB.** Conceptually similar (categorical naive Bayes),
but designed for batch classification with training/test splits. Our use case
is online single-instance updating in a streaming pipeline. Wrapping sklearn's
batch API into per-frame resolution would be more work than implementing the
math directly.

**PyMC / Stan / probabilistic programming languages.** PPLs are for complex
inference problems where the posterior cannot be derived analytically. The
Dirichlet-Categorical is a conjugate pair -- the update rule is just counting.
A PPL would add a heavy dependency for a problem solved by addition.

**filterpy.** Mentioned in `design/object-resolution-alternatives.md` for
Kalman filter tracking (Alt 7). Not needed for this design since we chose
exponential decay over motion prediction. Worth revisiting if Risk 7 (spatial
prior penalizing fast-moving objects) proves to be a real problem in practice.

### Rationale

One of the Dirichlet-Categorical model's genuine strengths is that it is simple
enough to implement directly. Pulling in a library would add API surface,
version constraints, and abstraction mismatch for ~50 lines of well-understood
math. The only external function we use (`logsumexp`) is already available and
solves a real numerical stability problem that we should not reimplement.

## ExpMod Registration

```python
class DirichletCategoricalResolution(ObjectResolutionExpMod):
    name = "dirichlet-categorical"
    ...
```

Selected via config `expmods_use = [("object-resolution", "dirichlet-categorical")]`
or used as default in `ObjectResolutionExpMod.get(default="dirichlet-categorical")`.

## Performance Impact

### Computational Cost

The per-frame cost is nearly identical to the current system:

| Operation | Current | Dirichlet-Categorical |
|---|---|---|
| Candidate discovery | O(F x G x O) graph walk | Same |
| Distance/likelihood | O(C x Fe) set ops | O(C x Fe) float ops |
| Prior computation | N/A | O(C) lookups + arithmetic |
| Normalization | N/A | O(C) additions |
| Alpha update | N/A | O(Fe) increments |

Where F = feature nodes, G = feature groups per node, O = objects per group,
C = total candidates, Fe = features per object.

The graph walk dominates. The arithmetic overhead is negligible.

### Memory Cost

~200 bytes per Object (alpha dict + position/tick). Negligible even with
thousands of Objects.

### Accuracy Impact (Expected)

- **Fewer spurious Objects**: soft matching tolerates occasional missing features,
  reducing Object proliferation. This is a compounding benefit -- fewer Objects
  means smaller candidate sets in future frames.
- **Better discrimination**: alpha profiles encode how reliably each feature
  appears, not just whether it's present right now.
- **Spatial/temporal context**: nearby, recently-seen Objects are preferred,
  disambiguating visually identical entities.

## Implementation Plan

The implementation is split into four phases. Each phase is independently
shippable and does not leave the system in a half-built state. This phasing
reduces risk, enables baseline measurement before algorithm changes, and
isolates infrastructure bugs from algorithm bugs.

### Why phased

The design mixes two categories of changes:

1. **Infrastructure** that benefits the existing system regardless of algorithm
   (Object node fields, ResolutionContext, telemetry).
2. **The algorithm itself** (Dirichlet-Categorical math, alpha vectors,
   priors).

Building them together means we cannot distinguish "the infrastructure broke
something" from "the algorithm is wrong." More importantly, we need baseline
telemetry from the current system before we can evaluate whether the new system
is better -- the acceptance criteria explicitly call for comparison to the
symmetric-difference baseline.

### Phase 1: Infrastructure

**Goal:** Enrich the Object node and resolution interface without changing
behavior.

**Changes:**

1. Add `last_x: XLoc | None`, `last_y: YLoc | None`, `last_tick: int` fields
   to the `Object` node in `roc/object.py`.
2. Add `ResolutionContext` dataclass with `x`, `y`, `tick` fields.
3. Update `ObjectResolutionExpMod.resolve` signature to accept
   `context: ResolutionContext`.
4. Update `SymmetricDifferenceResolution.resolve` to accept (and ignore) the
   context parameter.
5. Update `ObjectResolver.do_object_resolution` to:
   - Construct `ResolutionContext` from the focus point coordinates and
     current tick.
   - Pass it to `resolve()`.
   - Update `last_x`, `last_y`, `last_tick` on the Object after resolution
     (for both matched and newly created Objects).
6. Run full test suite. Verify no regressions.

**Deliverable:** Existing system works identically, but Objects now carry
position/tick metadata. This data is useful for debugging, analysis, and any
future resolution strategy.

**Risk:** Low. Additive changes only. The SymmetricDifferenceResolution ignores
the new context parameter.

### Phase 2: Telemetry and Baseline

**Goal:** Add resolution telemetry to the existing system and establish baseline
metrics for comparison.

**Changes:**

1. Add new metrics to `ObjectResolver` and/or
   `SymmetricDifferenceResolution`:
   - `roc.resolution.candidates` (histogram) -- candidate count per
     resolution. (Replaces or supplements the existing
     `roc.candidate_objects` counter with a histogram for distribution
     analysis.)
   - `roc.resolution.decision` (counter) -- with `outcome` attribute
     (`match` or `new_object`). (Supplements the existing
     `roc.objects_resolved` counter with clearer semantics.)
   - `roc.resolution.spatial_distance` (histogram) -- Manhattan distance
     between observation and matched Object's `last_x`/`last_y`.
   - `roc.resolution.temporal_gap` (histogram) -- ticks since matched
     Object's `last_tick`.
2. Add child spans under `do_object_resolution` for `find_candidate_objects`
   timing (already exists as a span, verify it is a child span).
3. Run a game session (e.g., 200 ticks). Record baseline values for:
   - Match rate (matched / total resolutions).
   - Object creation rate over time.
   - Candidate count distribution.
   - Spatial distance distribution for matches.
   - Temporal gap distribution for matches.
4. Document baseline values in the test results or a run log.

**Deliverable:** Existing system has better observability. We have concrete
baseline numbers to compare Phase 4 results against. This directly addresses
Risk 9 from Appendix A ("we don't actually know what good looks like yet").

**Note:** Some metrics from the Telemetry section do not apply to the
symmetric-difference algorithm (e.g., `posterior_max`, `posterior_margin`,
`new_object_posterior`, `alpha_sum`). These are added in Phase 3 when the
Dirichlet-Categorical ExpMod is implemented.

### Phase 3: Core Algorithm

**Goal:** Implement the Dirichlet-Categorical ExpMod with unit tests. No game
runs required.

**Changes:**

1. Implement `DirichletCategoricalResolution(ObjectResolutionExpMod)` with:
   - `name = "dirichlet-categorical"`
   - Alpha vector storage (side dict keyed by `Object.id`).
   - `_compute_priors()` -- spatial and temporal exponential decay.
   - `_compute_likelihoods()` -- log-space Dirichlet posterior predictive.
   - `_compute_posteriors()` -- Bayes rule with `logsumexp` normalization.
   - `resolve()` -- MAP decision with confidence threshold.
   - Alpha update on match.
   - Configurable parameters: `prior_alpha`, `spatial_scale`,
     `temporal_scale`, `confidence_threshold`, `excluded_feature_labels`.
2. Add Dirichlet-Categorical-specific metrics:
   - `roc.resolution.posterior_max` (histogram).
   - `roc.resolution.posterior_margin` (histogram).
   - `roc.resolution.new_object_posterior` (histogram).
   - `roc.resolution.alpha_sum` (histogram).
   - Update `roc.resolution.decision` to include `low_confidence` outcome.
3. Add child spans: `compute_priors`, `compute_likelihoods`,
   `compute_posteriors`.
4. Unit tests for the 8 scenarios from the Verification section:
   - Exact match (high posterior > 0.9).
   - Partial match (moderate posterior, beats "new object").
   - No match ("new object" wins).
   - Spatial disambiguation (closer Object wins).
   - Temporal decay (recent Object preferred).
   - Alpha accumulation (20 observations, high confidence).
   - Cold start (no existing Objects, "new" wins).
   - Confidence threshold (ambiguous posterior, new Object created with
     `low_confidence` outcome).
5. Run full test suite. Verify no regressions. The new ExpMod is registered
   but not the default -- it does not affect existing tests.

**Deliverable:** New ExpMod exists, is unit-tested, but is not the default.
The system continues to use `SymmetricDifferenceResolution` unless explicitly
configured otherwise.

**Risk:** Moderate. Math bugs (NaN posteriors, log-space underflow, division
by zero) are caught by unit tests before they can corrupt a game run.

### Phase 4: Integration and Tuning

**Goal:** Validate the Dirichlet-Categorical ExpMod against a real game and
compare to the Phase 2 baseline.

**Steps:**

1. Configure the agent to use the Dirichlet-Categorical ExpMod:
   `expmods_use = [("object-resolution", "dirichlet-categorical")]`
2. Run a game session (same duration and conditions as Phase 2 baseline).
3. Collect telemetry. Compare to Phase 2 baseline:
   - Match rate after warmup (acceptance: >= 70%).
   - Object growth rate after warmup (acceptance: < 1 per 10 ticks).
   - Median posterior confidence (acceptance: > 0.6).
   - Total Object count vs. baseline (acceptance: within 2x).
   - Zero NaN/infinity in posteriors.
   - Per-frame resolution time vs. baseline (acceptance: < 2x).
4. If acceptance criteria are not met:
   - Check diagnostic signals (see Telemetry section table).
   - Adjust parameters (`prior_alpha`, `spatial_scale`, `temporal_scale`,
     `confidence_threshold`).
   - Re-run and compare.
5. If acceptance criteria are met, update the default in
   `ObjectResolver.do_object_resolution` to use `"dirichlet-categorical"`.
6. Document final parameter values, baseline comparison, and any tuning
   rationale.

**Deliverable:** Dirichlet-Categorical ExpMod is validated against a real game
with measured improvement (or measured parity) versus the baseline. Decision
to make it the default is based on data, not assumption.

**Risk:** This is where we discover whether the design actually works. Risks
4, 7, 8, and 10 from Appendix A are most likely to surface here. The phased
approach ensures we have baseline data and unit-tested code before reaching
this point.

### Phase dependencies

```
Phase 1 (infrastructure) --> Phase 2 (telemetry needs last_x/last_y/last_tick)
Phase 2 (baseline)       --> Phase 4 (comparison needs baseline data)
Phase 1 (interface)      --> Phase 3 (ExpMod needs ResolutionContext)
Phase 3 (algorithm)      --> Phase 4 (game runs need the ExpMod to exist)
```

Phases 2 and 3 can run in parallel after Phase 1 is complete.

## Open Questions

- Should the `ResolutionContext` also carry the full saliency map or neighboring
  features? This would enable context-aware resolution (e.g., "this Object is
  usually near walls") but adds complexity.
- Should alpha vectors persist across agent restarts (graph storage) or are they
  ephemeral (reset each episode)?
- Is MAP decision sufficient, or do downstream components (Transformer, Action)
  benefit from receiving the full posterior distribution?

## References

- Dirichlet-Categorical conjugate model: Murphy, K.P. (2012). "Machine
  Learning: A Probabilistic Perspective." MIT Press. Section 3.4.
- Bayesian inference fundamentals: Gelman, A. et al. (2013). "Bayesian Data
  Analysis." 3rd edition. Chapter 2.
- Naive Bayes optimality: Zhang, H. (2004). "The optimality of Naive Bayes."
  Proceedings of FLAIRS Conference.
- Bayesian concept learning: Xu, F. & Tenenbaum, J.B. (2007). "Word Learning
  as Bayesian Inference." Psychological Review, 114(2), 245-272.
- SORT tracker (spatial/temporal priors in object tracking): Bewley, A. et al.
  (2016). "Simple Online and Realtime Tracking." IEEE ICIP.
  https://arxiv.org/abs/1602.00763
- Jeffreys prior: Jeffreys, H. (1946). "An invariant form for the prior
  probability in estimation problems." Proceedings of the Royal Society A,
  186(1007), 453-461.
- Conjugate prior tables: https://en.wikipedia.org/wiki/Conjugate_prior

---

## Appendix A: Design Critique and Risk Assessment

This appendix documents an objective critique of the design, conducted during
the design phase. These risks should be monitored during implementation and
evaluated after initial game runs.

### Risk 1: Overengineering for uncertain payoff

The current system is ~30 lines of set-difference logic. It is simple,
debuggable, and works. This design replaces it with a multi-step Bayesian
inference pipeline. The problems the design identifies (hardcoded features, no
spatial context, arbitrary threshold) could all be fixed incrementally: add a
configurable feature set, add a spatial distance filter, tune the threshold.
Those changes would take an afternoon. The Dirichlet-Categorical model is a
more elegant solution to the same problems, but "elegant" does not always mean
"better for a research project that may pivot."

### Risk 2: "Principled over hand-tuned" is somewhat misleading

We traded one hand-tuned parameter (`distance <= 1`) for five (`prior_alpha`,
`spatial_scale`, `temporal_scale`, `confidence_threshold`,
`excluded_feature_labels`). The design argues these have "principled defaults,"
but `spatial_scale=3.0` and `temporal_scale=50.0` are just as arbitrary as the
current threshold -- they are dressed in mathematical notation but not derived
from first principles. We have not eliminated tuning; we have moved it to a
higher-dimensional space. The defaults may work or they may not, and with five
interacting parameters, finding the right combination is harder than tuning one
threshold.

### Risk 3: The independence assumption may matter more than acknowledged

The design cites Zhang (2004) to argue naive Bayes works despite independence
violations. That paper studied classification with many training examples per
class. Object resolution has very few observations per Object -- sometimes just
1 or 2 before a match decision matters. With tiny sample sizes, model
misspecification has proportionally larger impact because there is less data to
overcome it. The design's "3-5 features so limited room for error" argument
cuts both ways: with so few features, each feature has outsized influence, and
getting the probability of even one feature wrong can flip the decision.

### Risk 4: Early mistakes are permanent (highest-severity risk)

The design is one-pass with no post-hoc correction. This is the most serious
structural weakness. During cold start, when alpha profiles are weak, the
system will make wrong matches. Those wrong matches update the alpha vector
with incorrect feature associations, making the Object's profile reflect a
mixture of different entities. This is self-reinforcing: a polluted Object
competes more strongly in future frames because it has accumulated alpha mass,
potentially attracting further incorrect matches. The same alpha accumulation
that makes correct Objects more confident makes incorrect Objects more
entrenched.

The design acknowledges this ("no merging, no splitting") but treats it as a
future enhancement rather than a fundamental design risk. In practice, this
means the system's steady-state accuracy depends heavily on cold-start
performance, which is exactly when the model is weakest.

### Risk 5: Game-agnosticism is aspirational and untested

The design makes the claim of game-agnosticism, but all acceptance criteria are
measured against NetHack. If we tune parameters to make NetHack work well, we
have not demonstrated generality -- we have demonstrated that a Bayesian model
can be tuned for one game, same as the current system. The game-agnostic claim
remains untested until the system runs on a second, meaningfully different
environment. The risk is that "generic" design decisions add complexity without
delivering real portability.

### Risk 6: Feature string keys are fragile

Alpha vectors key on `str(f)` -- the string representation of feature nodes.
This couples the Bayesian model to the exact `__str__` implementation of every
FeatureNode subclass. If a feature's string representation changes (e.g., a new
attribute is added, formatting changes), all accumulated alpha evidence becomes
stale because the keys no longer match. This is a hidden coupling between the
perception layer and the resolution layer -- ironic for a design motivated by
decoupling them.

### Risk 7: The spatial prior penalizes fast-moving objects

The exponential decay assumes objects do not move much between frames. For
fast-moving entities (a monster charging across the screen), the prior actively
penalizes the correct match because the entity has moved far from its last-seen
position. The design acknowledged the Kalman filter (Alt 7) handles this
correctly via motion prediction, then chose not to implement it. In
environments with lots of motion, the spatial prior may cause more harm than
good -- it could systematically fragment fast-moving entities into chains of
single-observation Objects.

### Risk 8: The "new object" model's behavior drifts over time

The new-object likelihood depends on the global feature vocabulary size (N),
which grows monotonically and never shrinks. Two runs with different game
experiences will have different vocabulary sizes and therefore different
effective new-object thresholds, even with identical parameter settings. As a
run progresses, the new-object hypothesis gets progressively weaker regardless
of whether genuinely new entities are appearing. This is a hidden
non-stationarity that no single parameter controls.

### Risk 9: Telemetry diagnoses but does not fix

The design specifies extensive telemetry, but all diagnostic signals require
human interpretation. There are no automated corrective mechanisms. If match
rate drops below 50%, a human must inspect metrics, hypothesize a cause, adjust
parameters, and re-run. This is acceptable for research but means the system
cannot self-correct. The acceptance criteria are "initial estimates" that "will
be refined" -- which means we do not actually know what good looks like yet.

### Risk 10: Building on an unvalidated assumption

The design assumes that object identity is well-modeled by feature frequency
distributions. But objects in games may be better identified by relational
properties (what is next to them, what corridor they are in), behavioral
properties (how they move, what they do), or temporal patterns (they always
appear after a specific event). The Dirichlet model can only capture "this
object usually has these features." If identity turns out to depend more on
context than on features, the entire Bayesian apparatus is optimizing the wrong
thing -- and we will not know until we try it.

### Summary Assessment

The biggest risk is not any single technical issue -- it is that the design's
complexity may not be justified by its benefits over simpler alternatives, and
we will not know until after implementation. The irrecoverability of early
mistakes (Risk 4) and the untested generality claim (Risk 5) are the two to
watch most carefully. If one thing could make this design fail in practice, it
is the compounding effect of cold-start errors with no correction mechanism.
