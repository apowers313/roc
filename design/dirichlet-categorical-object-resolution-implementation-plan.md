# Implementation Plan for Dirichlet-Categorical Object Resolution

## Overview

Replace the `SymmetricDifferenceResolution` ExpMod with a Bayesian
Dirichlet-Categorical model that computes posterior probabilities over object
identities. The implementation follows the four-phase plan from the design
document, with each phase independently shippable and testable.

The core deliverable is `DirichletCategoricalResolution`, a new
`ObjectResolutionExpMod` that:
- Computes spatial and temporal priors via exponential decay
- Computes likelihoods via Dirichlet posterior predictive probabilities
- Makes MAP decisions with a confidence threshold
- Updates alpha vectors online after each match

## Phase Breakdown

### Phase 1: Infrastructure -- Object Node and Resolution Interface

**What this phase accomplishes**: Enrich the Object node with position/tick
tracking and update the resolution interface to pass spatial/temporal context.
No behavioral changes -- the existing symmetric-difference algorithm continues
to work identically.

**Duration**: 1 day

**Tests to Write First**:

- `tests/unit/test_object.py` -- extend existing tests:

  ```python
  class TestObject:
      def test_last_position_defaults(self):
          """Already exists. Verify last_x, last_y, last_tick defaults."""
          o = Object()
          assert o.last_x is None
          assert o.last_y is None
          assert o.last_tick == 0

      def test_last_position_assignment(self):
          """Verify position and tick can be set."""
          o = Object()
          o.last_x = 5
          o.last_y = 10
          o.last_tick = 42
          assert o.last_x == 5
          assert o.last_y == 10
          assert o.last_tick == 42


  class TestResolutionContext:
      def test_constructor(self):
          """Already exists. Verify x, y, tick fields."""
          ctx = ResolutionContext(x=XLoc(5), y=YLoc(10), tick=42)
          assert ctx.x == 5
          assert ctx.y == 10
          assert ctx.tick == 42


  class TestSymmetricDifferenceResolution:
      def test_resolve_accepts_context(self):
          """Verify symmetric-difference accepts and ignores the context param."""
          # Same as existing test_resolve_no_candidates but explicitly
          # verifying the context parameter is accepted without error
  ```

**Implementation**:

- `roc/object.py`:
  1. Verify `last_x`, `last_y`, `last_tick` fields exist on `Object` (already
     added per git status).
  2. Verify `ResolutionContext` dataclass exists with `x`, `y`, `tick` fields
     (already added per git status).
  3. Verify `ObjectResolutionExpMod.resolve` signature accepts
     `context: ResolutionContext` (already added).
  4. Verify `SymmetricDifferenceResolution.resolve` accepts and ignores context
     (already added).
  5. Verify `ObjectResolver.do_object_resolution` constructs a
     `ResolutionContext` and passes it to `resolve()`, then updates `last_x`,
     `last_y`, `last_tick` on the Object after resolution (already added).

  The git status shows `roc/object.py` and `tests/unit/test_object.py` are
  already modified. This phase primarily involves verifying and testing those
  changes.

**Dependencies**:
- External: None
- Internal: None

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_object.py -v`
2. Expected: All tests pass, including new position/tick tests
3. Run: `make test`
4. Expected: Full suite passes with no regressions
5. Run: `make lint`
6. Expected: No new type errors or lint issues

**Success Criteria**:
- Object node carries `last_x`, `last_y`, `last_tick` fields
- `ResolutionContext` dataclass exists and is passed through the resolution pipeline
- `SymmetricDifferenceResolution` accepts and ignores the context
- `ObjectResolver.do_object_resolution` populates position/tick after resolution
- All existing tests pass unchanged

---

### Phase 2: Telemetry and Baseline Measurement

**What this phase accomplishes**: Add resolution-specific telemetry metrics to
the existing system and establish baseline measurements. These metrics are
needed to evaluate the Dirichlet-Categorical model against the current system
in Phase 4.

**Duration**: 1-2 days

**Tests to Write First**:

- `tests/unit/test_object.py` -- extend with telemetry verification:

  ```python
  class TestSymmetricDifferenceResolution:
      def test_resolution_decision_counter_match(self):
          """Verify decision counter records 'match' outcome."""
          resolution = SymmetricDifferenceResolution()
          # Set up mocks so resolve() returns a matched object
          # Verify resolution.decision_counter was incremented with
          # attributes={"outcome": "match"}

      def test_resolution_decision_counter_new(self):
          """Verify decision counter records 'new_object' outcome."""
          # Set up mocks so resolve() returns None
          # Verify counter incremented with {"outcome": "new_object"}

      def test_candidates_histogram_recorded(self):
          """Verify candidate count is recorded as histogram."""
          # After calling resolve(), verify the histogram was updated
  ```

- `tests/unit/test_object_telemetry.py` -- new file for telemetry-specific
  tests:

  ```python
  class TestResolutionTelemetry:
      def test_spatial_distance_recorded_on_match(self):
          """When an object is matched, spatial distance is recorded."""
          # Create object with last_x=5, last_y=5
          # Resolve at x=7, y=8
          # Verify spatial_distance histogram recorded manhattan distance 5

      def test_temporal_gap_recorded_on_match(self):
          """When an object is matched, temporal gap is recorded."""
          # Create object with last_tick=10
          # Resolve at tick=42
          # Verify temporal_gap histogram recorded gap of 32

      def test_no_spatial_distance_for_new_object(self):
          """New objects have no last position, so no spatial distance recorded."""

      def test_no_temporal_gap_for_new_object(self):
          """New objects have no meaningful temporal gap."""
  ```

**Implementation**:

- `roc/object.py` -- add metrics to `SymmetricDifferenceResolution` and
  `ObjectResolver`:

  ```python
  class SymmetricDifferenceResolution(ObjectResolutionExpMod):
      # Existing counter retained
      candidate_object_counter = Observability.meter.create_counter(...)

      # New metrics
      candidates_histogram = Observability.meter.create_histogram(
          "roc.resolution.candidates",
          unit="count",
          description="number of candidate objects per resolution",
      )
      decision_counter = Observability.meter.create_counter(
          "roc.resolution.decision",
          unit="resolution",
          description="resolution outcome: match, new_object, or low_confidence",
      )

      def resolve(self, feature_nodes, feature_group, context):
          candidates = self._find_candidates(feature_nodes)
          self.candidates_histogram.record(len(candidates))
          # ... existing logic ...
          if best_dist <= 1:
              self.decision_counter.add(1, attributes={"outcome": "match"})
              return best_obj
          self.decision_counter.add(1, attributes={"outcome": "new_object"})
          return None
  ```

- `roc/object.py` -- add post-resolution metrics to `ObjectResolver`:

  ```python
  class ObjectResolver(Component):
      spatial_distance_histogram = Observability.meter.create_histogram(
          "roc.resolution.spatial_distance",
          unit="cells",
          description="manhattan distance between observation and matched object",
      )
      temporal_gap_histogram = Observability.meter.create_histogram(
          "roc.resolution.temporal_gap",
          unit="ticks",
          description="ticks since matched object was last seen",
      )

      def do_object_resolution(self, e):
          # ... existing resolution logic ...
          if o is not None:  # matched existing object
              if o.last_x is not None and o.last_y is not None:
                  dist = abs(int(x) - o.last_x) + abs(int(y) - o.last_y)
                  self.spatial_distance_histogram.record(dist)
              if o.last_tick > 0:
                  gap = current_tick - o.last_tick
                  self.temporal_gap_histogram.record(gap)
  ```

**Dependencies**:
- External: None (OpenTelemetry already a dependency)
- Internal: Phase 1 (needs `last_x`, `last_y`, `last_tick` on Object)

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_object.py tests/unit/test_object_telemetry.py -v`
2. Expected: All tests pass
3. Run: `make test`
4. Expected: Full suite passes
5. Manual verification: If OpenTelemetry collector is available, run a short
   game session and verify metrics appear:
   - `roc.resolution.candidates` histogram has entries
   - `roc.resolution.decision` counter has `match` and `new_object` outcomes
   - `roc.resolution.spatial_distance` histogram has entries for matches
   - `roc.resolution.temporal_gap` histogram has entries for matches
6. Record baseline values from a 200-tick game run for Phase 4 comparison

**Success Criteria**:
- All four new metrics emit data during resolution
- `decision` counter correctly distinguishes match vs. new_object outcomes
- Spatial distance and temporal gap only recorded for matched (not new) objects
- Baseline measurements documented for Phase 4 comparison
- No performance regression from metric recording

---

### Phase 3: Core Dirichlet-Categorical Algorithm

**What this phase accomplishes**: Implement the full Dirichlet-Categorical
resolution ExpMod with all five algorithm steps (candidates, priors,
likelihoods, posteriors, decision + update). Unit tests verify correctness on
synthetic data. The ExpMod is registered but not the default.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_dirichlet_resolution.py` -- new file:

  ```python
  """Unit tests for DirichletCategoricalResolution."""

  import math
  from unittest.mock import MagicMock, PropertyMock, patch

  import pytest

  from roc.location import XLoc, YLoc
  from roc.object import (
      DirichletCategoricalResolution,
      FeatureGroup,
      Object,
      ResolutionContext,
  )


  @pytest.fixture(autouse=True)
  def mock_db():
      mock = MagicMock()
      mock.strict_schema = False
      mock.strict_schema_warns = False
      with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
          yield mock


  def make_feature_node(label: str, str_repr: str) -> MagicMock:
      """Create a mock FeatureNode with label and string representation."""
      fn = MagicMock()
      fn.labels = {label, "FeatureNode"}
      fn.configure_mock(**{"__str__": MagicMock(return_value=str_repr)})
      return fn


  def make_object_with_position(
      x: int | None, y: int | None, tick: int
  ) -> Object:
      """Create an Object with last-seen position and tick."""
      o = Object()
      o.last_x = x
      o.last_y = y
      o.last_tick = tick
      return o


  class TestColdStart:
      def test_no_existing_objects_creates_new(self):
          """First observation ever -- 'new object' hypothesis wins."""
          resolution = DirichletCategoricalResolution()
          fn1 = make_feature_node("SingleNode", "SingleNode(a)")
          fn2 = make_feature_node("ColorNode", "ColorNode(red)")
          fg = MagicMock()
          ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)

          # No candidates found (no predecessors)
          mock_nl = MagicMock()
          mock_nl.select.return_value = []
          fn1.predecessors = mock_nl
          fn2.predecessors = mock_nl

          result = resolution.resolve([fn1, fn2], fg, ctx)
          assert result is None  # None means "create new object"


  class TestExactMatch:
      def test_exact_feature_match_returns_object(self):
          """Observation exactly matches a known Object's features."""
          resolution = DirichletCategoricalResolution()
          fn1 = make_feature_node("SingleNode", "SingleNode(a)")
          fn2 = make_feature_node("ColorNode", "ColorNode(red)")

          # Pre-populate alpha vector for an object
          obj = make_object_with_position(5, 5, tick=10)
          resolution._alphas[obj.id] = {
              "SingleNode(a)": 11.0,  # prior_alpha + 10 observations
              "ColorNode(red)": 11.0,
          }
          resolution._global_vocab = {"SingleNode(a)", "ColorNode(red)"}

          # Mock graph walk to return this object as candidate
          # ... (mock predecessors to return obj)

          ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
          # When the candidate is found and scored, it should win with
          # high posterior (> 0.9)


  class TestPartialMatch:
      def test_two_of_three_features_still_matches(self):
          """2 of 3 features match -- should match with moderate posterior."""
          resolution = DirichletCategoricalResolution()
          # Object has alpha for features A, B, C
          # Observation has A, B, D
          # Object should still match (prior_alpha handles unseen feature D)


  class TestNoMatch:
      def test_entirely_different_features_creates_new(self):
          """Features are entirely unlike any known Object -- new wins."""
          resolution = DirichletCategoricalResolution()
          # Object has strong alpha for features A, B, C
          # Observation has features X, Y, Z (none matching)
          # "new object" hypothesis should win


  class TestSpatialDisambiguation:
      def test_closer_object_preferred(self):
          """Two objects with identical features, different positions."""
          resolution = DirichletCategoricalResolution()
          # obj_near at (5, 5), obj_far at (50, 50)
          # Both have same alpha vectors
          # Observation at (6, 6)
          # obj_near should win due to spatial prior


  class TestTemporalDecay:
      def test_recent_object_preferred(self):
          """Two similar objects, one seen recently, one stale."""
          resolution = DirichletCategoricalResolution()
          # obj_recent last seen tick=99, obj_stale last seen tick=1
          # Both at same position with same alphas
          # Observation at tick=100
          # obj_recent should win due to temporal prior


  class TestAlphaAccumulation:
      def test_well_characterized_object_resists_mismatch(self):
          """After 20 observations, object should resist different features."""
          resolution = DirichletCategoricalResolution()
          # Object has alpha=21.0 for features A, B, C (20 updates + prior)
          # Observation has features D, E, F
          # Even though observation is at same position, the likelihood
          # mismatch should cause "new object" to win


  class TestConfidenceThreshold:
      def test_ambiguous_posterior_creates_new(self):
          """Best posterior below threshold creates new with low_confidence."""
          resolution = DirichletCategoricalResolution()
          resolution.confidence_threshold = 0.8
          # Set up scenario where best posterior is ~0.6
          # resolve() should return None


  class TestAlphaUpdate:
      def test_alpha_updated_after_match(self):
          """After resolve() matches an object, its alphas are incremented."""
          resolution = DirichletCategoricalResolution()
          # Set up a match scenario
          # After resolve(), verify alpha values increased by 1 for each
          # observed feature

      def test_new_object_gets_initial_alphas(self):
          """When resolve() returns None and caller creates object,
          the ExpMod should be ready to initialize alphas on first update."""
          # Verify alpha initialization for new objects


  class TestFeatureExclusion:
      def test_excluded_features_ignored_in_likelihood(self):
          """Features with excluded labels don't affect likelihood."""
          resolution = DirichletCategoricalResolution()
          resolution.excluded_feature_labels = {"PositionNode"}
          # Create feature nodes including a PositionNode
          # Verify PositionNode doesn't affect the match decision


  class TestNumericalStability:
      def test_no_nan_with_many_features(self):
          """Log-space arithmetic stays stable with many features."""
          resolution = DirichletCategoricalResolution()
          # Create observation with 20 features
          # Verify no NaN or Inf in posterior

      def test_no_nan_with_zero_candidates(self):
          """Edge case: zero candidates should return None cleanly."""

      def test_no_nan_with_single_candidate(self):
          """Edge case: exactly one candidate, posteriors still valid."""


  class TestComputePriors:
      def test_spatial_weight_decay(self):
          """Spatial weight decreases with manhattan distance."""
          resolution = DirichletCategoricalResolution()
          resolution.spatial_scale = 3.0
          # Object at (0, 0), observation at (3, 0) -> distance 3
          # weight = exp(-3/3) = exp(-1) ~= 0.368
          # Verify approximately correct

      def test_temporal_weight_decay(self):
          """Temporal weight decreases with tick gap."""
          resolution = DirichletCategoricalResolution()
          resolution.temporal_scale = 50.0
          # Object last_tick=0, current tick=50
          # weight = exp(-50/50) = exp(-1) ~= 0.368

      def test_no_position_gets_uniform_spatial(self):
          """Object with no last position gets spatial weight 1.0."""
          resolution = DirichletCategoricalResolution()
          obj = make_object_with_position(None, None, tick=0)
          # spatial_weight should be 1.0


  class TestComputeLikelihoods:
      def test_seen_features_higher_likelihood(self):
          """Features with high alpha get higher likelihood."""
          resolution = DirichletCategoricalResolution()
          # Object alpha: {"A": 10.0, "B": 10.0, "C": 1.0}
          # Observation with A and B should have higher likelihood than
          # observation with A and C

      def test_new_object_likelihood_uses_uniform(self):
          """New object model assigns uniform probability across vocab."""
          resolution = DirichletCategoricalResolution()
          resolution.prior_alpha = 1.0
          resolution._global_vocab = {"A", "B", "C", "D", "E"}
          # New object likelihood for features ["A", "B"]:
          # P(A) = 1.0/5.0 = 0.2, P(B) = 1.0/5.0 = 0.2
          # log_likelihood = log(0.2) + log(0.2)
  ```

**Implementation**:

- `roc/object.py` -- add `DirichletCategoricalResolution` class:

  ```python
  class DirichletCategoricalResolution(ObjectResolutionExpMod):
      """Bayesian object resolution using Dirichlet-Categorical model.

      Computes posterior probabilities over object identities using
      spatial/temporal priors and Dirichlet posterior predictive likelihoods.
      """

      name = "dirichlet-categorical"

      # Configurable parameters
      prior_alpha: float = 1.0
      spatial_scale: float = 3.0
      temporal_scale: float = 50.0
      confidence_threshold: float = 0.5
      excluded_feature_labels: set[str] = set()

      # Internal state
      _alphas: dict[NodeId, dict[str, float]]  # object_id -> {feature_str: alpha}
      _global_vocab: set[str]  # all feature strings ever seen

      # Telemetry
      posterior_max_histogram = Observability.meter.create_histogram(...)
      posterior_margin_histogram = Observability.meter.create_histogram(...)
      new_object_posterior_histogram = Observability.meter.create_histogram(...)
      alpha_sum_histogram = Observability.meter.create_histogram(...)
      decision_counter = Observability.meter.create_counter(...)

      def __init__(self):
          super().__init__()
          self._alphas = {}
          self._global_vocab = set()

      def resolve(self, feature_nodes, feature_group, context):
          """Full Bayesian resolution pipeline."""
          # Step 1: Find candidates (same graph walk)
          candidates = self._find_candidates(feature_nodes)
          if not candidates:
              return None

          # Filter features by exclusion set
          active_features = self._filter_features(feature_nodes)
          feature_strs = [str(f) for f in active_features]

          # Update global vocabulary
          self._global_vocab.update(feature_strs)

          # Step 2: Compute priors
          log_priors = self._compute_priors(candidates, context)

          # Step 3: Compute likelihoods
          log_likelihoods = self._compute_likelihoods(
              candidates, feature_strs
          )

          # Step 4: Compute posteriors
          log_posteriors = self._compute_posteriors(
              log_priors, log_likelihoods
          )

          # Step 5: Decision
          result = self._decide(candidates, log_posteriors)

          # Step 6: Update alphas if matched
          if result is not None:
              self._update_alphas(result.id, feature_strs)

          return result

      def _find_candidates(self, feature_nodes):
          """Graph walk: FeatureNode -> FeatureGroup -> Object."""
          # Same logic as SymmetricDifferenceResolution._find_candidates
          # but returns list[Object] without distance computation

      def _filter_features(self, feature_nodes):
          """Remove features whose labels intersect excluded_feature_labels."""

      def _compute_priors(self, candidates, context):
          """Spatial and temporal exponential decay priors.

          Returns dict mapping object_id -> log_prior, plus "new" -> log_prior.
          """

      def _compute_likelihoods(self, candidates, feature_strs):
          """Dirichlet posterior predictive likelihoods in log-space.

          For each candidate:
            log P(obs | obj) = sum(log(alpha_j / sum_alpha)) for each feature_j

          For "new object":
            log P(obs | new) = sum(log(prior_alpha / (N * prior_alpha)))
                             = K * log(1/N)
          """

      def _compute_posteriors(self, log_priors, log_likelihoods):
          """Bayes rule: log_posterior = log_prior + log_likelihood - log_Z."""
          # Use scipy.special.logsumexp for normalization

      def _decide(self, candidates, log_posteriors):
          """MAP decision with confidence threshold."""

      def _update_alphas(self, object_id, feature_strs):
          """Increment alpha counts for matched object."""

      def initialize_alphas(self, object_id, feature_strs):
          """Initialize alpha vector for a newly created object."""
  ```

- Key implementation details:
  - All probability computation in log-space to avoid underflow
  - `scipy.special.logsumexp` for normalization (already a dependency)
  - Alpha vectors stored as `dict[NodeId, dict[str, float]]` (side dict, not
    in graph DB)
  - Global vocabulary tracks all feature strings ever seen (for new-object
    likelihood)
  - Feature exclusion via label set intersection

**Dependencies**:
- External: `scipy.special.logsumexp` (already a project dependency)
- Internal: Phase 1 (ResolutionContext, Object.last_x/last_y/last_tick)

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_dirichlet_resolution.py -v`
2. Expected: All 20+ test cases pass
3. Run: `make test`
4. Expected: Full suite passes -- the new ExpMod is registered but not default
5. Run: `make lint`
6. Expected: No type errors (mypy) or lint issues (ruff)
7. Manual spot check: In a Python REPL, verify the math:
   ```python
   from roc.object import DirichletCategoricalResolution
   r = DirichletCategoricalResolution()
   # Verify r.name == "dirichlet-categorical"
   # Verify ExpMod registry contains it
   ```

**Success Criteria**:
- `DirichletCategoricalResolution` is registered as an ExpMod
- All 8 design verification scenarios pass as unit tests
- Numerical stability tests pass (no NaN/Inf)
- Feature exclusion works
- Alpha update works correctly
- Existing tests still pass (no regressions)
- Mypy and ruff pass

---

### Phase 4: Integration, Game Runs, and Tuning

**What this phase accomplishes**: Validate the Dirichlet-Categorical model
against a real game, compare to the Phase 2 baseline, tune parameters if
needed, and optionally make it the default.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/integration/test_dirichlet_integration.py` -- integration tests that
  exercise the full pipeline without a live game:

  ```python
  """Integration tests for Dirichlet-Categorical resolution pipeline."""

  class TestDirichletIntegrationFlow:
      def test_full_resolution_cycle(self):
          """End-to-end: create objects, resolve matches, verify alphas grow."""
          # 1. Set up DirichletCategoricalResolution as active ExpMod
          # 2. Create a sequence of observations with known features
          # 3. First observation: should create new object
          # 4. Second observation (same features, same position): should match
          # 5. Third observation (different features): should create new
          # 6. Fourth observation (same as first): should match first object
          # Verify alpha vectors reflect observation counts

      def test_warmup_then_stable(self):
          """Simulate 50+ observations and verify match rate stabilizes."""
          # Create a fixed set of 5 "entities" with distinct feature profiles
          # Simulate 100 observations cycling through these entities
          # After warmup (~10 observations), match rate should be > 70%

      def test_object_count_does_not_explode(self):
          """Verify object count stays bounded for repeated observations."""
          # Same 3 entities observed 50 times each
          # Should produce approximately 3 objects, not 150

      def test_spatial_helps_disambiguate_identical_features(self):
          """Two entities with same features at different positions."""
          # Entity A always at (5, 5), Entity B always at (50, 50)
          # Same features for both
          # After warmup, observations at (5, 5) should match A, not B


  class TestDirichletTelemetry:
      def test_all_metrics_emitted(self):
          """Verify all Dirichlet-specific metrics are recorded."""
          # Run a few resolutions
          # Verify posterior_max, posterior_margin, new_object_posterior,
          # alpha_sum histograms all have recorded values

      def test_decision_counter_outcomes(self):
          """Verify decision counter tracks match, new_object, low_confidence."""
  ```

**Implementation**:

- `roc/object.py` -- connect alpha initialization for new objects:

  ```python
  class ObjectResolver(Component):
      def do_object_resolution(self, e):
          # ... existing resolution ...
          resolution = ObjectResolutionExpMod.get(default="symmetric-difference")
          o = resolution.resolve(fg.feature_nodes, fg, ctx)

          if o is None:
              o = Object.with_features(fg)
              # Initialize alphas for the new object in the ExpMod
              if hasattr(resolution, "initialize_alphas"):
                  feature_strs = [str(f) for f in fg.feature_nodes]
                  resolution.initialize_alphas(o.id, feature_strs)
  ```

- `tmp/run_baseline.py` -- script to collect baseline metrics:

  ```python
  """Run a game session and collect resolution metrics for comparison.

  Usage:
      uv run python tmp/run_baseline.py --ticks 200 --expmod symmetric-difference
      uv run python tmp/run_baseline.py --ticks 200 --expmod dirichlet-categorical
  """
  # Configure the specified ExpMod
  # Run agent for N ticks
  # Collect and print summary metrics:
  #   - Total objects created
  #   - Match rate (matched / total)
  #   - Mean/p95 posterior confidence (Dirichlet only)
  #   - Object creation rate over time
  #   - Candidate count distribution
  ```

- `tmp/compare_runs.py` -- script to compare two run outputs:

  ```python
  """Compare baseline and Dirichlet-Categorical run metrics.

  Checks acceptance criteria:
    - Match rate after warmup >= 70%
    - Object growth rate < 1 per 10 ticks after warmup
    - Median posterior confidence > 0.6
    - Object count within 2x of baseline
    - Zero NaN/Inf in posteriors
    - Resolution time < 2x baseline
  """
  ```

**Dependencies**:
- External: None new
- Internal: Phase 1, Phase 2 (baseline data), Phase 3 (the algorithm)

**Verification**:
1. Run integration tests:
   `uv run pytest -c pyproject.toml tests/integration/test_dirichlet_integration.py -v`
2. Expected: All integration tests pass
3. Run baseline collection:
   `uv run python tmp/run_baseline.py --ticks 200 --expmod symmetric-difference`
4. Run Dirichlet collection:
   `uv run python tmp/run_baseline.py --ticks 200 --expmod dirichlet-categorical`
5. Compare:
   `uv run python tmp/compare_runs.py`
6. Expected output shows acceptance criteria status:
   ```
   Match rate (tick>50):    82% >= 70% PASS
   Object growth rate:      0.3/10 ticks < 1/10 PASS
   Median posterior:        0.74 > 0.6 PASS
   Object count vs baseline: 1.3x < 2x PASS
   NaN/Inf occurrences:     0 PASS
   Resolution time ratio:   1.2x < 2x PASS
   ```
7. If criteria pass, update the default ExpMod:
   Change `ObjectResolutionExpMod.get(default="symmetric-difference")` to
   `ObjectResolutionExpMod.get(default="dirichlet-categorical")`
8. Run full test suite: `make test` -- verify no regressions

**Success Criteria**:
- Integration tests demonstrate correct multi-observation behavior
- Game run metrics meet all six acceptance criteria
- Performance is within 2x of baseline
- No NaN/Inf in any computation
- Decision to make default is data-driven, documented with run metrics

---

## Common Utilities Needed

- **`make_feature_node()` test helper**: Creates mock FeatureNode with labels
  and string representation. Used across `test_object.py` and
  `test_dirichlet_resolution.py`. Define in a shared `conftest.py` or in each
  test file.

- **Manhattan distance helper**: `abs(x1 - x2) + abs(y1 - y2)`. Used in prior
  computation and telemetry. Small enough to inline rather than extract.

## External Libraries Assessment

- **scipy.special.logsumexp**: Already a dependency. Used for numerically
  stable log-space normalization. No new dependency needed.
- **numpy**: Already a dependency. Considered and rejected for alpha vectors --
  Python dicts are faster at the 3-5 element scale of feature sets.
- **No new dependencies required.** The Dirichlet-Categorical model is
  analytically tractable and implemented with stdlib `math` + `logsumexp`.

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| **Cold-start errors compound** (Risk 4 from design) | Phase 3 tests explicitly cover cold-start. Phase 4 integration test verifies warmup behavior. Monitor new-object rate in early ticks. |
| **Numerical instability** (NaN/Inf) | All computation in log-space. Dedicated numerical stability tests in Phase 3. logsumexp handles edge cases. |
| **Performance regression** | Phase 4 measures per-frame time. The graph walk (unchanged) dominates; arithmetic overhead is O(candidates * features). |
| **Parameter sensitivity** (Risk 2) | Defaults from Bayesian literature (prior_alpha=1.0 Laplace, spatial_scale=3.0, temporal_scale=50.0). Phase 4 tuning guided by telemetry. |
| **Feature string fragility** (Risk 6) | Alpha keys use `str(f)` matching existing pattern. Document this coupling. Consider hash-based keys as future improvement. |
| **Breaking existing tests** | Each phase runs full test suite. New ExpMod is not default until Phase 4 validation passes. SymmetricDifferenceResolution remains available. |
| **Telemetry overhead** | Histograms and counters are lightweight. Phase 2 verifies no performance impact before algorithm changes. |
