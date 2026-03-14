# Implementation Plan for Saliency Attenuation (Inhibition of Return)

## Overview

Add a `saliency-attenuation` ExpMod that attenuates the strength image inside
`SaliencyMap.get_focus()` before peak-finding, implementing inhibition of return
(IOR). Three flavors: `none` (current behavior, default), `linear-decline`
(recency-weighted spatial attenuation per Snyder & Kingstone 2000), and
`active-inference` (discrete-state epistemic agent per Parr & Friston 2017).

The core integration is a small addition to `SaliencyMap.get_focus()` in
`attention.py` that calls the ExpMod to attenuate the strength image before
dilation-based peak detection. `ObjectResolver` and the rest of the pipeline
are completely unchanged.

## Phase Breakdown

### Phase 1: Scaffold, `none` Flavor, and Integration Point

**Objective**: Create the ExpMod base class, the `none` passthrough
implementation, wire it into `SaliencyMap.get_focus()`, and verify zero behavioral change.

**Duration**: 1 day

**Tests to Write First**:

- `tests/unit/test_saliency_attenuation.py` -- new file:

  ```python
  """Unit tests for SaliencyAttenuationExpMod."""

  import numpy as np
  import pytest
  from unittest.mock import MagicMock

  from roc.attention import SaliencyMap
  from roc.expmod import expmod_registry
  from roc.saliency_attenuation import (
      NoAttenuation,
      SaliencyAttenuationExpMod,
  )


  def make_strength_image(width: int = 10, height: int = 5) -> np.ndarray:
      """Create a test strength image with known peak structure."""
      img = np.zeros((width, height))
      img[3, 2] = 0.9  # primary peak
      img[7, 4] = 0.7  # secondary peak
      return img


  class TestExpModRegistration:
      def test_modtype_registered(self):
          """saliency-attenuation modtype exists in registry."""
          assert "saliency-attenuation" in expmod_registry

      def test_none_registered(self):
          """'none' flavor is registered."""
          assert "none" in expmod_registry["saliency-attenuation"]

      def test_get_default_returns_none_flavor(self):
          """get(default='none') returns NoAttenuation instance."""
          result = SaliencyAttenuationExpMod.get(default="none")
          assert isinstance(result, NoAttenuation)


  class TestNoAttenuation:
      def test_returns_image_unchanged(self):
          """Returns the strength image with identical values."""
          img = make_strength_image()
          sm = MagicMock(spec=SaliencyMap)
          result = NoAttenuation().attenuate(img, sm)
          np.testing.assert_array_equal(result, img)

      def test_different_image_sizes(self):
          """Works with various image dimensions."""
          sm = MagicMock(spec=SaliencyMap)
          for shape in [(5, 3), (80, 21), (1, 1)]:
              img = np.random.rand(*shape)
              result = NoAttenuation().attenuate(img, sm)
              np.testing.assert_array_equal(result, img)

      def test_all_zero_image(self):
          """Works with all-zero strength image."""
          img = np.zeros((10, 5))
          sm = MagicMock(spec=SaliencyMap)
          result = NoAttenuation().attenuate(img, sm)
          np.testing.assert_array_equal(result, img)
  ```

**Implementation**:

- `roc/saliency_attenuation.py` -- new file:

  ```python
  """Saliency attenuation ExpMod for inhibition of return.

  Attenuates the strength image inside SaliencyMap.get_focus() before
  peak-finding. The base class defines the interface; concrete implementations
  apply different attenuation strategies to bias attention away from recently
  attended locations.
  """

  from __future__ import annotations

  from typing import TYPE_CHECKING

  import numpy as np

  from .expmod import ExpMod

  if TYPE_CHECKING:
      from .attention import SaliencyMap, VisionAttentionSchema
      from strictly_typed_pandas import DataSet


  class SaliencyAttenuationExpMod(ExpMod):
      """Base class for saliency attenuation strategies."""

      modtype = "saliency-attenuation"

      def attenuate(
          self, strength_image: np.ndarray, saliency_map: SaliencyMap,
      ) -> np.ndarray:
          """Attenuate the strength image before peak-finding.

          Args:
              strength_image: 2D numpy array, shape (width, height), values [0, 1].
              saliency_map: The SaliencyMap instance for grid metadata.

          Returns:
              Attenuated strength image of the same shape, values in [0, 1].
          """
          raise NotImplementedError

      def notify_focus(
          self, focus_points: DataSet[VisionAttentionSchema],
      ) -> None:
          """Called after peak-finding with the resulting focus points.

          Override in stateful flavors to record the attended location.
          """
          pass


  class NoAttenuation(SaliencyAttenuationExpMod):
      """Passthrough: returns the strength image unchanged."""

      name = "none"

      def attenuate(
          self, strength_image: np.ndarray, saliency_map: SaliencyMap,
      ) -> np.ndarray:
          return strength_image
  ```

- `roc/attention.py` -- add attenuation call in `SaliencyMap.get_focus()`:

  ```python
  # After building fkimg (line ~213), before peak-finding:
  from .saliency_attenuation import SaliencyAttenuationExpMod
  attenuation = SaliencyAttenuationExpMod.get(default="none")
  fkimg = attenuation.attenuate(fkimg, self)

  # After building the DataSet, before returning (line ~249):
  attenuation.notify_focus(ds)
  ```

- Observability scaffolding in `roc/saliency_attenuation.py`:
  - OTel logger: `_otel_logger = Observability.get_logger("roc.saliency_attenuation")`
  - Shared metrics (created on base class, recorded by `get_focus()` integration):
    - `roc.saliency_attenuation.peak_count` (histogram)
    - `roc.saliency_attenuation.top_peak_strength` (histogram)
    - `roc.saliency_attenuation.top_peak_shifted` (counter, attributes: shifted)
  - Tracing span: `@Observability.tracer.start_as_current_span("saliency_attenuation")` on `attenuate()`
  - Structured log record emitted per tick (see design doc Section 12.2)

**Dependencies**:
- External: None
- Internal: None

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_saliency_attenuation.py -v`
2. Expected: All tests pass
3. Run: `make test`
4. Expected: Full suite passes with no regressions (behavior is identical)
5. Run: `make lint`
6. Expected: No type errors or lint issues

**Success Criteria**:
- `SaliencyAttenuationExpMod` base class exists with `modtype = "saliency-attenuation"`
- `NoAttenuation` is registered as `"none"` in the ExpMod registry
- `SaliencyMap.get_focus()` calls the ExpMod for attenuation
- Shared OTel metrics, structured log, and tracing span are in place
- All existing tests pass unchanged (zero behavioral change)

---

### Phase 2: `linear-decline` Flavor

**Objective**: Implement recency-weighted spatial attenuation with a FIFO
history buffer. After attending a location, nearby cells in the strength image
are attenuated on subsequent ticks, with penalty declining linearly by recency
and spatially by Manhattan distance.

**Duration**: 2 days

**Tests to Write First**:

- `tests/unit/test_saliency_attenuation.py` -- extend with linear-decline tests:

  ```python
  from roc.saliency_attenuation import LinearDeclineAttenuation


  class TestLinearDeclineAttenuation:
      def test_empty_history_no_change(self):
          """With no prior history, strength image is unchanged."""
          att = LinearDeclineAttenuation()
          img = make_strength_image()
          sm = MagicMock(spec=SaliencyMap)
          result = att.attenuate(img, sm)
          np.testing.assert_array_equal(result, img)

      def test_attenuates_recently_attended_cell(self):
          """After notify_focus records (3,2), that cell is attenuated."""
          att = LinearDeclineAttenuation()
          sm = MagicMock(spec=SaliencyMap)
          img = make_strength_image()  # peak at (3,2)=0.9, (7,4)=0.7

          # Simulate first tick: notify that (3, 2) was attended
          focus_df = make_focus_points([(3, 2, 0.9, 1), (7, 4, 0.7, 2)])
          att.notify_focus(focus_df)

          # Second tick: (3, 2) should be attenuated
          result = att.attenuate(img, sm)
          assert result[3, 2] < img[3, 2]  # attenuated
          assert result[7, 4] == img[7, 4]  # far away, unchanged

      def test_recency_decline(self):
          """Most recent location gets strongest attenuation; older ones get less."""
          att = LinearDeclineAttenuation()
          att.capacity = 3

          # Attend three locations in sequence
          for x in [3, 5, 7]:
              fp = make_focus_points([(x, 2, 0.9, 1)])
              att.notify_focus(fp)

          # All three in history. (7,2) most recent -> strongest penalty
          assert len(att._history) == 3

      def test_spatial_falloff(self):
          """Cells far from history entries receive no attenuation."""
          att = LinearDeclineAttenuation()
          att.radius = 3
          sm = MagicMock(spec=SaliencyMap)

          # Attend (3, 2)
          fp = make_focus_points([(3, 2, 0.9, 1)])
          att.notify_focus(fp)

          # (7, 4) is Manhattan distance 6 from (3, 2) -- beyond radius 3
          img = make_strength_image()
          result = att.attenuate(img, sm)
          assert result[7, 4] == img[7, 4]  # no attenuation

      def test_capacity_limit(self):
          """History buffer does not exceed capacity (FIFO eviction)."""
          att = LinearDeclineAttenuation()
          att.capacity = 3

          for x in range(10):
              fp = make_focus_points([(x, 0, 0.9, 1)])
              att.notify_focus(fp)

          assert len(att._history) == 3

      def test_max_attenuation_prevents_zeroing(self):
          """Strength is never reduced below (1 - max_attenuation) * raw."""
          att = LinearDeclineAttenuation()
          att.max_attenuation = 0.5  # can reduce to at most 50%
          sm = MagicMock(spec=SaliencyMap)

          fp = make_focus_points([(3, 2, 0.9, 1)])
          att.notify_focus(fp)

          img = make_strength_image()
          result = att.attenuate(img, sm)
          # Even with full penalty, attenuated strength >= 50% of original
          assert result[3, 2] >= img[3, 2] * 0.5

      def test_adjacent_cell_partially_attenuated(self):
          """Cell 1 step away from history gets partial attenuation."""
          att = LinearDeclineAttenuation()
          att.radius = 3
          sm = MagicMock(spec=SaliencyMap)

          fp = make_focus_points([(5, 5, 0.9, 1)])
          att.notify_focus(fp)

          img = np.ones((10, 10)) * 0.8
          result = att.attenuate(img, sm)
          # (5, 5) gets full penalty, (5, 6) gets partial, (5, 8) gets none
          assert result[5, 5] < result[5, 6] < result[5, 8]

      def test_cumulative_penalty_from_multiple_history(self):
          """Multiple nearby history entries produce cumulative attenuation."""
          att = LinearDeclineAttenuation()
          att.capacity = 5
          att.radius = 5
          sm = MagicMock(spec=SaliencyMap)

          # Attend two nearby locations
          fp1 = make_focus_points([(5, 5, 0.9, 1)])
          att.notify_focus(fp1)
          fp2 = make_focus_points([(6, 5, 0.9, 1)])
          att.notify_focus(fp2)

          img = np.ones((10, 10)) * 0.8
          result = att.attenuate(img, sm)
          # (5, 5) is near both history entries -- more attenuation than (8, 5)
          assert result[5, 5] < result[8, 5]
  ```

- `tests/unit/test_saliency_attenuation.py` -- telemetry tests:

  ```python
  class TestLinearDeclineTelemetry:
      def test_penalty_histogram_recorded(self):
          """Telemetry records max penalty applied to strength image."""

      def test_attenuation_count_recorded(self):
          """Telemetry records number of cells attenuated."""
  ```

**Implementation**:

- `roc/saliency_attenuation.py` -- add `LinearDeclineAttenuation`:

  ```python
  from collections import deque
  from dataclasses import dataclass


  @dataclass
  class AttendedLocation:
      """A previously attended location with the tick it was attended."""
      x: int
      y: int
      tick: int


  class LinearDeclineAttenuation(SaliencyAttenuationExpMod):
      """Attenuates recently attended locations with linear recency decline.

      Maintains a FIFO buffer of the last N attended locations. Each entry
      creates a spatial attenuation field in the strength image, with penalty
      declining linearly by recency rank and spatially by Manhattan distance.
      """

      name = "linear-decline"

      capacity: int = 5
      max_penalty: float = 1.0
      radius: int = 3
      max_attenuation: float = 0.9

      def __init__(self) -> None:
          super().__init__()
          self._history: deque[AttendedLocation] = deque(maxlen=self.capacity)

      def attenuate(
          self, strength_image: np.ndarray, saliency_map: SaliencyMap,
      ) -> np.ndarray:
          if len(self._history) == 0:
              return strength_image

          result = np.copy(strength_image)
          width, height = result.shape

          for x in range(width):
              for y in range(height):
                  if result[x, y] == 0:
                      continue
                  penalty = self._compute_penalty(x, y)
                  multiplier = max(1.0 - self.max_attenuation, 1.0 - penalty)
                  result[x, y] *= multiplier

          return result

      def notify_focus(
          self, focus_points: DataSet[VisionAttentionSchema],
      ) -> None:
          """Record the top-ranked peak in history."""
          from .sequencer import tick as current_tick
          if len(focus_points) == 0:
              return
          top = focus_points.iloc[0]
          self._history.append(
              AttendedLocation(
                  x=int(top["x"]),
                  y=int(top["y"]),
                  tick=current_tick,
              )
          )

      def _compute_penalty(self, x: int, y: int) -> float:
          """Sum penalties from all history entries."""
          total = 0.0
          n = len(self._history)
          for i, loc in enumerate(reversed(list(self._history))):
              # i=0 is most recent
              recency_weight = (n - i) / n
              dist = abs(x - loc.x) + abs(y - loc.y)
              spatial_weight = max(0.0, 1.0 - dist / self.radius)
              total += self.max_penalty * recency_weight * spatial_weight
          return total
  ```

**Dependencies**:
- External: None
- Internal: Phase 1 (base class, integration point)

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_saliency_attenuation.py -v`
2. Expected: All tests pass (Phase 1 + Phase 2 tests)
3. Run: `make test`
4. Expected: Full suite passes
5. Run: `make lint`
6. Expected: No type errors or lint issues
7. Manual game run:
   ```bash
   roc_expmods_use='[("action", "weighted"), ("saliency-attenuation", "linear-decline")]' \
   roc_debug_log=true uv run play
   ```
8. Expected: Debug log shows different focus points being selected across ticks
   rather than the same highest-saliency point every tick

- Add linear-decline-specific metrics:
  - `roc.saliency_attenuation.max_penalty` (histogram)
  - `roc.saliency_attenuation.history_size` (histogram)
- Add linear-decline fields to structured log record (`history`, `max_penalty_applied`)
- Add loguru debug output: `logger.debug("attenuation: {} peaks -> {}, top shifted: {}", ...)`

**Success Criteria**:
- `LinearDeclineAttenuation` is registered as `"linear-decline"`
- Strength image is attenuated near recently attended locations
- Attenuation declines linearly by recency rank
- Attenuation declines spatially by Manhattan distance
- History buffer respects capacity limit
- max_attenuation prevents total zeroing
- Flavor-specific metrics and structured log fields emit data
- All existing tests pass (default is still `"none"`)

---

### Phase 3: `active-inference` Core -- State Vocabulary and Belief Structure

**Objective**: Implement the state vocabulary, per-location belief structure,
and the uncertainty propagation mechanism. Unit-test the belief update math
independently of the ExpMod interface.

**Duration**: 2 days

**Tests to Write First**:

- `tests/unit/test_active_inference.py` -- new file:

  ```python
  """Unit tests for active inference saliency attenuation internals."""

  import math
  import numpy as np
  import pytest

  from roc.saliency_attenuation import (
      LocationBelief,
      StateVocabulary,
  )


  class TestStateVocabulary:
      def test_encode_new_features(self):
          """New feature set gets a unique state ID."""
          vocab = StateVocabulary(max_states=64)
          features = ["SingleNode(a)", "ColorNode(red)"]
          sid = vocab.encode(features)
          assert isinstance(sid, int)
          assert 0 <= sid < 64

      def test_encode_same_features_same_id(self):
          """Same feature set always maps to same state ID."""
          vocab = StateVocabulary(max_states=64)
          features = ["SingleNode(a)", "ColorNode(red)"]
          sid1 = vocab.encode(features)
          sid2 = vocab.encode(features)
          assert sid1 == sid2

      def test_encode_different_features_different_id(self):
          """Different feature sets get different state IDs."""
          vocab = StateVocabulary(max_states=64)
          sid1 = vocab.encode(["SingleNode(a)"])
          sid2 = vocab.encode(["SingleNode(b)"])
          assert sid1 != sid2

      def test_encode_order_independent(self):
          """Feature set order does not affect state ID."""
          vocab = StateVocabulary(max_states=64)
          sid1 = vocab.encode(["SingleNode(a)", "ColorNode(red)"])
          sid2 = vocab.encode(["ColorNode(red)", "SingleNode(a)"])
          assert sid1 == sid2

      def test_capacity_limit(self):
          """State IDs stay within [0, max_states)."""
          vocab = StateVocabulary(max_states=4)
          ids = set()
          for i in range(10):
              sid = vocab.encode([f"feature_{i}"])
              ids.add(sid)
          assert all(0 <= sid < 4 for sid in ids)

      def test_vocab_size(self):
          """size property tracks unique entries."""
          vocab = StateVocabulary(max_states=64)
          assert vocab.size == 0
          vocab.encode(["a"])
          assert vocab.size == 1
          vocab.encode(["b"])
          assert vocab.size == 2
          vocab.encode(["a"])  # duplicate
          assert vocab.size == 2


  class TestLocationBelief:
      def test_initial_belief_is_uniform(self):
          """New LocationBelief has uniform distribution over states."""
          belief = LocationBelief.uniform(n_states=8)
          assert belief.q_s.shape == (8,)
          assert np.allclose(belief.q_s, 1.0 / 8)

      def test_entropy_of_uniform(self):
          """Uniform distribution has maximum entropy."""
          belief = LocationBelief.uniform(n_states=8)
          expected = math.log(8)
          assert belief.entropy() == pytest.approx(expected, rel=1e-6)

      def test_entropy_of_peaked(self):
          """Near-deterministic distribution has near-zero entropy."""
          belief = LocationBelief.uniform(n_states=8)
          belief.q_s = np.zeros(8)
          belief.q_s[0] = 1.0
          assert belief.entropy() == pytest.approx(0.0, abs=1e-9)

      def test_observe_reduces_entropy(self):
          """Observing a state reduces entropy at that location."""
          belief = LocationBelief.uniform(n_states=8)
          initial_entropy = belief.entropy()
          belief.observe(state_id=3, zeta=2.0)
          assert belief.entropy() < initial_entropy

      def test_observe_concentrates_on_state(self):
          """After observation, belief is concentrated on observed state."""
          belief = LocationBelief.uniform(n_states=8)
          belief.observe(state_id=3, zeta=5.0)
          assert belief.q_s[3] > belief.q_s[0]
          assert np.argmax(belief.q_s) == 3

      def test_propagate_increases_entropy(self):
          """Uncertainty propagation increases entropy toward uniform."""
          belief = LocationBelief.uniform(n_states=8)
          belief.observe(state_id=3, zeta=5.0)
          peaked_entropy = belief.entropy()

          belief.propagate(omega=0.5, ticks_elapsed=5)
          assert belief.entropy() > peaked_entropy

      def test_propagate_rate_depends_on_omega(self):
          """Higher omega (more volatility) -> faster entropy increase."""
          b_low = LocationBelief.uniform(n_states=8)
          b_high = LocationBelief.uniform(n_states=8)
          # Same peaked state
          b_low.observe(state_id=3, zeta=5.0)
          b_high.observe(state_id=3, zeta=5.0)

          b_low.propagate(omega=0.1, ticks_elapsed=5)
          b_high.propagate(omega=1.0, ticks_elapsed=5)

          assert b_high.entropy() > b_low.entropy()

      def test_propagate_rate_depends_on_ticks(self):
          """More elapsed ticks -> more entropy increase."""
          b_short = LocationBelief.uniform(n_states=8)
          b_long = LocationBelief.uniform(n_states=8)
          b_short.observe(state_id=3, zeta=5.0)
          b_long.observe(state_id=3, zeta=5.0)

          b_short.propagate(omega=0.5, ticks_elapsed=1)
          b_long.propagate(omega=0.5, ticks_elapsed=10)

          assert b_long.entropy() > b_short.entropy()

      def test_propagate_bounded_by_uniform(self):
          """Propagation never exceeds uniform entropy."""
          belief = LocationBelief.uniform(n_states=8)
          belief.observe(state_id=3, zeta=5.0)
          belief.propagate(omega=10.0, ticks_elapsed=1000)

          max_entropy = math.log(8)
          assert belief.entropy() <= max_entropy + 1e-9
  ```

**Implementation**:

- `roc/saliency_attenuation.py` -- add belief structures:

  ```python
  class StateVocabulary:
      """Maps feature-set hashes to discrete state IDs."""

      def __init__(self, max_states: int = 64) -> None:
          self._hash_to_id: dict[int, int] = {}
          self._next_id: int = 0
          self._max_states: int = max_states

      @property
      def size(self) -> int:
          return len(self._hash_to_id)

      def encode(self, features: list[str]) -> int:
          h = hash(frozenset(features))
          if h not in self._hash_to_id:
              self._hash_to_id[h] = self._next_id % self._max_states
              self._next_id += 1
          return self._hash_to_id[h]


  @dataclass
  class LocationBelief:
      """Beliefs about a single spatial location."""

      q_s: np.ndarray           # categorical distribution over hidden states
      last_observed_tick: int
      last_observation: int     # state ID of last observation
      zeta: float               # sensory precision
      zeta_alpha: float         # Gamma shape for zeta
      zeta_beta: float          # Gamma rate for zeta

      @classmethod
      def uniform(cls, n_states: int, ...) -> LocationBelief:
          """Create a belief with uniform distribution."""
          ...

      def entropy(self) -> float:
          """Shannon entropy of the state belief."""
          # H = -sum(q * ln(q)) for q > 0
          ...

      def observe(self, state_id: int, zeta: float) -> None:
          """Update belief after observing a state at this location.

          Applies precision-weighted Bayesian update:
          q_s ~ softmax(zeta * log(A[o, :]) + log(q_s_prior))
          For identity A, this simplifies to boosting q_s[state_id] by zeta.
          Also updates zeta based on prediction error.
          """
          ...

      def propagate(self, omega: float, ticks_elapsed: int) -> None:
          """Increase uncertainty for unobserved location via volatility.

          q_s = (1 - rate) * q_s + rate * uniform
          rate = 1 - exp(-omega * ticks_elapsed)
          """
          ...
  ```

**Dependencies**:
- External: `numpy` (already a dependency)
- Internal: Phase 1

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_active_inference.py -v`
2. Expected: All belief and vocabulary tests pass
3. Run: `make lint`
4. Expected: No type errors or lint issues

**Success Criteria**:
- `StateVocabulary` correctly encodes feature sets to stable, order-independent IDs
- `LocationBelief` correctly computes entropy
- `observe()` concentrates belief and reduces entropy
- `propagate()` increases entropy toward uniform at a rate controlled by omega
- Entropy is bounded: never negative, never exceeds log(n_states)
- All numerical operations are stable (no NaN/Inf)

---

### Phase 4: `active-inference` Flavor -- Full ExpMod Integration

**Objective**: Wire `LocationBelief` and `StateVocabulary` into the
`ActiveInferenceAttenuation` ExpMod. Implement epistemic-value-based
attenuation of the strength image, precision/volatility updates, and the
`attenuate()` / `notify_focus()` interface. Integration test with synthetic
saliency maps.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_active_inference.py` -- extend with ExpMod tests:

  ```python
  from roc.saliency_attenuation import ActiveInferenceAttenuation


  class TestActiveInferenceAttenuation:
      def test_registered(self):
          """'active-inference' is registered in the ExpMod registry."""
          assert "active-inference" in expmod_registry["saliency-attenuation"]

      def test_no_prior_observations_no_change(self):
          """With no observation history, strength image is unchanged
          (all locations have uniform/maximum entropy)."""
          att = ActiveInferenceAttenuation()
          img = make_strength_image()
          sm = MagicMock(spec=SaliencyMap)
          result = att.attenuate(img, sm)
          np.testing.assert_array_equal(result, img)

      def test_ior_effect(self):
          """After observing location (3,2), that cell is attenuated."""
          att = ActiveInferenceAttenuation()
          att.saliency_weight = 0.3  # bias toward epistemic
          sm = MagicMock(spec=SaliencyMap)
          sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: f"f_{x}_{y}")]

          # Tick 1: notify that (3, 2) was attended
          fp = make_focus_points([(3, 2, 0.9, 1), (7, 4, 0.7, 2)])
          att._last_saliency_map = sm
          att.notify_focus(fp)

          # Tick 2: (3, 2) entropy is low -> attenuated
          img = make_strength_image()
          result = att.attenuate(img, sm)
          assert result[3, 2] < img[3, 2]

      def test_entropy_recovery_reduces_attenuation(self):
          """After enough ticks, previously attended location is less attenuated."""
          att = ActiveInferenceAttenuation()
          att.saliency_weight = 0.0  # pure epistemic
          sm = MagicMock(spec=SaliencyMap)
          sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "f_a")]

          # Observe (3, 2)
          fp = make_focus_points([(3, 2, 0.9, 1)])
          att._last_saliency_map = sm
          att.notify_focus(fp)

          # Simulate many ticks passing by propagating uncertainty
          for belief in att._beliefs.values():
              belief.propagate(omega=1.0, ticks_elapsed=100)

          # Now (3, 2) should have high entropy -> minimal attenuation
          img = make_strength_image()
          result = att.attenuate(img, sm)
          assert result[3, 2] > img[3, 2] * 0.8  # nearly no attenuation

      def test_saliency_weight_1_no_attenuation(self):
          """With saliency_weight=1.0, no epistemic attenuation occurs."""
          att = ActiveInferenceAttenuation()
          att.saliency_weight = 1.0
          sm = MagicMock(spec=SaliencyMap)
          sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: f"f_{x}")]

          # Observe (3, 2)
          fp = make_focus_points([(3, 2, 0.9, 1)])
          att._last_saliency_map = sm
          att.notify_focus(fp)

          # Even after observation, no attenuation with weight=1.0
          img = make_strength_image()
          result = att.attenuate(img, sm)
          np.testing.assert_array_equal(result, img)

      def test_high_volatility_short_ior(self):
          """With high omega, entropy recovers quickly -- less attenuation."""
          att_low = ActiveInferenceAttenuation()
          att_high = ActiveInferenceAttenuation()
          att_low.omega_alpha_prior = 1.0
          att_low.omega_beta_prior = 10.0   # low omega
          att_high.omega_alpha_prior = 10.0
          att_high.omega_beta_prior = 1.0   # high omega

          # Both observe same location, then propagate 5 ticks
          # High-omega agent should have less attenuation (higher entropy recovery)

      def test_max_locations_eviction(self):
          """Belief dict respects max_locations via LRU eviction."""
          att = ActiveInferenceAttenuation()
          att.max_locations = 3
          sm = MagicMock(spec=SaliencyMap)
          sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "f")]

          for i in range(10):
              fp = make_focus_points([(i * 10, 0, 0.9, 1)])
              att._last_saliency_map = sm
              att.notify_focus(fp)

          assert len(att._beliefs) <= 3
  ```

- `tests/integration/test_saliency_attenuation_integration.py` -- new file:

  ```python
  """Integration tests for saliency attenuation over multi-tick sequences."""


  class TestLinearDeclineIntegration:
      def test_five_tick_sequence(self):
          """Run 5 ticks through get_focus() with linear-decline.

          Expected: peaks rotate rather than persevering on the same location.
          """

      def test_returns_to_location_after_decay(self):
          """After capacity ticks, the oldest location is evicted and can
          re-emerge as a peak."""


  class TestActiveInferenceIntegration:
      def test_explores_all_peaks(self):
          """Over 10 ticks with 3 stable saliency peaks, all 3 emerge as
          top-ranked peak at least once."""

      def test_volatile_environment_faster_return(self):
          """When features change at a location, agent revisits sooner."""

      def test_stable_environment_longer_inhibition(self):
          """When features are stable, agent takes longer to revisit."""
  ```

**Implementation**:

- `roc/saliency_attenuation.py` -- add `ActiveInferenceAttenuation`:

  ```python
  class ActiveInferenceAttenuation(SaliencyAttenuationExpMod):
      """Discrete-state active inference agent for saliency attenuation.

      Attenuates the strength image based on epistemic value at each cell.
      Recently observed locations have low entropy (known state) and are
      attenuated. Unobserved locations gain entropy over time via
      volatility-weighted transition propagation, reducing attenuation.
      """

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
          self._omega: float = self.omega_alpha_prior / self.omega_beta_prior
          self._omega_alpha: float = self.omega_alpha_prior
          self._omega_beta: float = self.omega_beta_prior
          self._last_tick: int = 0
          self._last_saliency_map: SaliencyMap | None = None

      def attenuate(
          self, strength_image: np.ndarray, saliency_map: SaliencyMap,
      ) -> np.ndarray:
          from .sequencer import tick as current_tick

          self._last_saliency_map = saliency_map

          # 1. Propagate uncertainty at all tracked locations
          for belief in self._beliefs.values():
              if belief.last_observed_tick < current_tick:
                  belief.propagate(
                      omega=self._omega,
                      ticks_elapsed=current_tick - belief.last_observed_tick,
                  )

          # 2. Build attenuation mask from epistemic values
          max_entropy = np.log(self.max_states)
          result = np.copy(strength_image)

          for (bx, by), belief in self._beliefs.items():
              width, height = result.shape
              if 0 <= bx < width and 0 <= by < height:
                  normalized_entropy = belief.entropy() / max_entropy
                  epistemic_mult = (
                      (1 - self.max_attenuation)
                      + self.max_attenuation * normalized_entropy
                  )
                  effective_mult = (
                      (1 - self.saliency_weight) * epistemic_mult
                      + self.saliency_weight
                  )
                  result[bx, by] *= effective_mult

          self._last_tick = current_tick
          return result

      def notify_focus(
          self, focus_points: DataSet[VisionAttentionSchema],
      ) -> None:
          """Observe features at the top-ranked peak and update beliefs."""
          from .sequencer import tick as current_tick

          if len(focus_points) == 0 or self._last_saliency_map is None:
              return
          top = focus_points.iloc[0]
          sx, sy = int(top["x"]), int(top["y"])
          features = self._last_saliency_map.get_val(sx, sy)
          feature_strs = [str(f) for f in features]
          state_id = self._vocab.encode(feature_strs)
          self._update_belief(sx, sy, state_id, current_tick)

      def _update_belief(
          self, x: int, y: int, state_id: int, tick: int,
      ) -> None:
          """Update belief at (x, y) after observation, update precision."""
          ...

      def _update_volatility(self, prediction_error: float) -> None:
          """Update global omega estimate based on state prediction error."""
          ...
  ```

**Dependencies**:
- External: `numpy` (already a dependency)
- Internal: Phase 1 (base class), Phase 3 (belief structures)

**Verification**:
1. Run: `uv run pytest -c pyproject.toml tests/unit/test_active_inference.py tests/unit/test_saliency_attenuation.py -v`
2. Expected: All tests pass
3. Run: `uv run pytest -c pyproject.toml tests/integration/test_saliency_attenuation_integration.py -v`
4. Expected: Integration tests pass
5. Run: `make test`
6. Expected: Full suite passes
7. Run: `make lint`
8. Expected: No type errors or lint issues
9. Manual game run:
   ```bash
   roc_expmods_use='[("action", "weighted"), ("saliency-attenuation", "active-inference")]' \
   roc_debug_log=true uv run play
   ```
10. Expected: Debug log shows the agent exploring different focus points across
    ticks, with revisitation after entropy recovery. Use Remote Logger MCP to
    monitor live entropy values.

- Add active-inference-specific metrics:
  - `roc.saliency_attenuation.entropy_at_focus` (histogram)
  - `roc.saliency_attenuation.entropy_range` (histogram)
  - `roc.saliency_attenuation.omega` (histogram)
  - `roc.saliency_attenuation.vocab_size` (counter)
  - `roc.saliency_attenuation.beliefs_tracked` (histogram)
- Add active-inference fields to structured log record (`entropy_at_focus`, `entropy_max`, `entropy_min`, `omega`, `beliefs_tracked`, `vocab_size`)

**Success Criteria**:
- `ActiveInferenceAttenuation` is registered as `"active-inference"`
- Epistemic value drives attenuation of recently observed locations
- Entropy recovers over time, reducing attenuation (allowing revisitation)
- Volatility omega adapts: faster recovery in changing environments
- saliency_weight correctly blends epistemic attenuation with raw saliency
- Belief dict respects max_locations capacity
- Flavor-specific metrics and structured log fields emit data
- No NaN/Inf in any computation
- All existing tests pass (default is still `"none"`)

---

## Common Utilities Needed

- **`make_attention_data()` test helper**: Creates `VisionAttentionData` from
  a list of (x, y, strength, label) tuples with a mock saliency map. Defined
  in `tests/unit/test_saliency_attenuation.py` and shared across test classes.

- **`make_focus_points()` test helper**: Creates the typed DataFrame from
  tuples. Used by `make_attention_data()`.

- **Manhattan distance**: `abs(x1 - x2) + abs(y1 - y2)`. Used in
  `LinearDeclineAttenuation._compute_penalty()`. Small enough to inline.

## External Libraries Assessment

- **numpy**: Already a dependency. Used for belief arrays in active inference.
- **scipy**: Already a dependency. `scipy.special.logsumexp` available if
  needed for numerical stability in softmax computation.
- **No new dependencies required.** All math is standard Bayesian inference
  using categorical distributions and Gamma priors.

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| **Integration breaks existing behavior** | Phase 1 `none` flavor is a pure passthrough (returns strength image unchanged). Full test suite validates zero behavioral change before any attenuation is added. |
| **Linear-decline parameters need tuning** | Defaults based on literature (capacity=5 per Snyder & Kingstone, radius=3 for grid game). Telemetry records attenuation metrics. Parameters are class attributes, easy to adjust. |
| **Active inference too slow per tick** | Attenuation only modifies cells with tracked beliefs (max_locations=32), not full grid. Belief update is O(max_states) per location. Profile in Phase 4 game runs. |
| **Entropy never recovers (stuck IOR)** | Phase 3 tests verify propagation increases entropy. omega > 0 guarantees eventual recovery. Test with extreme parameters. |
| **Entropy recovers too fast (no IOR effect)** | Default B_self_transition=0.9 and moderate omega_prior create multi-tick IOR. Phase 4 integration tests verify IOR duration. |
| **State vocabulary grows unbounded** | max_states=64 caps vocabulary. Wraps around on overflow. Phase 3 tests verify capacity constraint. |
| **Saliency weight wrong for game** | Default saliency_weight=0.5 is balanced. Phase 4 game runs test different values. Can be tuned without code changes. |
| **Circular import between attention.py and saliency_attenuation.py** | `attention.py` imports `SaliencyAttenuationExpMod` inside `get_focus()` (lazy import). `saliency_attenuation.py` uses `TYPE_CHECKING` guard for `SaliencyMap` type hints. No runtime circular import. |

## Future Work

### Linear-decline: Optimize attenuate() to only iterate cells within radius

Currently `attenuate()` iterates every cell in the grid and calls
`_compute_penalty()` for each non-zero one. This is O(grid_size *
history_size) per tick. With radius=3 and capacity=5, we see 40-58 cells
attenuated per tick -- many of which receive negligible penalty from distant
history entries.

**Optimization**: Instead of iterating the full grid, iterate only cells
within the Manhattan-distance diamond of each history entry. For each history
entry at (hx, hy), iterate x in [hx-radius, hx+radius] and y such that
|x-hx| + |y-hy| <= radius. Accumulate penalties into a result array.
This reduces the inner loop from O(width * height) to O(capacity * radius^2).

### Per-ExpMod configuration (DONE)

ExpMods that need tunable parameters add individual fields to `Config` with
a descriptive prefix (e.g. `saliency_attenuation_radius`). The ExpMod reads
these in `__init__` via `Config.get()`, falling back to class-attribute
defaults if Config is not yet initialized. Parameters are discoverable,
type-checked, and settable via environment variables:

```bash
roc_saliency_attenuation_radius=5 roc_saliency_attenuation_capacity=10 uv run play
```

The pattern is documented in `roc/expmod.py`'s module docstring.
