"""Saliency attenuation ExpMod for inhibition of return.

Attenuates the strength image inside SaliencyMap.get_focus() before
peak-finding. The base class defines the interface; concrete implementations
apply different attenuation strategies to bias attention away from recently
attended locations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

from .expmod import ExpMod
from .reporting.metrics import RocMetrics
from .reporting.observability import Observability

if TYPE_CHECKING:
    from strictly_typed_pandas import DataSet

    from .attention import SaliencyMap, VisionAttentionSchema

_otel_logger = Observability.get_logger("roc.saliency_attenuation")

peak_count_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.peak_count",
    description="Number of peaks found after attenuation",
)

top_peak_strength_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.top_peak_strength",
    description="Strength of the top peak after attenuation",
)

top_peak_shifted_counter = Observability.meter.create_counter(
    "roc.saliency_attenuation.top_peak_shifted",
    description="Count of times the top peak shifted due to attenuation",
)


class SaliencyAttenuationExpMod(ExpMod):
    """Base class for saliency attenuation strategies."""

    modtype = "saliency-attenuation"

    @Observability.tracer.start_as_current_span("saliency_attenuation")
    def attenuate(
        self,
        strength_image: npt.NDArray[np.floating[Any]],
        saliency_map: SaliencyMap,
    ) -> npt.NDArray[np.floating[Any]]:
        """Attenuate the strength image before peak-finding.

        Args:
            strength_image: 2D numpy array, shape (width, height), values [0, 1].
            saliency_map: The SaliencyMap instance for grid metadata.

        Returns:
            Attenuated strength image of the same shape, values in [0, 1].
        """
        raise NotImplementedError

    def notify_focus(
        self,
        focus_points: DataSet[VisionAttentionSchema],
    ) -> None:
        """Called after peak-finding with the resulting focus points.

        Override in stateful flavors to record the attended location.
        """


class NoAttenuation(SaliencyAttenuationExpMod):
    """Passthrough: returns the strength image unchanged."""

    name = "none"

    def attenuate(
        self,
        strength_image: npt.NDArray[np.floating[Any]],
        saliency_map: SaliencyMap,
    ) -> npt.NDArray[np.floating[Any]]:
        """Return strength image unchanged."""
        return strength_image


# -- Linear-decline specific metrics --

max_penalty_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.max_penalty",
    description="Maximum penalty applied to strength image",
)

history_size_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.history_size",
    description="Number of entries in the linear-decline history buffer",
)


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

    Configuration is read from Config fields with prefix ``saliency_attenuation_``:
        - ``saliency_attenuation_capacity``: Max history entries (default 5)
        - ``saliency_attenuation_radius``: Manhattan distance radius (default 3)
        - ``saliency_attenuation_max_penalty``: Peak penalty multiplier (default 1.0)
        - ``saliency_attenuation_max_attenuation``: Floor on attenuation (default 0.9)
    """

    name = "linear-decline"

    capacity: int = 5
    max_penalty: float = 1.0
    radius: int = 3
    max_attenuation: float = 0.9

    def __init__(self) -> None:
        super().__init__()
        self._load_config()
        self._history: deque[AttendedLocation] = deque(maxlen=self.capacity)

    def _load_config(self) -> None:
        """Load parameters from Config if available."""
        from .config import Config

        try:
            settings = Config.get()
            self.capacity = settings.saliency_attenuation_capacity
            self.radius = settings.saliency_attenuation_radius
            self.max_penalty = settings.saliency_attenuation_max_penalty
            self.max_attenuation = settings.saliency_attenuation_max_attenuation
        except Exception:
            # Config may not be initialized yet (e.g. during import-time registration)
            pass

    def _make_history(self) -> deque[AttendedLocation]:
        """Create a new history deque with the current capacity."""
        return deque(maxlen=self.capacity)

    @Observability.tracer.start_as_current_span("saliency_attenuation_linear_decline")
    def attenuate(
        self,
        strength_image: npt.NDArray[np.floating[Any]],
        saliency_map: SaliencyMap,
    ) -> npt.NDArray[np.floating[Any]]:
        """Attenuate strength image based on recency-weighted spatial penalty."""
        if len(self._history) == 0:
            return strength_image

        result = np.copy(strength_image)
        width, height = result.shape

        overall_max_penalty = 0.0
        attenuated_count = 0

        for x in range(width):
            for y in range(height):
                if result[x, y] == 0:
                    continue
                penalty = self._compute_penalty(x, y)
                if penalty > 0:
                    overall_max_penalty = max(overall_max_penalty, penalty)
                    attenuated_count += 1
                    multiplier = max(1.0 - self.max_attenuation, 1.0 - penalty)
                    result[x, y] *= multiplier

        # Record telemetry (dual OTel + W&B)
        RocMetrics.record_histogram(
            "roc.saliency_attenuation.max_penalty",
            overall_max_penalty,
            description="Maximum penalty applied to strength image",
        )
        RocMetrics.record_histogram(
            "roc.saliency_attenuation.history_size",
            float(len(self._history)),
            description="Number of entries in the linear-decline history buffer",
        )

        logger.debug(
            "attenuation: {} history entries, max_penalty={:.3f}, {} cells attenuated",
            len(self._history),
            overall_max_penalty,
            attenuated_count,
        )

        return result

    def notify_focus(
        self,
        focus_points: DataSet[VisionAttentionSchema],
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


# -- Active-inference specific metrics --

entropy_at_focus_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.entropy_at_focus",
    description="Entropy at the focused location after observation",
)

entropy_range_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.entropy_range",
    description="Range (max - min) of entropy across tracked beliefs",
)

omega_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.omega",
    description="Current volatility (omega) estimate",
)

vocab_size_counter = Observability.meter.create_counter(
    "roc.saliency_attenuation.vocab_size",
    description="Number of unique states in the vocabulary",
)

beliefs_tracked_histogram = Observability.meter.create_histogram(
    "roc.saliency_attenuation.beliefs_tracked",
    description="Number of spatial locations with active beliefs",
)


class StateVocabulary:
    """Maps feature-set hashes to discrete state IDs.

    Encodes arbitrary feature sets (lists of strings) into integer state IDs
    in [0, max_states). Order of features does not affect the ID. When more
    than max_states unique feature sets are seen, IDs wrap around.
    """

    def __init__(self, max_states: int = 64) -> None:
        self._hash_to_id: dict[int, int] = {}
        self._next_id: int = 0
        self._max_states: int = max_states

    @property
    def size(self) -> int:
        """Number of unique feature sets encoded so far."""
        return len(self._hash_to_id)

    def encode(self, features: list[str]) -> int:
        """Encode a feature set to a state ID.

        Args:
            features: List of feature strings. Order does not matter.

        Returns:
            Integer state ID in [0, max_states).
        """
        h = hash(frozenset(features))
        if h not in self._hash_to_id:
            self._hash_to_id[h] = self._next_id % self._max_states
            self._next_id += 1
        return self._hash_to_id[h]


@dataclass
class LocationBelief:
    """Beliefs about a single spatial location.

    Maintains a categorical distribution over hidden states (q_s) and
    supports Bayesian observation updates and volatility-driven uncertainty
    propagation.
    """

    q_s: npt.NDArray[np.floating[Any]]
    last_observed_tick: int
    last_observation: int
    zeta: float
    zeta_alpha: float
    zeta_beta: float

    @classmethod
    def uniform(
        cls,
        n_states: int,
        zeta: float = 1.0,
        zeta_alpha: float = 2.0,
        zeta_beta: float = 1.0,
    ) -> LocationBelief:
        """Create a belief with uniform distribution.

        Args:
            n_states: Number of hidden states.
            zeta: Initial sensory precision.
            zeta_alpha: Gamma shape parameter for zeta prior.
            zeta_beta: Gamma rate parameter for zeta prior.

        Returns:
            A LocationBelief with uniform q_s.
        """
        return cls(
            q_s=np.ones(n_states) / n_states,
            last_observed_tick=0,
            last_observation=-1,
            zeta=zeta,
            zeta_alpha=zeta_alpha,
            zeta_beta=zeta_beta,
        )

    def entropy(self) -> float:
        """Shannon entropy of the state belief.

        Returns:
            H = -sum(q * ln(q)) for q > 0. Always non-negative.
        """
        # Filter out zero probabilities to avoid log(0)
        nonzero = self.q_s[self.q_s > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    def observe(self, state_id: int, zeta: float) -> None:
        """Update belief after observing a state at this location.

        Applies precision-weighted Bayesian update:
        q_s ~ softmax(zeta * log(A[o, :]) + log(q_s_prior))
        For identity A, this simplifies to boosting q_s[state_id] by zeta.

        Args:
            state_id: The observed state ID.
            zeta: Sensory precision weighting the observation.
        """
        log_q = np.log(np.maximum(self.q_s, 1e-300))
        log_q[state_id] += zeta
        # Numerically stable softmax
        log_q -= np.max(log_q)
        q_new = np.exp(log_q)
        q_new /= np.sum(q_new)
        self.q_s = q_new
        self.last_observation = state_id

    def propagate(self, omega: float, ticks_elapsed: int) -> None:
        """Increase uncertainty for unobserved location via volatility.

        Blends q_s toward uniform at a rate determined by omega and elapsed
        ticks: rate = 1 - exp(-omega * ticks_elapsed).

        Args:
            omega: Volatility parameter controlling decay rate.
            ticks_elapsed: Number of ticks since last observation.
        """
        n = len(self.q_s)
        uniform = np.ones(n) / n
        rate = 1.0 - math.exp(-omega * ticks_elapsed)
        self.q_s = (1.0 - rate) * self.q_s + rate * uniform


class ActiveInferenceAttenuation(SaliencyAttenuationExpMod):
    """Discrete-state active inference agent for saliency attenuation.

    Attenuates the strength image based on epistemic value at each cell.
    Recently observed locations have low entropy (known state) and are
    attenuated. Unobserved locations gain entropy over time via
    volatility-weighted transition propagation, reducing attenuation.

    Configuration is read from Config fields with prefix ``saliency_attenuation_ai_``:
        - ``saliency_attenuation_ai_max_states``: State vocabulary size (default 64)
        - ``saliency_attenuation_ai_max_locations``: Max tracked beliefs (default 32)
        - ``saliency_attenuation_ai_max_attenuation``: Floor on attenuation (default 0.9)
        - ``saliency_attenuation_ai_saliency_weight``: Blend weight for raw saliency (default 0.5)
        - ``saliency_attenuation_ai_omega_alpha_prior``: Gamma shape for omega (default 2.0)
        - ``saliency_attenuation_ai_omega_beta_prior``: Gamma rate for omega (default 1.0)
        - ``saliency_attenuation_ai_zeta_alpha_prior``: Gamma shape for zeta (default 2.0)
        - ``saliency_attenuation_ai_zeta_beta_prior``: Gamma rate for zeta (default 1.0)
        - ``saliency_attenuation_ai_b_self_transition``: Self-transition prob (default 0.9)
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
        self._load_config()
        self._vocab = StateVocabulary(max_states=self.max_states)
        self._beliefs: dict[tuple[int, int], LocationBelief] = {}
        self._omega: float = self.omega_alpha_prior / self.omega_beta_prior
        self._omega_alpha: float = self.omega_alpha_prior
        self._omega_beta: float = self.omega_beta_prior
        self._last_tick: int = 0
        self._last_saliency_map: SaliencyMap | None = None

    def _load_config(self) -> None:
        """Load parameters from Config if available."""
        from .config import Config

        try:
            settings = Config.get()
            self.max_states = settings.saliency_attenuation_ai_max_states
            self.max_locations = settings.saliency_attenuation_ai_max_locations
            self.max_attenuation = settings.saliency_attenuation_ai_max_attenuation
            self.saliency_weight = settings.saliency_attenuation_ai_saliency_weight
            self.omega_alpha_prior = settings.saliency_attenuation_ai_omega_alpha_prior
            self.omega_beta_prior = settings.saliency_attenuation_ai_omega_beta_prior
            self.zeta_alpha_prior = settings.saliency_attenuation_ai_zeta_alpha_prior
            self.zeta_beta_prior = settings.saliency_attenuation_ai_zeta_beta_prior
            self.B_self_transition = settings.saliency_attenuation_ai_b_self_transition
        except Exception:
            pass

    @Observability.tracer.start_as_current_span("saliency_attenuation_active_inference")
    def attenuate(
        self,
        strength_image: npt.NDArray[np.floating[Any]],
        saliency_map: SaliencyMap,
    ) -> npt.NDArray[np.floating[Any]]:
        """Attenuate strength image based on epistemic value at each cell."""
        self._last_saliency_map = saliency_map

        if len(self._beliefs) == 0:
            return strength_image

        max_entropy = np.log(self.max_states)
        if max_entropy == 0:
            return strength_image

        result = np.copy(strength_image)
        width, height = result.shape

        entropies: list[float] = []

        for (bx, by), belief in self._beliefs.items():
            if 0 <= bx < width and 0 <= by < height:
                ent = belief.entropy()
                entropies.append(ent)
                normalized_entropy = ent / max_entropy
                # epistemic_mult: 0 when entropy=0 (known), 1 when entropy=max (unknown)
                epistemic_mult = (
                    1 - self.max_attenuation
                ) + self.max_attenuation * normalized_entropy
                # Blend epistemic attenuation with raw saliency passthrough
                effective_mult = (1 - self.saliency_weight) * epistemic_mult + self.saliency_weight
                result[bx, by] *= effective_mult

        # Record telemetry (dual OTel + W&B)
        RocMetrics.record_histogram(
            "roc.saliency_attenuation.beliefs_tracked",
            float(len(self._beliefs)),
            description="Number of spatial locations with active beliefs",
        )
        RocMetrics.record_histogram(
            "roc.saliency_attenuation.omega",
            self._omega,
            description="Current volatility (omega) estimate",
        )
        if entropies:
            RocMetrics.record_histogram(
                "roc.saliency_attenuation.entropy_range",
                max(entropies) - min(entropies),
                description="Range (max - min) of entropy across tracked beliefs",
            )

        logger.debug(
            "active-inference attenuation: {} beliefs, omega={:.4f}, vocab={}",
            len(self._beliefs),
            self._omega,
            self._vocab.size,
        )

        return result

    def notify_focus(
        self,
        focus_points: DataSet[VisionAttentionSchema],
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

        # Record telemetry (dual OTel + W&B)
        if (sx, sy) in self._beliefs:
            RocMetrics.record_histogram(
                "roc.saliency_attenuation.entropy_at_focus",
                self._beliefs[(sx, sy)].entropy(),
                description="Entropy at the focused location after observation",
            )
        RocMetrics.increment_counter(
            "roc.saliency_attenuation.vocab_size",
            1,
            description="Number of unique states in the vocabulary",
        )

    def _update_belief(
        self,
        x: int,
        y: int,
        state_id: int,
        tick: int,
    ) -> None:
        """Update belief at (x, y) after observation, update precision."""
        key = (x, y)
        if key not in self._beliefs:
            # Evict oldest if at capacity
            if len(self._beliefs) >= self.max_locations:
                self._evict_lru()
            self._beliefs[key] = LocationBelief.uniform(
                n_states=self.max_states,
                zeta=self.zeta_alpha_prior / self.zeta_beta_prior,
                zeta_alpha=self.zeta_alpha_prior,
                zeta_beta=self.zeta_beta_prior,
            )

        belief = self._beliefs[key]

        # Compute prediction error before updating
        prediction_error = 1.0 - belief.q_s[state_id]

        # Update belief with observation
        belief.observe(state_id=state_id, zeta=belief.zeta)
        belief.last_observed_tick = tick

        # Update global volatility based on prediction error
        self._update_volatility(prediction_error)

        # Update precision based on prediction error
        self._update_precision(belief, prediction_error)

    def _update_precision(self, belief: LocationBelief, prediction_error: float) -> None:
        """Update sensory precision (zeta) based on prediction error.

        Low prediction error -> increase precision (observations are reliable).
        High prediction error -> decrease precision (observations are noisy).
        """
        # Bayesian update of Gamma posterior for zeta
        belief.zeta_alpha += 0.5
        belief.zeta_beta += 0.5 * prediction_error**2
        belief.zeta = belief.zeta_alpha / belief.zeta_beta

    def _update_volatility(self, prediction_error: float) -> None:
        """Update global omega estimate based on state prediction error.

        High prediction error -> increase omega (environment is volatile).
        Low prediction error -> decrease omega (environment is stable).
        """
        # Bayesian update of Gamma posterior for omega
        self._omega_alpha += 0.5
        self._omega_beta += 0.5 * (1.0 - prediction_error) ** 2
        self._omega = self._omega_alpha / self._omega_beta

    def _evict_lru(self) -> None:
        """Evict the least-recently-observed belief."""
        if not self._beliefs:
            return
        oldest_key = min(self._beliefs, key=lambda k: self._beliefs[k].last_observed_tick)
        del self._beliefs[oldest_key]
