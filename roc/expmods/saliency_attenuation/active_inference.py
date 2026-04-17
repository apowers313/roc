"""Active-inference saliency attenuation: epistemic value via belief tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

from roc.framework.expmod import ExpModConfig
from roc.pipeline.attention.saliency_attenuation import SaliencyAttenuationExpMod
from roc.reporting.metrics import RocMetrics
from roc.reporting.observability import Observability

if TYPE_CHECKING:
    from strictly_typed_pandas import DataSet

    from roc.pipeline.attention.attention import SaliencyMap, VisionAttentionSchema


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
        """Encode a feature set to a state ID in [0, max_states)."""
        h = hash(frozenset(features))
        if h not in self._hash_to_id:
            self._hash_to_id[h] = self._next_id % self._max_states
            self._next_id += 1
        return self._hash_to_id[h]


@dataclass
class LocationBelief:
    """Beliefs about a single spatial location.

    Maintains a categorical distribution over hidden states (``q_s``) and supports
    Bayesian observation updates and volatility-driven uncertainty propagation.
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
        """Create a belief with uniform distribution."""
        return cls(
            q_s=np.ones(n_states) / n_states,
            last_observed_tick=0,
            last_observation=-1,
            zeta=zeta,
            zeta_alpha=zeta_alpha,
            zeta_beta=zeta_beta,
        )

    def entropy(self) -> float:
        """Shannon entropy of the state belief."""
        nonzero = self.q_s[self.q_s > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    def observe(self, state_id: int, zeta: float) -> None:
        """Precision-weighted Bayesian update."""
        log_q = np.log(np.maximum(self.q_s, 1e-300))
        log_q[state_id] += zeta
        log_q -= np.max(log_q)
        q_new = np.exp(log_q)
        q_new /= np.sum(q_new)
        self.q_s = q_new
        self.last_observation = state_id

    def propagate(self, omega: float, ticks_elapsed: int) -> None:
        """Blend q_s toward uniform via volatility-weighted decay."""
        n = len(self.q_s)
        uniform = np.ones(n) / n
        rate = 1.0 - math.exp(-omega * ticks_elapsed)
        self.q_s = (1.0 - rate) * self.q_s + rate * uniform


class ActiveInferenceConfig(ExpModConfig):
    """Configuration for active-inference saliency attenuation."""

    max_states: int = 64
    max_locations: int = 32
    max_attenuation: float = 0.9
    saliency_weight: float = 0.5
    omega_alpha_prior: float = 2.0
    omega_beta_prior: float = 1.0
    zeta_alpha_prior: float = 2.0
    zeta_beta_prior: float = 1.0
    b_self_transition: float = 0.9


class ActiveInferenceAttenuation(SaliencyAttenuationExpMod):
    """Discrete-state active inference agent for saliency attenuation.

    Attenuates the strength image based on epistemic value at each cell. Recently
    observed locations have low entropy (known state) and are attenuated; unobserved
    locations gain entropy over time via volatility-weighted transition propagation,
    reducing attenuation.
    """

    name = "active-inference"
    config_schema = ActiveInferenceConfig

    def __init__(self) -> None:
        super().__init__()
        cfg = self._cfg()
        self._vocab = StateVocabulary(max_states=cfg.max_states)
        self._beliefs: dict[tuple[int, int], LocationBelief] = {}
        self._omega: float = cfg.omega_alpha_prior / cfg.omega_beta_prior
        self._omega_alpha: float = cfg.omega_alpha_prior
        self._omega_beta: float = cfg.omega_beta_prior
        self._last_tick: int = 0
        self._last_saliency_map: SaliencyMap | None = None

    def _cfg(self) -> ActiveInferenceConfig:
        assert isinstance(self.config, ActiveInferenceConfig)
        return self.config

    # Config fields are exposed as read/write properties so tests and callers can
    # tune parameters on an instance without reaching into ``self.config``.
    @property
    def max_states(self) -> int:
        return self._cfg().max_states

    @max_states.setter
    def max_states(self, value: int) -> None:
        self._cfg().max_states = value

    @property
    def max_locations(self) -> int:
        return self._cfg().max_locations

    @max_locations.setter
    def max_locations(self, value: int) -> None:
        self._cfg().max_locations = value

    @property
    def max_attenuation(self) -> float:
        return self._cfg().max_attenuation

    @max_attenuation.setter
    def max_attenuation(self, value: float) -> None:
        self._cfg().max_attenuation = value

    @property
    def saliency_weight(self) -> float:
        return self._cfg().saliency_weight

    @saliency_weight.setter
    def saliency_weight(self, value: float) -> None:
        self._cfg().saliency_weight = value

    @property
    def B_self_transition(self) -> float:
        return self._cfg().b_self_transition

    @B_self_transition.setter
    def B_self_transition(self, value: float) -> None:
        self._cfg().b_self_transition = value

    @Observability.tracer.start_as_current_span("saliency_attenuation_active_inference")
    def attenuate(
        self,
        strength_image: npt.NDArray[np.floating[Any]],
        saliency_map: SaliencyMap,
    ) -> npt.NDArray[np.floating[Any]]:
        """Attenuate strength image based on epistemic value at each cell."""
        self._last_saliency_map = saliency_map
        cfg = self._cfg()

        if len(self._beliefs) == 0:
            return strength_image

        max_entropy = np.log(cfg.max_states)
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
                epistemic_mult = (
                    1 - cfg.max_attenuation
                ) + cfg.max_attenuation * normalized_entropy
                effective_mult = (1 - cfg.saliency_weight) * epistemic_mult + cfg.saliency_weight
                result[bx, by] *= effective_mult

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
        from roc.framework.clock import Clock

        if len(focus_points) == 0 or self._last_saliency_map is None:
            return
        top = focus_points.iloc[0]
        sx, sy = int(top["x"]), int(top["y"])
        features = self._last_saliency_map.get_val(sx, sy)
        feature_strs = [str(f) for f in features]
        state_id = self._vocab.encode(feature_strs)
        self._update_belief(sx, sy, state_id, Clock.get())

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

    def _update_belief(self, x: int, y: int, state_id: int, tick: int) -> None:
        """Update belief at (x, y) after observation, update precision."""
        cfg = self._cfg()
        key = (x, y)
        if key not in self._beliefs:
            if len(self._beliefs) >= cfg.max_locations:
                self._evict_lru()
            self._beliefs[key] = LocationBelief.uniform(
                n_states=cfg.max_states,
                zeta=cfg.zeta_alpha_prior / cfg.zeta_beta_prior,
                zeta_alpha=cfg.zeta_alpha_prior,
                zeta_beta=cfg.zeta_beta_prior,
            )

        belief = self._beliefs[key]
        prediction_error = 1.0 - belief.q_s[state_id]
        belief.observe(state_id=state_id, zeta=belief.zeta)
        belief.last_observed_tick = tick
        self._update_volatility(prediction_error)
        self._update_precision(belief, prediction_error)

    def _update_precision(self, belief: LocationBelief, prediction_error: float) -> None:
        """Bayesian update of the zeta (precision) Gamma posterior."""
        belief.zeta_alpha += 0.5
        belief.zeta_beta += 0.5 * prediction_error**2
        belief.zeta = belief.zeta_alpha / belief.zeta_beta

    def _update_volatility(self, prediction_error: float) -> None:
        """Bayesian update of the omega (volatility) Gamma posterior."""
        self._omega_alpha += 0.5
        self._omega_beta += 0.5 * (1.0 - prediction_error) ** 2
        self._omega = self._omega_alpha / self._omega_beta

    def _evict_lru(self) -> None:
        """Evict the least-recently-observed belief."""
        if not self._beliefs:
            return
        oldest_key = min(self._beliefs, key=lambda k: self._beliefs[k].last_observed_tick)
        del self._beliefs[oldest_key]
