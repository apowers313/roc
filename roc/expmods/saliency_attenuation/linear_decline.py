"""Linear-decline saliency attenuation: recency-weighted spatial penalty."""

from __future__ import annotations

from collections import deque
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


class LinearDeclineConfig(ExpModConfig):
    """Configuration for linear-decline saliency attenuation."""

    capacity: int = 5
    radius: int = 3
    max_penalty: float = 1.0
    max_attenuation: float = 0.9


class LinearDeclineAttenuation(SaliencyAttenuationExpMod):
    """Attenuates recently attended locations with linear recency decline.

    Maintains a FIFO buffer of the last ``capacity`` attended locations. Each entry
    creates a spatial attenuation field in the strength image, with penalty
    declining linearly by recency rank and spatially by Manhattan distance.
    """

    name = "linear-decline"
    config_schema = LinearDeclineConfig

    def __init__(self) -> None:
        super().__init__()
        self._history: deque[AttendedLocation] = self._make_history()

    def _cfg(self) -> LinearDeclineConfig:
        assert isinstance(self.config, LinearDeclineConfig)
        return self.config

    # Config fields are exposed as read/write properties so tests and callers can
    # tune parameters on an instance without reaching into ``self.config``.
    @property
    def capacity(self) -> int:
        return self._cfg().capacity

    @capacity.setter
    def capacity(self, value: int) -> None:
        self._cfg().capacity = value

    @property
    def radius(self) -> int:
        return self._cfg().radius

    @radius.setter
    def radius(self, value: int) -> None:
        self._cfg().radius = value

    @property
    def max_penalty(self) -> float:
        return self._cfg().max_penalty

    @max_penalty.setter
    def max_penalty(self, value: float) -> None:
        self._cfg().max_penalty = value

    @property
    def max_attenuation(self) -> float:
        return self._cfg().max_attenuation

    @max_attenuation.setter
    def max_attenuation(self, value: float) -> None:
        self._cfg().max_attenuation = value

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
        from roc.framework.clock import Clock

        if len(focus_points) == 0:
            return
        top = focus_points.iloc[0]
        if self._history.maxlen != self.capacity:
            new_history: deque[AttendedLocation] = deque(self._history, maxlen=self.capacity)
            self._history = new_history
        self._history.append(
            AttendedLocation(
                x=int(top["x"]),
                y=int(top["y"]),
                tick=Clock.get(),
            )
        )

    def _compute_penalty(self, x: int, y: int) -> float:
        """Sum penalties from all history entries."""
        total = 0.0
        n = len(self._history)
        for i, loc in enumerate(reversed(list(self._history))):
            recency_weight = (n - i) / n
            dist = abs(x - loc.x) + abs(y - loc.y)
            spatial_weight = max(0.0, 1.0 - dist / self.radius)
            total += self.max_penalty * recency_weight * spatial_weight
        return total
