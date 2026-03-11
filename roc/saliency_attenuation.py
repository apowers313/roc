"""Saliency attenuation ExpMod for inhibition of return.

Attenuates the strength image inside SaliencyMap.get_focus() before
peak-finding. The base class defines the interface; concrete implementations
apply different attenuation strategies to bias attention away from recently
attended locations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

from .expmod import ExpMod
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

        # Record telemetry
        max_penalty_histogram.record(overall_max_penalty)
        history_size_histogram.record(len(self._history))

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
