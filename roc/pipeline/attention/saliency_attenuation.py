"""Saliency attenuation ExpMod interface for inhibition of return.

The base class defines the contract that Attention uses to attenuate the strength
image inside ``SaliencyMap.get_focus()`` before peak-finding. Concrete flavors live
under ``roc/expmods/saliency_attenuation/``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from roc.framework.expmod import ExpMod
from roc.reporting.observability import Observability

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
    """Base class for saliency attenuation strategies.

    Concrete flavors live under ``roc/expmods/saliency_attenuation/``.
    """

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
