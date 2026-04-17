"""No-op saliency attenuation: returns input unchanged."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from roc.pipeline.attention.saliency_attenuation import SaliencyAttenuationExpMod

if TYPE_CHECKING:
    from roc.pipeline.attention.attention import SaliencyMap


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
