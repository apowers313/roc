"""Object-based prediction-candidate ExpMod."""

from __future__ import annotations

from roc.pipeline.temporal.predict import PredictionCandidateFramesExpMod
from roc.pipeline.temporal.sequencer import Frame


class ObjectBasedPrediction(PredictionCandidateFramesExpMod):
    """Finds prediction candidates by looking up frames associated with current objects."""

    name = "object-based"

    def get_candidates(self, frame: Frame) -> list[Frame]:
        """Return all frames that share objects with the given frame."""
        ret: list[Frame] = []
        for obj in frame.objects:
            ret.extend(obj.frames)
        return ret
