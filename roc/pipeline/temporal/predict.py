"""Prediction system that uses past transforms to predict future frames."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...framework.component import Component
from ...framework.event import Event, EventBus
from ...framework.expmod import ExpMod
from .sequencer import Frame
from .transformer import Transformer, TransformResult


class NoPrediction:
    """Sentinel indicating that no prediction could be made."""


@dataclass
class PredictionMeta:
    """Metadata about the most recent prediction attempt."""

    made: bool = False
    candidate_count: int = 0
    confidence: float = 0.0
    all_scores: list[float] = field(default_factory=list)
    predicted_intrinsics: dict[str, float] = field(default_factory=dict)


PredictData = Frame | NoPrediction

PredictEvent = Event[PredictData]


class Predict(Component):
    """Component that predicts future frames based on past transforms."""

    name: str = "predict"
    type: str = "predict"
    auto: bool = True
    bus = EventBus[PredictData]("predict")

    def __init__(self) -> None:
        super().__init__()
        self.predict_conn = self.connect_bus(Predict.bus)
        self.transformer_conn = self.connect_bus(Transformer.bus)
        self.transformer_conn.listen(self.do_predict)
        self.last_prediction_meta = PredictionMeta()

    def event_filter(self, e: Event[Any]) -> bool:
        """Only process TransformResult events."""
        return isinstance(e.data, TransformResult)

    def do_predict(self, e: Event[TransformResult]) -> None:
        """Finds candidate frames, applies transforms, scores predictions, and emits the best one."""
        from ..intrinsic import IntrinsicNode

        # get current frame
        transform_edges = e.data.transform.dst_edges
        transform_src = transform_edges.select(type="Change")
        if len(transform_src) < 1:
            return
        assert len(transform_src) == 1
        self.current_frame = transform_src[0].src
        assert isinstance(self.current_frame, Frame)

        # find matching frame(s)
        candidates = PredictionCandidateFramesExpMod.get(default="object-based").get_candidates(
            self.current_frame
        )
        if len(candidates) == 0:
            self.last_prediction_meta = PredictionMeta(
                made=False,
                candidate_count=0,
            )
            self.predict_conn.send(NoPrediction())
            return

        # predict next frame(s) by applying transform(s) to current frame
        predicted_frames: list[Frame] = []
        for candidate in candidates:
            predicted_frames.append(Frame.merge_transforms(self.current_frame, candidate))

        # score predicted frames
        scoring = PredictionConfidenceExpMod.get(default="naive")
        scored_frames = [scoring.calculate_confidence(f) for f in predicted_frames]
        max_score = max(scored_frames)
        idx = scored_frames.index(max_score)
        best_prediction = predicted_frames[idx]

        # extract predicted intrinsics from best prediction
        predicted_intrinsics: dict[str, float] = {}
        for t in best_prediction.transformable:
            if isinstance(t, IntrinsicNode):
                predicted_intrinsics[t.name] = t.normalized_value

        self.last_prediction_meta = PredictionMeta(
            made=True,
            candidate_count=len(candidates),
            confidence=max_score,
            all_scores=scored_frames,
            predicted_intrinsics=predicted_intrinsics,
        )

        self.predict_conn.send(best_prediction)


class PredictionCandidateFramesExpMod(ExpMod):
    """Base class for modules that find candidate frames for prediction.

    Concrete implementations live under ``roc/expmods/prediction_candidate/``.
    """

    modtype = "prediction-candidate"

    @abstractmethod
    def get_candidates(self, frame: Frame) -> list[Frame]:
        """Returns frames that are candidates for predicting the next state."""
        ...


class PredictionConfidenceExpMod(ExpMod):
    """Base class for modules that score how confident a prediction is.

    Concrete implementations live under ``roc/expmods/prediction_confidence/``.
    """

    modtype = "prediction-confidence"

    @abstractmethod
    def calculate_confidence(self, f: Frame) -> float:
        """Returns a confidence score for a predicted frame."""
        ...
