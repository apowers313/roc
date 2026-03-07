from abc import abstractmethod
from typing import Any

from .component import Component
from .event import Event, EventBus
from .expmod import ExpMod
from .sequencer import Frame
from .transformer import Transformer, TransformResult


class NoPrediction:
    None


PredictData = Frame | NoPrediction

PredictEvent = Event[PredictData]


class Predict(Component):
    name: str = "predict"
    type: str = "predict"
    auto: bool = True
    bus = EventBus[PredictData]("predict")

    def __init__(self) -> None:
        # print("predict init")
        super().__init__()
        self.predict_conn = self.connect_bus(Predict.bus)
        self.transformer_conn = self.connect_bus(Transformer.bus)
        self.transformer_conn.listen(self.do_predict)

    def event_filter(self, e: Event[Any]) -> bool:
        return isinstance(e.data, TransformResult)

    def do_predict(self, e: Event[TransformResult]) -> None:
        # print("do_predict")

        # print("e.data", e.data)
        # print("e.data", e.data.__class__.__name__)
        # print(e.data.transform.src_frame.objects)

        # get current frame
        transform_edges = e.data.transform.dst_edges
        transform_src = transform_edges.select(type="Change")
        if len(transform_src) < 1:
            return
        assert len(transform_src) == 1
        self.current_frame = transform_src[0].src
        # print("self.current_frame", self.current_frame.__class__.__name__)
        assert isinstance(self.current_frame, Frame)

        # find matching frame(s)
        candidates = PredictionCandidateFramesExpMod.get(default="object-based").get_candidates(
            self.current_frame
        )
        # print("candidates", candidates)
        if len(candidates) == 0:
            # print(">>> no prediction\n")
            self.predict_conn.send(NoPrediction())
            return

        # predict next frame(s) by applying transform(s) to current frame
        # TODO: play forward multiple frames?
        predicted_frames: list[Frame] = []
        for candidate in candidates:
            # edges = candidate.dst_edges.select(type="Change")
            # transforms += [e.dst for e in edges if isinstance(e.dst, Transform)]
            predicted_frames.append(Frame.merge_transforms(self.current_frame, candidate))

        # score predected frames
        scoring = PredictionConfidenceExpMod.get(default="naive")
        scored_frames = [scoring.calculate_confidence(f) for f in predicted_frames]
        max_score = max(scored_frames)
        idx = scored_frames.index(max_score)
        best_prediction = predicted_frames[idx]

        # take action that would lead to desired frame
        # print(">>> predict sending\n")
        self.predict_conn.send(best_prediction)


class PredictionCandidateFramesExpMod(ExpMod):
    modtype = "prediction-candidate"

    @abstractmethod
    def get_candidates(self, frame: Frame) -> list[Frame]: ...


class PredictionConfidenceExpMod(ExpMod):
    modtype = "prediction-confidence"

    @abstractmethod
    def calculate_confidence(self, f: Frame) -> float: ...


class ObjectBasedPrediction(PredictionCandidateFramesExpMod):
    name = "object-based"

    def get_candidates(self, frame: Frame) -> list[Frame]:
        ret: list[Frame] = []

        objs = frame.objects
        for obj in objs:
            ret.extend(obj.frames)

        return ret
