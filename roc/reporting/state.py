from __future__ import annotations

import dataclasses
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar

import nle
import numpy as np

from roc.attention import Attention, SaliencyMap, VisionAttentionData
from roc.component import Component
from roc.event import Event
from roc.graphdb import Edge, Node
from roc.logger import logger
from roc.object import Object, ObjectResolver
from roc.reporting.observability import Observability, ObservabilityEvent, Observation

StateType = TypeVar("StateType")
_state_init_done = False


class StateComponent(Component):
    pass


class State(ABC, Generic[StateType]):
    def __init__(self, name: str, display_name: str | None = None) -> None:
        self.name = name
        self.display_name = display_name or name
        self.val: StateType | None = None

    def __str__(self) -> str:
        return f"{self.display_name}: {self.val}"

    def get(self) -> StateType:
        if self.val is None:
            raise Exception("Trying to get state value before it is set")

        return self.val

    def set(self, v: StateType) -> None:
        self.val = v

    @staticmethod
    def get_states() -> "StateList":
        return states

    @staticmethod
    def get_state_names() -> list[str]:
        return [field.name for field in dataclasses.fields(StateList)]

    @staticmethod
    def init() -> None:
        global _state_init_done
        if _state_init_done:
            return

        # attention
        att_conn = Attention.bus.connect(StateComponent())

        def att_evt_handler(e: Event[VisionAttentionData]) -> None:
            assert isinstance(e.data, VisionAttentionData)
            states.salency.set(deepcopy(e.data.saliency_map))
            states.attention.set(deepcopy(e.data))

        att_conn.listen(att_evt_handler, filter=lambda e: isinstance(e.data, VisionAttentionData))

        # object
        obj_conn = ObjectResolver.bus.connect(StateComponent())

        def obj_evt_handler(e: Event[Object]) -> None:
            states.object.set(e.data)

        obj_conn.listen(obj_evt_handler, filter=lambda e: isinstance(e.data, Object))

        State.print_startup_info()

        _state_init_done = True

    @staticmethod
    def print_startup_info() -> None:
        logger.info("Starting ROC")

    @staticmethod
    def send_events() -> None:
        states = State.get_states()

        if states.screen.val is not None:
            screen = states.screen.val["chars"]
            Observability.event(ScreenObsEvent(screen))

        if states.salency.val is not None:
            saliency = states.salency.val
            Observability.event(SaliencyEvent(saliency))

    @staticmethod
    def print() -> None:
        State.init()

        def header(s: str) -> None:
            print(f"\n=== {s.upper()} ===")  # noqa: T201

        header("Environment")
        print(states.loop)  # noqa: T201
        print(states.screen)  # noqa: T201
        # TODO: blstats

        header("Graph DB")
        print(states.node_cache)  # noqa: T201
        print(states.edge_cache)  # noqa: T201

        header("Agent")
        print(states.salency)  # noqa: T201
        print(states.attention)  # noqa: T201
        print(states.object)  # noqa: T201


class LoopState(State[int]):
    def __init__(self) -> None:
        super().__init__("loop", display_name="Loop Number")
        self.val = 0

    def incr(self) -> None:
        self.val = self.get() + 1


class NodeCacheState(State[float]):
    def __init__(self) -> None:
        super().__init__("node-cache", display_name="Node Cache")
        self.val = 0

    def get(self) -> float:
        c = Node.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Node.get_cache()
        return f"Node Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class EdgeCacheState(State[float]):
    def __init__(self) -> None:
        super().__init__("edge-cache", display_name="Edge Cache")
        self.val = 0

    def get(self) -> float:
        c = Edge.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Edge.get_cache()
        return f"Edge Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class CurrentScreenState(State[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__("curr-screen", display_name="Current Screen")

    def set(self, screen: dict[str, Any]) -> None:
        self.val = screen

    def __str__(self) -> str:
        # save the current screen

        if self.val is not None:
            screen = nle.nethack.tty_render(
                self.val["chars"], self.val["colors"], self.val["cursor"]
            )
            return f"Current Screen:\n-------------\n{screen}\n-------------"
        else:
            return "Current Screen: None"


class CurrentSaliencyMapState(State[SaliencyMap]):
    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, sal: SaliencyMap) -> None:
        self.val = sal

    def __str__(self) -> str:
        if self.val is not None:
            s = f"Current Saliency Map:\n{str(self.val)}\n"
            s += "\tFeatures:\n"
            features = self.val.feature_report()
            for feat_name in features:
                s += f"\t\t{feat_name}: {features[feat_name]}\n"
            return s
        else:
            return "Current Saliency Map: None"


class CurrentAttentionState(State[VisionAttentionData]):
    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, att: VisionAttentionData) -> None:
        self.val = att

    def __str__(self) -> str:
        if self.val is not None:
            s = f"Current Attention:\n{str(self.val)}\n"
            return s
        else:
            return "Current Attention: None"


class CurrentObjectState(State[Object]):
    def __init__(self) -> None:
        super().__init__("curr-object", display_name="Current Object")

    def set(self, obj: Object) -> None:
        self.val = obj

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Object:\n{str(self.val)}\n"
        else:
            return "Current Object: None"


class ComponentsState(State[list[str]]):
    def __init__(self) -> None:
        super().__init__("components", display_name="Components")
        self.val = []

    def get(self) -> list[str]:
        self.val = Component.get_loaded_components()
        return self.val

    def __str__(self) -> str:
        component_str = "\t" + "\n\t".join(self.get())
        return f"{Component.get_component_count()} components loaded:\n{component_str}"


class BlstatsState(State[list[tuple[str, str]]]):
    pass


@dataclass
class StateList:
    loop: LoopState = LoopState()
    node_cache: NodeCacheState = NodeCacheState()
    edge_cache: EdgeCacheState = EdgeCacheState()
    screen: CurrentScreenState = CurrentScreenState()
    salency: CurrentSaliencyMapState = CurrentSaliencyMapState()
    attention: CurrentAttentionState = CurrentAttentionState()
    object: CurrentObjectState = CurrentObjectState()
    components: ComponentsState = ComponentsState()


states = StateList()


def bytes2human(n: int) -> str:
    # stolen from: https://psutil.readthedocs.io/en/latest/#recipes
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return "{:.1f}{}B".format(value, s)
    return "%sB" % n


class SaliencyEvent(ObservabilityEvent):
    def __init__(self, sm: SaliencyMap) -> None:
        super().__init__("roc.attention.saliency", body=sm.to_html_vals())


class ScreenObsEvent(ObservabilityEvent):
    def __init__(self, tty_chars: np.ndarray[Any, Any]) -> None:
        screen = ""
        for row in tty_chars:
            for ch in row:
                screen += chr(ch)
            screen += "\n"
        super().__init__("roc.screen", body=screen)


class IntrinsicObsEvent(ObservabilityEvent):
    def __init__(self, bl: dict[str, Any]) -> None:
        super().__init__("roc.intrinsics", body=bl)


def node_cache_gague(*args: Any) -> Iterable[Observation]:
    # NOTE: need send state events every time metrics are recorded, just
    # sticking this here because it needs to be somewhere
    State.send_events()

    c = Node.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


def edge_cache_gague(*args: Any) -> Iterable[Observation]:
    c = Edge.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


Observability.meter.create_observable_gauge("roc.node_cache", callbacks=[node_cache_gague])
Observability.meter.create_observable_gauge("roc.edge_cache", callbacks=[edge_cache_gague])
