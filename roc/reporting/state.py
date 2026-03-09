"""Runtime state tracking and observability event emission for debugging and monitoring."""

from __future__ import annotations

import dataclasses
import subprocess
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar

import nle
import numpy as np

from roc.attention import Attention, SaliencyMap, VisionAttentionData
from roc.component import Component
from roc.config import Config
from roc.event import Event
from roc.graphdb import Edge, Node, Schema
from roc.location import DebugGrid
from roc.logger import logger
from roc.object import Object, ObjectResolver, ResolvedObject
from roc.reporting.observability import Observability, ObservabilityEvent, Observation, instance_id
from roc.sequencer import Sequencer  # noqa: F401

StateType = TypeVar("StateType")
_state_init_done = False


class StateComponent(Component):
    """Internal component used to listen on event buses for state updates."""

    name: str = "state"
    type: str = "reporting"


class State(ABC, Generic[StateType]):
    """Base class for tracking a piece of runtime state."""

    def __init__(self, name: str, display_name: str | None = None) -> None:
        self.name = name
        self.display_name = display_name or name
        self.val: StateType | None = None

    def __str__(self) -> str:
        return f"{self.display_name}: {self.val}"

    def get(self) -> StateType:
        """Returns the current state value."""
        if self.val is None:
            raise Exception("Trying to get state value before it is set")

        return self.val

    def set(self, v: StateType) -> None:
        """Sets the current state value."""
        self.val = v

    @staticmethod
    def get_states() -> "StateList":
        """Returns the global StateList singleton."""
        return states

    @staticmethod
    def get_state_names() -> list[str]:
        """Returns the names of all tracked states."""
        return [field.name for field in dataclasses.fields(StateList)]

    @staticmethod
    def init() -> None:
        """Initializes state tracking by connecting to event buses."""
        global _state_init_done
        if _state_init_done:
            return

        state_component = StateComponent()

        # attention
        att_conn = Attention.bus.connect(state_component)

        def att_evt_handler(e: Event[VisionAttentionData]) -> None:
            assert isinstance(e.data, VisionAttentionData)
            states.salency.set(deepcopy(e.data.saliency_map))
            states.attention.set(deepcopy(e.data))

        att_conn.listen(att_evt_handler, filter=lambda e: isinstance(e.data, VisionAttentionData))

        # object
        obj_conn = ObjectResolver.bus.connect(state_component)

        def obj_evt_handler(e: Event[ResolvedObject]) -> None:
            states.object.set(e.data)

        obj_conn.listen(obj_evt_handler, filter=lambda e: isinstance(e.data, Object))

        State.print_startup_info()

        _state_init_done = True

    @staticmethod
    def print_startup_info() -> None:
        """Logs system info, git status, and loaded components at startup."""
        logger.info(f"Starting ROC, instance id: {instance_id}")

        def log_cmd(
            msg: str,
            cmd: list[str],
            *,
            out: str = "stdout",
            multiline: bool = False,
        ) -> None:
            ret = subprocess.run(cmd, capture_output=True)
            if out == "stderr":
                output_bytes = ret.stderr
            else:
                output_bytes = ret.stdout

            if multiline:
                logger_fn = logger.debug
                separator = "\n"
            else:
                logger_fn = logger.info
                separator = ": "

            output = output_bytes.decode("utf-8").rstrip()
            logger_fn(f"{msg}{separator}{output}")

        log_cmd("git hash", ["git", "show", "--no-patch", '--pretty=format:"%H"'])
        log_cmd("git status", ["git", "status"], multiline=True)
        log_cmd("python location", ["which", "python"])
        log_cmd("python version", ["python", "--version"])
        log_cmd("uv version", ["uv", "--version"])
        log_cmd("python packages", ["uv", "pip", "list"], multiline=True)
        log_cmd("system info", ["uname", "-a"])
        log_cmd("cpu info", ["lscpu"], multiline=True)
        log_cmd("memory info", ["lsmem"], multiline=True)

        component_str = "\t" + "\n\t".join(Component.get_loaded_components())
        logger.debug(f"{Component.get_component_count()} components loaded\n{component_str}")

        settings = Config.get()
        logger.debug(f"config\n{settings}")

        schema = Schema()
        logger.debug(f"schema\n{schema.to_dot()}")

    @staticmethod
    def send_events() -> None:
        """Emits observability events for the current state of all tracked values."""
        states = State.get_states()

        if states.screen.val is not None:
            screen = states.screen.val["chars"]
            Observability.event(ScreenObsEvent(screen))

        if states.salency.val is not None:
            saliency = states.salency.val
            Observability.event(SaliencyObsEvent(saliency))
            Observability.event(FeatureObsEvent(states.salency))

        if states.object.val is not None:
            Observability.event(ObjectObsEvent(states.object))

        if states.attention.val is not None:
            #     Observability.event(AttentionObsEvent(states.attention.val))
            Observability.event(FocusObsEvent(states.attention.val))

    @staticmethod
    def print() -> None:
        """Prints all current state values to stdout."""
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
    """Tracks the current game loop iteration number."""

    def __init__(self) -> None:
        super().__init__("loop", display_name="Loop Number")
        self.val = 0

    def incr(self) -> None:
        """Increments the loop counter by one."""
        self.val = self.get() + 1


class NodeCacheState(State[float]):
    """Tracks the Node cache utilization ratio."""

    def __init__(self) -> None:
        super().__init__("node-cache", display_name="Node Cache")
        self.val = 0

    def get(self) -> float:
        """Returns the current cache utilization as a fraction."""
        c = Node.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Node.get_cache()
        return f"Node Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class EdgeCacheState(State[float]):
    """Tracks the Edge cache utilization ratio."""

    def __init__(self) -> None:
        super().__init__("edge-cache", display_name="Edge Cache")
        self.val = 0

    def get(self) -> float:
        """Returns the current cache utilization as a fraction."""
        c = Edge.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Edge.get_cache()
        return f"Edge Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class CurrentScreenState(State[dict[str, Any]]):
    """Tracks the most recent screen data from the environment."""

    def __init__(self) -> None:
        super().__init__("curr-screen", display_name="Current Screen")

    def set(self, screen: dict[str, Any]) -> None:
        """Sets the current screen data."""
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
    """Tracks the most recent saliency map."""

    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, sal: SaliencyMap) -> None:
        """Sets the current saliency map."""
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
    """Tracks the most recent attention data with focus points."""

    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, att: VisionAttentionData) -> None:
        """Sets the current attention data."""
        self.val = att

    def __str__(self) -> str:
        if self.val is not None:
            s = f"Current Attention:\n{str(self.val)}\n"
            return s
        else:
            return "Current Attention: None"


class CurrentObjectState(State[ResolvedObject]):
    """Tracks the most recently resolved object."""

    def __init__(self) -> None:
        super().__init__("curr-object", display_name="Current Object")

    def set(self, obj: ResolvedObject) -> None:
        """Sets the current resolved object."""
        self.val = obj

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Object:\n{str(self.val)}\n"
        else:
            return "Current Object: None"


class ComponentsState(State[list[str]]):
    """Tracks the list of loaded components."""

    def __init__(self) -> None:
        super().__init__("components", display_name="Components")
        self.val = []

    def get(self) -> list[str]:
        """Returns the list of currently loaded component names."""
        self.val = Component.get_loaded_components()
        return self.val

    def __str__(self) -> str:
        component_str = "\t" + "\n\t".join(self.get())
        return f"{Component.get_component_count()} components loaded:\n{component_str}"


class BlstatsState(State[list[tuple[str, str]]]):
    """Tracks the bottom-line stats from the NetHack environment."""


@dataclass
class StateList:
    """Container holding all tracked state objects."""

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
    """Converts a byte count to a human-readable string (e.g. '1.5GB')."""
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


class SaliencyObsEvent(ObservabilityEvent):
    """Observability event carrying saliency map data."""

    def __init__(self, sm: SaliencyMap) -> None:
        super().__init__("roc.attention.saliency", body=sm.to_html_vals())


class ObjectObsEvent(ObservabilityEvent):
    """Observability event carrying the current resolved object."""

    def __init__(self, o: CurrentObjectState) -> None:
        super().__init__("roc.attention.object", body=str(o))


class FeatureObsEvent(ObservabilityEvent):
    """Observability event carrying the feature report from a saliency map."""

    def __init__(self, sm: CurrentSaliencyMapState) -> None:
        s = ""
        if sm.val is not None:
            features = sm.val.feature_report()
            for feat_name in features:
                s += f"\t\t{feat_name}: {features[feat_name]}\n"
        else:
            s = "No features."

        super().__init__("roc.attention.features", body=s)


class AttentionObsEvent(ObservabilityEvent):
    """Observability event carrying the attention grid with focus point overlays."""

    def __init__(self, vd: VisionAttentionData) -> None:
        sm = vd.saliency_map
        assert sm.grid is not None
        dg = DebugGrid(sm.grid)

        for idx, row in vd.focus_points.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            dg.set_style(x, y, back_brightness=row["strength"], back_hue=1)

        super().__init__("roc.attention.grid", body=str(dg.to_html_vals()))


class FocusObsEvent(ObservabilityEvent):
    """Observability event carrying the focus points from attention."""

    def __init__(self, vd: VisionAttentionData) -> None:
        super().__init__("roc.attention.focus_points", body=str(vd.focus_points))


class ScreenObsEvent(ObservabilityEvent):
    """Observability event carrying the current screen text."""

    def __init__(self, tty_chars: np.ndarray[Any, Any]) -> None:
        screen = ""
        for row in tty_chars:
            for ch in row:
                screen += chr(ch)
            screen += "\n"
        super().__init__("roc.screen", body=screen)


class IntrinsicObsEvent(ObservabilityEvent):
    """Observability event carrying intrinsic state (bottom-line stats)."""

    def __init__(self, bl: dict[str, Any]) -> None:
        super().__init__("roc.intrinsics", body=bl)


def node_cache_gague(*args: Any) -> Iterable[Observation]:
    """OpenTelemetry gauge callback that reports Node cache size and triggers state events."""
    # NOTE: need send and print state events every time metrics are recorded, just
    # sticking this here because it needs to be somewhere
    State.send_events()
    State.print()

    c = Node.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


def edge_cache_gague(*args: Any) -> Iterable[Observation]:
    """OpenTelemetry gauge callback that reports Edge cache size."""
    c = Edge.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


Observability.meter.create_observable_gauge("roc.node_cache", callbacks=[node_cache_gague])
Observability.meter.create_observable_gauge("roc.edge_cache", callbacks=[edge_cache_gague])
