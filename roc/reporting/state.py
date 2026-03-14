"""Runtime state tracking and observability event emission for debugging and monitoring."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from time import time_ns
from typing import Any, Generic, Iterable, TypeVar

import nle
from opentelemetry import trace as otel_trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LogRecord

from roc.attention import Attention, SaliencyMap, VisionAttentionData
from roc.component import Component
from roc.config import Config
from roc.event import Event
from roc.graphdb import Edge, Node, Schema
from roc.logger import logger
from roc.object import Object, ObjectResolver, ResolvedObject
from roc.reporting.observability import Observability, Observation, instance_id, resource
from roc.sequencer import Sequencer  # noqa: F401

StateType = TypeVar("StateType")
_state_init_done = False


def _get_otel_logger() -> Any:
    """Get a fresh OTel logger bound to the current provider."""
    return Observability.get_logger("roc.state")


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
        log_cmd("python location", ["which", "python3"])
        log_cmd("python version", ["python3", "--version"])
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
    def emit_state_logs() -> None:
        """Emits current state values as OTel log records."""
        current_states = State.get_states()

        if current_states.screen.val is not None:
            from roc.reporting.screen_renderer import screen_to_html_vals

            screen_vals = screen_to_html_vals(current_states.screen.val)
            _emit_state_record("roc.screen", json.dumps(screen_vals, separators=(",", ":")))

        if current_states.salency.val is not None:
            saliency = current_states.salency.val
            saliency_vals = saliency.to_html_vals()
            _emit_state_record(
                "roc.attention.saliency", json.dumps(saliency_vals, separators=(",", ":"))
            )
            s = ""
            features = saliency.feature_report()
            for feat_name in features:
                s += f"\t\t{feat_name}: {features[feat_name]}\n"
            _emit_state_record("roc.attention.features", s)

        if current_states.object.val is not None:
            _emit_state_record("roc.attention.object", str(current_states.object))

        if current_states.attention.val is not None:
            _emit_state_record(
                "roc.attention.focus_points", str(current_states.attention.val.focus_points)
            )

        # Graph DB summary
        node_cache = Node.get_cache()
        edge_cache = Edge.get_cache()
        _emit_state_record(
            "roc.graphdb.summary",
            json.dumps(
                {
                    "node_count": node_cache.currsize,
                    "node_max": node_cache.maxsize,
                    "edge_count": edge_cache.currsize,
                    "edge_max": edge_cache.maxsize,
                },
                separators=(",", ":"),
            ),
        )

        # Event bus activity summary
        step_counts = Event.get_step_counts()
        if step_counts:
            _emit_state_record(
                "roc.event.summary",
                json.dumps(step_counts, separators=(",", ":")),
            )

    @staticmethod
    def maybe_emit_snapshot(tick: int) -> None:
        """Emit a state snapshot as an OTel log record if the tick matches the interval.

        Args:
            tick: The current tick number.
        """
        settings = Config.get()
        interval = settings.debug_snapshot_interval
        if interval <= 0 or tick <= 0 or tick % interval != 0:
            return

        snapshot: dict[str, Any] = {"tick": tick}

        # Screen
        if states.screen.val is not None:
            screen_text = ""
            for row in states.screen.val["chars"]:
                for ch in row:
                    screen_text += chr(ch)
                screen_text += "\n"
            snapshot["screen"] = screen_text
        else:
            snapshot["screen"] = None

        # Objects
        if states.object.val is not None:
            snapshot["objects"] = str(states.object.val)
        else:
            snapshot["objects"] = None

        # Loop state
        snapshot["loop"] = states.loop.val

        span_context = otel_trace.get_current_span().get_span_context()
        log_record = LogRecord(
            timestamp=time_ns(),
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=json.dumps(snapshot, default=str),
            resource=resource,
            attributes={"event.name": "roc.state.snapshot"},
            trace_id=span_context.trace_id,
            span_id=span_context.span_id,
            trace_flags=span_context.trace_flags,
        )
        _get_otel_logger().emit(log_record)

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


def _emit_state_record(event_name: str, body: str) -> None:
    """Emit a state value as an OTel log record."""
    span_context = otel_trace.get_current_span().get_span_context()
    log_record = LogRecord(
        timestamp=time_ns(),
        severity_number=SeverityNumber.INFO,
        severity_text="INFO",
        body=body,
        resource=resource,
        attributes={"event.name": event_name},
        trace_id=span_context.trace_id,
        span_id=span_context.span_id,
        trace_flags=span_context.trace_flags,
    )
    _get_otel_logger().emit(log_record)


def node_cache_gague(*args: Any) -> Iterable[Observation]:
    """OpenTelemetry gauge callback that reports Node cache size."""
    c = Node.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


def edge_cache_gague(*args: Any) -> Iterable[Observation]:
    """OpenTelemetry gauge callback that reports Edge cache size."""
    c = Edge.get_cache()
    yield Observation(c.currsize, attributes={"max": c.maxsize})


Observability.meter.create_observable_gauge("roc.node_cache", callbacks=[node_cache_gague])
Observability.meter.create_observable_gauge("roc.edge_cache", callbacks=[edge_cache_gague])
