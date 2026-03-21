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

from roc.action import Action, ActionData, TakeAction
from roc.attention import Attention, SaliencyMap, VisionAttentionData
from roc.component import Component
from roc.config import Config
from roc.event import Event
from roc.graphdb import Edge, Node, Schema
from roc.intrinsic import Intrinsic, IntrinsicData
from roc.logger import logger
from roc.object import Object, ObjectResolver, ResolvedObject
from roc.feature_extractors.phoneme import PhonemeFeature, PhonemeWord
from roc.perception import AuditoryData, Perception, PerceptionData
from roc.predict import NoPrediction, Predict, PredictData
from roc.reporting.observability import Observability, Observation, instance_id, resource
from roc.sequencer import Sequencer  # noqa: F401
from roc.significance import Significance, SignificanceData
from roc.transformer import TransformResult, Transformer

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

        # intrinsics
        intr_conn = Intrinsic.bus.connect(state_component)

        def intr_evt_handler(e: Event[IntrinsicData]) -> None:
            assert isinstance(e.data, IntrinsicData)
            states.intrinsic.set(e.data)

        intr_conn.listen(intr_evt_handler, filter=lambda e: isinstance(e.data, IntrinsicData))

        # significance
        sig_conn = Significance.bus.connect(state_component)

        def sig_evt_handler(e: Event[SignificanceData]) -> None:
            assert isinstance(e.data, SignificanceData)
            states.significance.set(e.data)

        sig_conn.listen(sig_evt_handler, filter=lambda e: isinstance(e.data, SignificanceData))

        # action
        act_conn = Action.bus.connect(state_component)

        def act_evt_handler(e: Event[ActionData]) -> None:
            assert isinstance(e.data, TakeAction)
            states.action.set(e.data)

        act_conn.listen(act_evt_handler, filter=lambda e: isinstance(e.data, TakeAction))

        # transformer
        txf_conn = Transformer.bus.connect(state_component)

        def txf_evt_handler(e: Event[TransformResult]) -> None:
            assert isinstance(e.data, TransformResult)
            states.transform.set(e.data)

        txf_conn.listen(txf_evt_handler, filter=lambda e: isinstance(e.data, TransformResult))

        # predict
        pred_conn = Predict.bus.connect(state_component)

        def pred_evt_handler(e: Event[PredictData]) -> None:
            if not isinstance(e.data, NoPrediction):
                states.predict.set(e.data)

        pred_conn.listen(pred_evt_handler, filter=lambda e: not isinstance(e.data, NoPrediction))

        # auditory messages
        msg_conn = Perception.bus.connect(state_component)

        def msg_evt_handler(e: Event[PerceptionData]) -> None:
            assert isinstance(e.data, AuditoryData)
            states.message.set(e.data.msg.strip("\x00").strip())

        msg_conn.listen(msg_evt_handler, filter=lambda e: isinstance(e.data, AuditoryData))

        # phonemes
        phoneme_conn = Perception.bus.connect(state_component)

        def phoneme_evt_handler(e: Event[PerceptionData]) -> None:
            assert isinstance(e.data, PhonemeFeature)
            states.phonemes.set(list(e.data.phonemes))

        phoneme_conn.listen(
            phoneme_evt_handler, filter=lambda e: isinstance(e.data, PhonemeFeature)
        )

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
        from roc.config import Config

        cfg = Config.get()
        current_states = State.get_states()

        if cfg.emit_state_screen and current_states.screen.val is not None:
            from roc.reporting.screen_renderer import screen_to_html_vals

            screen_vals = screen_to_html_vals(current_states.screen.val)
            _emit_state_record("roc.screen", json.dumps(screen_vals, separators=(",", ":")))

        if cfg.emit_state_saliency and current_states.salency.val is not None:
            saliency = current_states.salency.val
            saliency_vals = saliency.to_html_vals()
            _emit_state_record(
                "roc.attention.saliency", json.dumps(saliency_vals, separators=(",", ":"))
            )
            if cfg.emit_state_features:
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

        # Intrinsics
        if current_states.intrinsic.val is not None:
            intr_data = current_states.intrinsic.val
            _emit_state_record(
                "roc.intrinsics",
                json.dumps(
                    {
                        "raw": intr_data.intrinsics,
                        "normalized": intr_data.normalized_intrinsics,
                    },
                    separators=(",", ":"),
                    default=str,
                ),
            )

        # Significance
        if current_states.significance.val is not None:
            _emit_state_record(
                "roc.significance",
                json.dumps(
                    {"significance": current_states.significance.val.significance},
                    separators=(",", ":"),
                ),
            )

        # Action
        if current_states.action.val is not None:
            action_val = current_states.action.val.action
            action_dict: dict[str, Any] = {"action_id": int(action_val)}
            try:
                gym_actions = cfg.gym_actions
                if gym_actions and int(action_val) < len(gym_actions):
                    act_enum = gym_actions[int(action_val)]
                    action_dict["action_name"] = str(getattr(act_enum, "name", act_enum))
            except Exception:
                pass
            try:
                from roc.action import DefaultActionExpMod

                action_dict["expmod_name"] = DefaultActionExpMod.get(default="pass").name
            except Exception:
                pass
            _emit_state_record(
                "roc.action",
                json.dumps(action_dict, separators=(",", ":")),
            )

        # Transform summary
        if current_states.transform.val is not None:
            t = current_states.transform.val.transform
            changes = []
            for edge in t.src_edges:
                changes.append(str(edge.dst))
            _emit_state_record(
                "roc.transform_summary",
                json.dumps(
                    {"count": len(changes), "changes": changes},
                    separators=(",", ":"),
                    default=str,
                ),
            )

        # Prediction
        if current_states.predict.val is not None:
            pred = current_states.predict.val
            pred_dict: dict[str, Any] = {"made": not isinstance(pred, NoPrediction)}
            try:
                from roc.predict import (
                    PredictionCandidateFramesExpMod,
                    PredictionConfidenceExpMod,
                )

                pred_dict["candidate_expmod"] = PredictionCandidateFramesExpMod.get(
                    default="object-based"
                ).name
                pred_dict["confidence_expmod"] = PredictionConfidenceExpMod.get(
                    default="naive"
                ).name
            except Exception:
                pass
            _emit_state_record(
                "roc.prediction",
                json.dumps(pred_dict, separators=(",", ":")),
            )

        # Message
        if current_states.message.val is not None:
            msg = current_states.message.val.strip()
            if msg:
                _emit_state_record("roc.message", msg)

        # Phonemes
        if current_states.phonemes.val is not None:
            phoneme_dicts = [
                {"word": pw.word, "phonemes": pw.phonemes, "is_break": pw.is_break}
                for pw in current_states.phonemes.val
            ]
            _emit_state_record(
                "roc.phonemes",
                json.dumps(phoneme_dicts, separators=(",", ":")),
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


class CurrentIntrinsicState(State[IntrinsicData]):
    """Tracks the most recent intrinsic data."""

    def __init__(self) -> None:
        super().__init__("curr-intrinsic", display_name="Current Intrinsics")

    def set(self, data: IntrinsicData) -> None:
        """Sets the current intrinsic data."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Intrinsics:\n{repr(self.val)}\n"
        else:
            return "Current Intrinsics: None"


class CurrentSignificanceState(State[SignificanceData]):
    """Tracks the most recent significance score."""

    def __init__(self) -> None:
        super().__init__("curr-significance", display_name="Current Significance")

    def set(self, data: SignificanceData) -> None:
        """Sets the current significance data."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Significance: {self.val.significance}"
        else:
            return "Current Significance: None"


class CurrentActionState(State[TakeAction]):
    """Tracks the most recent action taken."""

    def __init__(self) -> None:
        super().__init__("curr-action", display_name="Current Action")

    def set(self, data: TakeAction) -> None:
        """Sets the current action."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Action: {self.val.action}"
        else:
            return "Current Action: None"


class CurrentTransformState(State[TransformResult]):
    """Tracks the most recent transform result."""

    def __init__(self) -> None:
        super().__init__("curr-transform", display_name="Current Transform")

    def set(self, data: TransformResult) -> None:
        """Sets the current transform result."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Transform:\n{str(self.val.transform)}\n"
        else:
            return "Current Transform: None"


class CurrentPredictState(State[PredictData]):
    """Tracks the most recent prediction result."""

    def __init__(self) -> None:
        super().__init__("curr-predict", display_name="Current Prediction")

    def set(self, data: PredictData) -> None:
        """Sets the current prediction data."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Prediction: {type(self.val).__name__}"
        else:
            return "Current Prediction: None"


class CurrentMessageState(State[str]):
    """Tracks the most recent auditory message."""

    def __init__(self) -> None:
        super().__init__("curr-message", display_name="Current Message")

    def set(self, msg: str) -> None:
        """Sets the current message."""
        self.val = msg

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Message: {self.val}"
        else:
            return "Current Message: None"


class CurrentPhonemeState(State[list[PhonemeWord]]):
    """Tracks the most recent phoneme decomposition."""

    def __init__(self) -> None:
        super().__init__("curr-phonemes", display_name="Current Phonemes")

    def set(self, phonemes: list[PhonemeWord]) -> None:
        """Sets the current phonemes."""
        self.val = phonemes

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Phonemes: {len(self.val)} entries"
        else:
            return "Current Phonemes: None"


class CurrentResolutionState(State[dict[str, Any]]):
    """Tracks the most recent object resolution decision (set from OTel emission site)."""

    def __init__(self) -> None:
        super().__init__("curr-resolution", display_name="Current Resolution")

    def set(self, data: dict[str, Any]) -> None:
        """Sets the current resolution decision data."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Resolution: {self.val.get('outcome', 'unknown')}"
        else:
            return "Current Resolution: None"


class CurrentAttenuationState(State[dict[str, Any]]):
    """Tracks the most recent attenuation data (set from OTel emission site)."""

    def __init__(self) -> None:
        super().__init__("curr-attenuation", display_name="Current Attenuation")

    def set(self, data: dict[str, Any]) -> None:
        """Sets the current attenuation data."""
        self.val = data

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Attenuation: {self.val.get('flavor', 'unknown')}"
        else:
            return "Current Attenuation: None"


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
    intrinsic: CurrentIntrinsicState = CurrentIntrinsicState()
    significance: CurrentSignificanceState = CurrentSignificanceState()
    action: CurrentActionState = CurrentActionState()
    transform: CurrentTransformState = CurrentTransformState()
    predict: CurrentPredictState = CurrentPredictState()
    message: CurrentMessageState = CurrentMessageState()
    phonemes: CurrentPhonemeState = CurrentPhonemeState()
    resolution: CurrentResolutionState = CurrentResolutionState()
    attenuation_data: CurrentAttenuationState = CurrentAttenuationState()
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
