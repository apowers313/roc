"""Runtime state tracking and observability event emission for debugging and monitoring."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from time import time_ns
from typing import Any, Iterable

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

_state_init_done = False


def _get_otel_logger() -> Any:
    """Get a fresh OTel logger bound to the current provider."""
    return Observability.get_logger("roc.state")


class StateComponent(Component):
    """Internal component used to listen on event buses for state updates."""

    name: str = "state"
    type: str = "reporting"


class State[StateType](ABC):
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
            raise RuntimeError("Trying to get state value before it is set")

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

        # attention -- subscribe synchronously (no scheduler) so that
        # states.salency is updated inline when VisionAttention emits.
        Attention.bus.connect(state_component)

        def att_evt_handler(e: Any) -> None:
            if not isinstance(e.data, VisionAttentionData):
                return
            states.salency.set(deepcopy(e.data.saliency_map))
            states.attention.set(deepcopy(e.data))

        Attention.bus.subject.subscribe(att_evt_handler)

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

        # Save schema to the run directory and emit as OTel event
        schema_dict = schema.to_dict()
        store = Observability.get_ducklake_store()
        if store is not None:
            schema_path = store.run_dir / "schema.json"
            schema_path.write_text(json.dumps(schema_dict, indent=2))
            logger.debug("Schema saved to {}", schema_path)
        _emit_state_record("roc.schema", json.dumps(schema_dict, separators=(",", ":")))

    @staticmethod
    def emit_state_logs() -> None:
        """Emits current state values as OTel log records."""
        from roc.config import Config

        cfg = Config.get()
        current_states = State.get_states()

        _emit_screen_log(cfg, current_states)
        _emit_saliency_log(cfg, current_states)
        _emit_object_and_attention_log(current_states)
        _emit_intrinsics_log(current_states)
        _emit_significance_log(current_states)
        _emit_action_log(cfg, current_states)
        _emit_transform_log(current_states)
        _emit_sequence_log(current_states)
        _emit_prediction_log(current_states)
        _emit_message_log(current_states)
        _emit_phonemes_log(current_states)
        _emit_graphdb_summary_log()
        _emit_event_summary_log()

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

        snapshot: dict[str, Any] = {
            "tick": tick,
            "screen": _render_screen_text(states.screen.val),
            "objects": str(states.object.val) if states.object.val is not None else None,
            "loop": states.loop.val,
        }

        _emit_state_record("roc.state.snapshot", json.dumps(snapshot, default=str))

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


def _render_screen_text(screen_val: dict[str, Any] | None) -> str | None:
    """Render screen chars to a text string, or return None if no screen data."""
    if screen_val is None:
        return None
    lines: list[str] = []
    for row in screen_val["chars"]:
        lines.append("".join(chr(ch) for ch in row))
    return "\n".join(lines) + "\n"


def _emit_screen_log(cfg: Any, current_states: "StateList") -> None:
    """Emit screen state as an OTel log record."""
    if not cfg.emit_state_screen or current_states.screen.val is None:
        return
    from roc.reporting.screen_renderer import screen_to_html_vals

    screen_vals = screen_to_html_vals(current_states.screen.val)
    _emit_state_record("roc.screen", json.dumps(screen_vals, separators=(",", ":")))


def _emit_saliency_log(cfg: Any, current_states: "StateList") -> None:
    """Emit saliency and feature state as OTel log records."""
    if not cfg.emit_state_saliency or current_states.salency.val is None:
        return
    saliency = current_states.salency.val
    saliency_vals = saliency.to_html_vals()
    _emit_state_record("roc.attention.saliency", json.dumps(saliency_vals, separators=(",", ":")))
    if not cfg.emit_state_features:
        return
    s = ""
    features = saliency.feature_report()
    for feat_name in features:
        s += f"\t\t{feat_name}: {features[feat_name]}\n"
    _emit_state_record("roc.attention.features", s)


def _emit_object_and_attention_log(current_states: "StateList") -> None:
    """Emit object and focus point state as OTel log records."""
    if current_states.object.val is not None:
        _emit_state_record("roc.attention.object", str(current_states.object))
    if current_states.attention.val is not None:
        _emit_state_record(
            "roc.attention.focus_points", str(current_states.attention.val.focus_points)
        )


def _emit_intrinsics_log(current_states: "StateList") -> None:
    """Emit intrinsics state as an OTel log record."""
    if current_states.intrinsic.val is None:
        return
    intr_data = current_states.intrinsic.val
    _emit_state_record(
        "roc.intrinsics",
        json.dumps(
            {"raw": intr_data.intrinsics, "normalized": intr_data.normalized_intrinsics},
            separators=(",", ":"),
            default=str,
        ),
    )


def _emit_significance_log(current_states: "StateList") -> None:
    """Emit significance state as an OTel log record."""
    if current_states.significance.val is None:
        return
    _emit_state_record(
        "roc.significance",
        json.dumps(
            {"significance": current_states.significance.val.significance},
            separators=(",", ":"),
        ),
    )


def _emit_action_log(cfg: Any, current_states: "StateList") -> None:
    """Emit action state as an OTel log record."""
    if current_states.action.val is None:
        return
    action_val = current_states.action.val.action
    action_dict: dict[str, Any] = {"action_id": int(action_val)}
    _enrich_action_from_gym(cfg, action_val, action_dict)
    _enrich_action_expmod(action_dict)
    _emit_state_record("roc.action", json.dumps(action_dict, separators=(",", ":")))


def _enrich_action_from_gym(cfg: Any, action_val: Any, action_dict: dict[str, Any]) -> None:
    """Add gym action name and key to the action dict if available."""
    try:
        gym_actions = cfg.gym_actions
        if not gym_actions or int(action_val) >= len(gym_actions):
            return
        act_enum = gym_actions[int(action_val)]
        action_dict["action_name"] = str(getattr(act_enum, "name", act_enum))
        _val = getattr(act_enum, "value", None)
        if not isinstance(_val, int):
            return
        from roc.gymnasium import action_value_to_key

        _key = action_value_to_key(_val)
        if _key is not None:
            action_dict["action_key"] = _key
    except Exception:
        pass


def _enrich_action_expmod(action_dict: dict[str, Any]) -> None:
    """Add expmod name to the action dict if available."""
    try:
        from roc.action import DefaultActionExpMod

        action_dict["expmod_name"] = DefaultActionExpMod.get(default="pass").name
    except Exception:
        pass


def _emit_transform_log(current_states: "StateList") -> None:
    """Emit transform summary as an OTel log record."""
    if current_states.transform.val is None:
        return
    t = current_states.transform.val.transform
    changes: list[dict[str, Any]] = []
    for edge in t.src_edges:
        dst = edge.dst
        change_dict: dict[str, Any] = {"description": str(dst)}
        if hasattr(dst, "name"):
            change_dict["type"] = type(dst).__name__
            change_dict["name"] = dst.name
        if hasattr(dst, "normalized_change"):
            change_dict["normalized_change"] = dst.normalized_change
        changes.append(change_dict)
    _emit_state_record(
        "roc.transform_summary",
        json.dumps(
            {"count": len(changes), "changes": changes},
            separators=(",", ":"),
            default=str,
        ),
    )


def _emit_sequence_log(current_states: "StateList") -> None:
    """Emit sequence summary as an OTel log record."""
    try:
        from roc.component import (
            ComponentName as _CN,
            ComponentType as _CT,
            loaded_components as _lc,
        )
        from roc.sequencer import Sequencer as _Seq

        seq_comp = _lc.get((_CN("sequencer"), _CT("sequencer")))
        if not isinstance(seq_comp, _Seq) or seq_comp.last_frame is None:
            return
        frame = seq_comp.last_frame
        seq_dict = _build_sequence_dict(frame, current_states)
        _emit_state_record(
            "roc.sequence_summary",
            json.dumps(seq_dict, separators=(",", ":"), default=str),
        )
    except Exception:
        pass


def _build_sequence_dict(frame: Any, current_states: "StateList") -> dict[str, Any]:
    """Build the sequence summary dict from a frame."""
    from roc.intrinsic import IntrinsicNode

    objs = frame.objects
    obj_dicts: list[dict[str, Any]] = []
    for obj in objs:
        obj_dict: dict[str, Any] = {"id": str(obj.id)[:8]}
        if hasattr(obj, "last_x") and hasattr(obj, "last_y"):
            obj_dict["x"] = obj.last_x
            obj_dict["y"] = obj.last_y
        if hasattr(obj, "resolve_count"):
            obj_dict["resolve_count"] = obj.resolve_count
        obj_dicts.append(obj_dict)

    intrinsic_snapshot: dict[str, float] = {}
    for t_node in frame.transformable:
        if isinstance(t_node, IntrinsicNode):
            intrinsic_snapshot[t_node.name] = round(t_node.normalized_value, 4)

    seq_dict: dict[str, Any] = {
        "tick": frame.tick,
        "object_count": len(objs),
        "objects": obj_dicts,
        "intrinsic_count": len(intrinsic_snapshot),
        "intrinsics": intrinsic_snapshot,
    }
    if current_states.significance.val is not None:
        seq_dict["significance"] = current_states.significance.val.significance
    return seq_dict


def _emit_prediction_log(current_states: "StateList") -> None:
    """Emit prediction state as an OTel log record."""
    if current_states.predict.val is None:
        return
    pred = current_states.predict.val
    pred_dict: dict[str, Any] = {"made": not isinstance(pred, NoPrediction)}
    _enrich_prediction_meta(pred_dict)
    _emit_state_record(
        "roc.prediction",
        json.dumps(pred_dict, separators=(",", ":"), default=str),
    )


def _enrich_prediction_meta(pred_dict: dict[str, Any]) -> None:
    """Add prediction metadata from expmod and component state."""
    try:
        from roc.predict import (
            Predict,
            PredictionCandidateFramesExpMod,
            PredictionConfidenceExpMod,
        )

        pred_dict["candidate_expmod"] = PredictionCandidateFramesExpMod.get(
            default="object-based"
        ).name
        pred_dict["confidence_expmod"] = PredictionConfidenceExpMod.get(default="naive").name

        from roc.component import ComponentName, ComponentType, loaded_components

        predict_comp = loaded_components.get((ComponentName("predict"), ComponentType("predict")))
        if not isinstance(predict_comp, Predict):
            return
        meta = predict_comp.last_prediction_meta
        pred_dict["candidate_count"] = meta.candidate_count
        pred_dict["confidence"] = meta.confidence
        pred_dict["all_scores"] = meta.all_scores
        if meta.predicted_intrinsics:
            pred_dict["predicted_intrinsics"] = meta.predicted_intrinsics
    except Exception:
        pass


def _emit_message_log(current_states: "StateList") -> None:
    """Emit message state as an OTel log record."""
    if current_states.message.val is None:
        return
    msg = current_states.message.val.strip()
    if msg:
        _emit_state_record("roc.message", msg)


def _emit_phonemes_log(current_states: "StateList") -> None:
    """Emit phonemes state as an OTel log record."""
    if current_states.phonemes.val is None:
        return
    phoneme_dicts = [
        {"word": pw.word, "phonemes": pw.phonemes, "is_break": pw.is_break}
        for pw in current_states.phonemes.val
    ]
    _emit_state_record(
        "roc.phonemes",
        json.dumps(phoneme_dicts, separators=(",", ":")),
    )


def _emit_graphdb_summary_log() -> None:
    """Emit graph DB summary as an OTel log record."""
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


def _emit_event_summary_log() -> None:
    """Emit event bus activity summary as an OTel log record."""
    step_counts = Event.get_step_counts()
    if step_counts:
        _emit_state_record(
            "roc.event.summary",
            json.dumps(step_counts, separators=(",", ":")),
        )


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
