# pragma: no cover

"""This module is a wrapper around the Gym / Gymnasium interfaces and drives all
the interactions between the agent and the system, including the main event loop.
"""

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]
import gymnasium as gym
import networkx as nx
import nle
from pydantic import BaseModel, Field


def action_value_to_key(val: int) -> str | None:
    """Convert an NLE action enum value to a human-readable key string.

    NLE encodes actions in three ranges:
    - Printable ASCII (32-126): single keystroke, e.g. ord("a") -> "a"
    - Control chars (1-31): C(c) = 0x1F & c, e.g. C("d") = 4 -> "^d"
    - Meta/extended (128+): M(c) = 0x80 | c, e.g. M("f") = 230 -> "M-f"

    Returns None for values that don't map to a key (0, 127, etc).
    """
    if 32 <= val <= 126:
        return chr(val)
    if 1 <= val <= 31:
        return f"^{chr(val + 0x40)}"
    if val >= 128:
        base = val & 0x7F
        if 32 <= base <= 126:
            return f"M-{chr(base)}"
    return None


from ..pipeline.action import Action, ActionRequest, TakeAction
from .breakpoint import breakpoints
from ..framework.component import Component
from ..framework.config import Config
from ..db.graphdb import GraphDB
from ..pipeline.intrinsic import Intrinsic, IntrinsicData
from ..framework import logger as roc_logger
from ..framework.logger import logger
from ..perception.base import AuditoryData, Perception, ProprioceptiveData, VisionData
from ..reporting.metrics import RocMetrics
from ..reporting.observability import Observability
from ..reporting.screen_renderer import screen_to_html_vals
from ..reporting.state import State, _emit_state_record
from ..reporting.step_buffer import StepBuffer

# Cumulative glyph sets for attention spread tracking.
_attended_glyphs: set[int] = set()
_seen_glyphs: set[int] = set()


@dataclass
class GameLoopContext:
    """Invariant state for the game observation loop."""

    observation_counter: Any


class Gym(Component, ABC):
    """A wrapper around an OpenAI Gym / Farama Gymnasium that drives the event
    loop and interfaces to the ROC agent.
    """

    name: str = "gym"
    type: str = "game"

    def __init__(self, gym_id: str, *, gym_opts: dict[str, Any] | None = None) -> None:
        logger.debug("Initializing Gym...")
        super().__init__()
        gym_opts = gym_opts or {}
        logger.debug(f"Gym options: {gym_opts}")
        self.env = gym.make(gym_id, **gym_opts)

        # setup communications
        self.env_bus_conn = Perception.bus.connect(self)
        self.action_bus_conn = Action.bus.connect(self)
        self.intrinsic_bus_conn = Intrinsic.bus.connect(self)

        # config
        self.config(self.env)

        # TODO: config environment
        # setup which features detectors to use on each bus

    @abstractmethod
    def send_obs(self, obs: Any) -> None: ...

    @abstractmethod
    def config(self, env: gym.core.Env[Any, Any]) -> None: ...

    @abstractmethod
    def get_action(self) -> Any: ...

    @logger.catch
    @Observability.tracer.start_as_current_span("start")
    def start(
        self,
        stop_event: threading.Event | None = None,
        step_buffer: StepBuffer | None = None,
    ) -> None:
        from roc.framework.clock import Clock

        logger.debug("Starting NLE loop...")
        # Reset the tick clock so each run begins at 0 regardless of
        # whatever state prior tests / server lifetimes may have left.
        Clock.reset()
        obs, _reset_info = self.env.reset()
        settings = Config.get()

        _publish_action_map(settings.gym_actions)

        done = False
        truncated = False
        _dump_env_start()

        loop_num = 0
        game_num = 1
        game_counter = Observability.meter.create_counter(
            "roc.game_total", unit="games", description="total number of games completed"
        )
        observation_counter = Observability.meter.create_counter(
            "roc.obs_total", unit="observations", description="total number of observations"
        )
        game_counter.add(1)
        _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')
        loop_ctx = GameLoopContext(observation_counter)

        try:
            # main environment loop
            while game_num <= settings.num_games:
                if stop_event is not None and stop_event.is_set():
                    logger.info("Stop event set, exiting game loop.")
                    break

                with Observability.tracer.start_as_current_span("observation"):
                    obs, done, truncated, loop_num = self._run_observation_step(
                        obs, loop_num, game_num, loop_ctx, step_buffer
                    )

                    if done or truncated:
                        _handle_game_over(obs, game_num, done, settings)

                    if done or truncated:
                        self.env.reset()
                        game_counter.add(1)
                        game_num += 1
                        _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')

                    if stop_event is not None and stop_event.is_set():
                        logger.info("Stop event set, exiting game loop.")
                        break
        finally:
            # Always write the graph archive on exit, even when the user
            # stops mid-game or an exception breaks the loop. Without this,
            # `_handle_game_over` is the only caller, which runs only on
            # natural game-end (done/truncated), and historical runs stopped
            # via the REST API would have no graph.json -- making the
            # dashboard's Graph Visualization panel show "No graph data".
            try:
                _export_graph_archive()
            except Exception:
                logger.exception("Failed to export graph archive during cleanup")

            logger.info("NLE loop done, exiting.")
            from roc.reporting.api_server import stop_dashboard

            stop_dashboard()
            Observability.shutdown()
            _dump_env_end()

    def _run_observation_step(
        self,
        obs: Any,
        loop_num: int,
        game_num: int,
        loop_ctx: GameLoopContext,
        step_buffer: StepBuffer | None = None,
    ) -> tuple[Any, bool, bool, int]:
        """Execute a single observation-action-step cycle.

        Returns the new (obs, done, truncated, loop_num) tuple.
        """
        from roc.framework.clock import Clock
        from roc.reporting.step_log_sink import set_current_step

        # Clock advances at the top of each observation cycle. Everything
        # downstream -- Sequencer Frame.tick, ObjectInstance.tick,
        # ResolutionContext.tick, ParquetExporter step stamp, log records
        # -- reads from Clock.get(), so this is the single source of truth
        # for "which observation cycle is this?".
        Clock.set(loop_num + 1)
        set_current_step(loop_num + 1)

        logger.trace(f"Sending observation: {obs}")
        breakpoints.check()

        # Copy numpy arrays -- NLE reuses internal buffers
        State.get_states().screen.set(
            {
                "chars": obs["tty_chars"].copy(),
                "colors": obs["tty_colors"].copy(),
                "cursor": obs["tty_cursor"],
            }
        )

        self.send_obs(obs)
        action = self.get_action()
        obs, _reward, done, truncated, _info = self.env.step(action)
        _dump_env_record(obs, loop_num)

        # Update screen state with post-step obs so death/end screens are captured
        if done or truncated:
            State.get_states().screen.set(
                {
                    "chars": obs["tty_chars"].copy(),
                    "colors": obs["tty_colors"].copy(),
                    "cursor": obs["tty_cursor"],
                }
            )

        loop_ctx.observation_counter.add(1)
        loop_num += 1
        State.get_states().loop.set(loop_num)
        State.maybe_emit_snapshot(loop_num)
        if Config.get().emit_state:
            State.emit_state_logs()

        _push_dashboard_data(obs, loop_num, game_num, step_buffer)

        # Reset per-step cycle accumulators AFTER dashboard data has been read
        cycle_states = State.get_states()
        cycle_states.saliency_cycles.reset()
        cycle_states.resolution_cycles.reset()
        cycle_states.attenuation_cycles.reset()

        return obs, done, truncated, loop_num


def _build_action_map(
    gym_actions: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Build the full action map from gym action enums."""
    action_map: list[dict[str, Any]] = []
    for idx, act in enumerate(gym_actions):
        entry: dict[str, Any] = {"action_id": idx}
        entry["action_name"] = str(getattr(act, "name", act))
        aval = getattr(act, "value", None)
        if isinstance(aval, int):
            key = action_value_to_key(aval)
            if key is not None:
                entry["action_key"] = key
        action_map.append(entry)
    return action_map


def _publish_action_map(
    gym_actions: tuple[Any, ...] | None,
) -> None:
    """Build and save the action map to disk."""
    if not gym_actions:
        return

    action_map = _build_action_map(gym_actions)
    _save_action_map_to_file(action_map)


def _save_action_map_to_file(action_map: list[dict[str, Any]]) -> None:
    """Save the action map directly to the run directory."""
    from roc.reporting.observability import Observability

    dl_store = Observability.get_ducklake_store()
    if dl_store is None:
        return
    try:
        import json

        map_path = dl_store.run_dir / "action_map.json"
        map_path.write_text(json.dumps(action_map))
        logger.debug("Saved action_map.json ({} entries)", len(action_map))
    except Exception:
        logger.opt(exception=True).debug("Failed to save action_map.json")


def _collect_screen_data(states: Any) -> tuple[Any, Any, Any]:
    """Collect screen, saliency, and feature data from states."""
    screen_state = states.screen.val
    screen_vals = screen_to_html_vals(screen_state) if screen_state is not None else None
    saliency_state = states.salency.val
    saliency_vals = saliency_state.to_html_vals() if saliency_state is not None else None
    features = None
    if saliency_state is not None:
        feat_report = saliency_state.feature_report()
        features = [feat_report] if feat_report else None
    return screen_vals, saliency_vals, features


def _collect_object_data(states: Any) -> tuple[Any, Any]:
    """Collect object info and focus points from states."""
    object_info = None
    if states.object.val is not None:
        object_info = [{"raw": str(states.object)}]
    focus_points = None
    if states.attention.val is not None:
        focus_points = [{"raw": str(states.attention.val.focus_points)}]
    return object_info, focus_points


def _collect_graph_summary() -> dict[str, Any]:
    """Collect graph database cache summary."""
    from roc.db.graphdb import Edge, Node

    node_cache = Node.get_cache()
    edge_cache = Edge.get_cache()
    return {
        "node_count": node_cache.currsize,
        "node_max": node_cache.maxsize,
        "edge_count": edge_cache.currsize,
        "edge_max": edge_cache.maxsize,
    }


def _collect_event_summary() -> list[dict[str, Any]] | None:
    """Collect event bus step counts."""
    from roc.framework.event import Event

    step_counts = Event.get_step_counts()
    return [step_counts] if step_counts else None


def _build_action_taken_dict(states: Any) -> dict[str, Any] | None:
    """Build the action_taken dict from current state."""
    if states.action.val is None:
        return None
    act_id = int(states.action.val.action)
    result: dict[str, Any] = {"action_id": act_id}
    try:
        gym_actions = Config.get().gym_actions
        if gym_actions and act_id < len(gym_actions):
            act_enum = gym_actions[act_id]
            result["action_name"] = str(getattr(act_enum, "name", act_enum))
            val = getattr(act_enum, "value", None)
            if isinstance(val, int):
                key = action_value_to_key(val)
                if key is not None:
                    result["action_key"] = key
    except Exception:
        pass
    return result


def _shape_type_to_glyph(shape_type: int) -> str:
    """Convert a numeric shape_type to a printable glyph character."""
    return chr(shape_type) if 32 <= shape_type < 127 else str(shape_type)


def _build_oi_lookup(frame: Any) -> tuple[dict[int, Any], dict[int, str]]:
    """Build ObjectInstance lookup dicts from a frame's edges.

    Returns (oi_by_uuid, glyph_by_uuid) maps keyed by object_uuid.
    """
    from roc.pipeline.object.object_instance import ObjectInstance, SituatedObjectInstance

    oi_by_uuid: dict[int, Any] = {}
    glyph_by_uuid: dict[int, str] = {}
    for e in frame.src_edges:
        if not (isinstance(e, SituatedObjectInstance) and isinstance(e.dst, ObjectInstance)):
            continue
        oi = e.dst
        oi_by_uuid[oi.object_uuid] = oi
        if oi.shape_type is not None:
            glyph_by_uuid[oi.object_uuid] = _shape_type_to_glyph(oi.shape_type)
    return oi_by_uuid, glyph_by_uuid


def _reconstruct_coord_from_delta(
    change_entry: dict[str, Any], prop_name: str, prop_node: Any, matched_oi: Any
) -> None:
    """Infer old/new coordinate values from current position and delta."""
    cur_val = getattr(matched_oi, prop_name, None)
    delta = getattr(prop_node, "delta", None)
    if cur_val is not None:
        change_entry["new_value"] = cur_val
        if delta is not None:
            change_entry["old_value"] = int(cur_val - delta)


def _apply_coord_values(
    change_entry: dict[str, Any], prop_name: str, prop_node: Any, matched_oi: Any
) -> None:
    """Fill old_value/new_value in change_entry for a property transform node.

    For x/y props where old_value is absent, reconstructs values from the
    current position and delta. Uses raw old/new values otherwise.
    """
    ov = getattr(prop_node, "old_value", None)
    nv = getattr(prop_node, "new_value", None)
    if prop_name in ("x", "y") and ov is None and matched_oi is not None:
        _reconstruct_coord_from_delta(change_entry, prop_name, prop_node, matched_oi)
    else:
        if ov is not None:
            change_entry["old_value"] = ov
        if nv is not None:
            change_entry["new_value"] = nv


def _process_property_node(prop_node: Any, matched_oi: Any) -> dict[str, Any] | None:
    """Build a change entry dict for a single property transform node.

    Returns None if the node has no property_name attribute.
    """
    prop_name = getattr(prop_node, "property_name", None)
    if prop_name is None:
        return None
    change_entry: dict[str, Any] = {
        "property": prop_name,
        "delta": getattr(prop_node, "delta", None),
    }
    ct = getattr(prop_node, "change_type", None)
    if ct is not None:
        change_entry["type"] = ct
    _apply_coord_values(change_entry, prop_name, prop_node, matched_oi)
    return change_entry


def _process_object_transform(
    dst: Any,
    oi_by_uuid: dict[int, Any],
    glyph_by_uuid: dict[int, str],
) -> dict[str, Any]:
    """Build a summary dict for a single ObjectTransform node."""
    matched_oi = oi_by_uuid.get(dst.object_uuid)
    ot_dict: dict[str, Any] = {"uuid": dst.object_uuid}
    glyph = glyph_by_uuid.get(dst.object_uuid)
    if glyph is not None:
        ot_dict["glyph"] = glyph
    if matched_oi is not None:
        ot_dict["node_id"] = matched_oi.id
        if matched_oi.color_type is not None:
            ot_dict["color"] = str(matched_oi.color_type)
    ot_changes: list[dict[str, Any]] = []
    for detail_edge in dst.src_edges:
        entry = _process_property_node(detail_edge.dst, matched_oi)
        if entry is not None:
            ot_changes.append(entry)
    ot_dict["changes"] = ot_changes
    return ot_dict


def _process_intrinsic_change(dst: Any) -> dict[str, Any] | None:
    """Build a change dict for an intrinsic node, or None if not applicable."""
    if not hasattr(dst, "name") or not hasattr(dst, "normalized_change"):
        return None
    return {
        "description": str(dst),
        "type": type(dst).__name__,
        "name": dst.name,
        "normalized_change": dst.normalized_change,
    }


def _build_transform_summary(states: Any) -> dict[str, Any] | None:
    """Build transform summary from the current transform state."""
    if states.transform.val is None:
        return None

    frame = _get_last_frame()
    if frame is not None:
        oi_by_uuid, glyph_by_uuid = _build_oi_lookup(frame)
    else:
        oi_by_uuid, glyph_by_uuid = {}, {}

    from roc.pipeline.object.object_transform import ObjectTransform as _OT

    t = states.transform.val.transform
    changes_list: list[dict[str, Any]] = []
    object_transforms: list[dict[str, Any]] = []
    for edge in t.src_edges:
        dst = edge.dst
        if isinstance(dst, _OT):
            object_transforms.append(_process_object_transform(dst, oi_by_uuid, glyph_by_uuid))
            continue
        ch = _process_intrinsic_change(dst)
        if ch is not None:
            changes_list.append(ch)
    summary: dict[str, Any] = {"count": len(changes_list) + len(object_transforms)}
    summary["changes"] = changes_list
    if object_transforms:
        summary["object_transforms"] = object_transforms
    return summary


def _update_attended_glyphs(att: dict[str, Any], glyphs: Any, bg_glyph: int) -> None:
    """Update the cumulative attended glyphs set from the top focus point."""
    focus_points = att.get("focus_points", [])
    if not focus_points:
        return
    top = focus_points[0]
    fx, fy = int(top["x"]), int(top["y"])
    if 0 <= fy < glyphs.shape[0] and 0 <= fx < glyphs.shape[1]:
        g_id = int(glyphs[fy, fx])
        if g_id != bg_glyph:
            _attended_glyphs.add(g_id)


def _inject_attention_spread(states: Any, obs: Any) -> None:
    """Add attention spread metrics to the attenuation data dict.

    Spread tracks how many unique glyph types the attention system has examined
    vs how many exist on screen. Both are cumulative sets of NLE glyph IDs.
    """
    import numpy as np

    att = states.attenuation_data.val
    if att is None or not isinstance(att, dict):
        return

    bg_glyph = nle.nethack.GLYPH_CMAP_OFF
    glyphs = obs["glyphs"]

    # Update cumulative seen glyphs (all non-background glyphs on screen)
    for g in np.unique(glyphs):
        g_int = int(g)
        if g_int != bg_glyph:
            _seen_glyphs.add(g_int)

    # Update cumulative attended glyphs (glyph at top focus point)
    _update_attended_glyphs(att, glyphs, bg_glyph)

    attended = len(_attended_glyphs)
    total = len(_seen_glyphs)
    pct = round(attended / total * 100, 1) if total > 0 else 0.0

    att["spread_attended"] = attended
    att["spread_total"] = total
    att["spread_pct"] = pct


def _build_sequence_summary(states: Any) -> dict[str, Any] | None:
    """Build sequence summary from the sequencer's last frame."""
    try:
        if states.transform.val is None:
            return None
        frame = _get_last_frame()
        if frame is None:
            return None
        return _frame_to_summary(frame, states)
    except Exception:
        return None


def _get_last_frame() -> Any:
    """Get the sequencer's last frame, or None if unavailable."""
    from roc.framework.component import ComponentName, ComponentType, loaded_components
    from roc.pipeline.temporal.sequencer import Sequencer

    seq_comp = loaded_components.get((ComponentName("sequencer"), ComponentType("sequencer")))
    if not isinstance(seq_comp, Sequencer) or seq_comp.last_frame is None:
        return None
    return seq_comp.last_frame


def _apply_oi_fields(od: dict[str, Any], obj: Any, oi: Any, matched_uuids: set[int] | None) -> None:
    """Populate fields from ObjectInstance when available."""
    od["x"] = oi.x
    od["y"] = oi.y
    if oi.shape_type is not None:
        od["glyph"] = _shape_type_to_glyph(oi.shape_type)
        od["shape"] = oi.shape_type
    if oi.color_type is not None:
        od["color"] = oi.color_type
    obj_uuid = getattr(obj, "uuid", None)
    od["matched_previous"] = bool(matched_uuids and obj_uuid in matched_uuids)


def _obj_to_dict(
    obj: Any,
    oi: Any | None = None,
    matched_uuids: set[int] | None = None,
) -> dict[str, Any]:
    """Convert a single object to a summary dict."""
    od: dict[str, Any] = {"id": str(obj.id)[:8]}
    if oi is not None:
        _apply_oi_fields(od, obj, oi, matched_uuids)
    elif hasattr(obj, "last_x") and hasattr(obj, "last_y"):
        od["x"] = obj.last_x
        od["y"] = obj.last_y
    if hasattr(obj, "resolve_count"):
        od["resolve_count"] = obj.resolve_count
    return od


def _collect_matched_uuids(states: Any) -> set[int]:
    """Collect the set of object uuids that appear in the current transform."""
    matched_uuids: set[int] = set()
    if states.transform.val is None:
        return matched_uuids
    for edge in states.transform.val.transform.src_edges:
        if hasattr(edge.dst, "object_uuid"):
            matched_uuids.add(edge.dst.object_uuid)
    return matched_uuids


def _frame_to_summary(frame: Any, states: Any) -> dict[str, Any]:
    """Build the summary dict from a frame and current states."""
    from roc.pipeline.intrinsic import IntrinsicNode

    objs = frame.objects
    oi_by_uuid, _ = _build_oi_lookup(frame)
    matched_uuids = _collect_matched_uuids(states)

    obj_dicts = []
    for obj in objs:
        obj_uuid: int | None = getattr(obj, "uuid", None)
        oi = oi_by_uuid.get(obj_uuid) if obj_uuid is not None else None
        obj_dicts.append(_obj_to_dict(obj, oi, matched_uuids))
    intr_snap = {
        tn.name: round(tn.normalized_value, 4)
        for tn in frame.transformable
        if isinstance(tn, IntrinsicNode)
    }
    summary: dict[str, Any] = {
        "tick": frame.tick,
        "object_count": len(objs),
        "objects": obj_dicts,
        "intrinsic_count": len(intr_snap),
        "intrinsics": intr_snap,
    }
    if states.significance.val is not None:
        summary["significance"] = states.significance.val.significance
    return summary


def _build_prediction_data(states: Any) -> dict[str, Any] | None:
    """Build prediction data from the current predict state."""
    if states.predict.val is None:
        return None
    from roc.pipeline.temporal.predict import NoPrediction

    prediction: dict[str, Any] = {"made": not isinstance(states.predict.val, NoPrediction)}
    try:
        from roc.framework.component import (
            ComponentName,
            ComponentType,
            loaded_components,
        )
        from roc.pipeline.temporal.predict import Predict

        pred_comp = loaded_components.get((ComponentName("predict"), ComponentType("predict")))
        if isinstance(pred_comp, Predict):
            meta = pred_comp.last_prediction_meta
            prediction["candidate_count"] = meta.candidate_count
            prediction["confidence"] = meta.confidence
            prediction["all_scores"] = meta.all_scores
            if meta.predicted_intrinsics:
                prediction["predicted_intrinsics"] = meta.predicted_intrinsics
    except Exception:
        pass
    return prediction


def _collect_message(states: Any) -> str | None:
    """Extract a non-empty message string from states."""
    if states.message.val is None:
        return None
    msg = states.message.val.strip()
    return msg if msg else None


def _collect_phonemes(states: Any) -> list[dict[str, Any]] | None:
    """Collect phoneme data from states."""
    if states.phonemes.val is None:
        return None
    return [
        {"word": pw.word, "phonemes": pw.phonemes, "is_break": pw.is_break}
        for pw in states.phonemes.val
    ]


def _parse_inventory(obs: Any) -> list[dict[str, Any]] | None:
    """Parse inventory items from the observation dict."""
    try:
        inv_strs = obs["inv_strs"]
        inv_letters = obs["inv_letters"]
        inv_glyphs = obs["inv_glyphs"]
        inv_items: list[dict[str, Any]] = []
        for i in range(len(inv_strs)):
            item_str = "".join(chr(ch) for ch in inv_strs[i]).strip("\x00 ")
            glyph = int(inv_glyphs[i])
            if not item_str or glyph == 5976:
                continue
            inv_items.append(
                {
                    "letter": chr(int(inv_letters[i])),
                    "item": item_str,
                    "glyph": glyph,
                }
            )
        return inv_items if inv_items else None
    except Exception:
        return None


def _extract_game_metrics(obs: Any) -> dict[str, int]:
    """Extract bottom-line stats into a game metrics dict."""
    blstats = obs["blstats"]
    return {
        "score": int(blstats[blstat_offsets.SCORE]),
        "hp": int(blstats[blstat_offsets.HP]),
        "hp_max": int(blstats[blstat_offsets.HPMAX]),
        "energy": int(blstats[blstat_offsets.ENE]),
        "energy_max": int(blstats[blstat_offsets.ENEMAX]),
        "depth": int(blstats[blstat_offsets.DEPTH]),
        "gold": int(blstats[blstat_offsets.GOLD]),
        "x": int(blstats[blstat_offsets.X]),
        "y": int(blstats[blstat_offsets.Y]),
        "hunger": int(blstats[blstat_offsets.HUNGER]),
        "xp_level": int(blstats[blstat_offsets.XP]),
        "experience": int(blstats[blstat_offsets.EXP]),
        "ac": int(blstats[blstat_offsets.AC]),
    }


def _build_intrinsics_dict(states: Any) -> dict[str, Any] | None:
    """Build intrinsics payload dict from state, or None if unavailable."""
    if states.intrinsic.val is None:
        return None
    return {
        "raw": states.intrinsic.val.intrinsics,
        "normalized": states.intrinsic.val.normalized_intrinsics,
    }


def _build_significance_val(states: Any) -> Any:
    """Return the significance value from state, or None if unavailable."""
    if states.significance.val is None:
        return None
    return states.significance.val.significance


def _push_dashboard_data(
    obs: Any,
    loop_num: int,
    game_num: int,
    step_buffer: StepBuffer | None = None,
) -> None:
    """Collect step data and push directly to the StepBuffer.

    If step_buffer is provided (thread mode), pushes to it directly.
    Otherwise falls back to the globally registered StepBuffer (standalone mode).
    """
    import json as _json

    game_metrics = _extract_game_metrics(obs)
    RocMetrics.log_step(game_metrics)
    _emit_state_record(
        "roc.game_metrics",
        _json.dumps(game_metrics, separators=(",", ":")),
    )

    from roc.reporting.step_buffer import get_step_buffer

    buf = step_buffer or get_step_buffer()
    if buf is None:
        return

    inventory = _parse_inventory(obs)

    if inventory is not None:
        _emit_state_record(
            "roc.inventory",
            _json.dumps(inventory, separators=(",", ":")),
        )

    from time import time_ns

    from roc.reporting.run_store import StepData
    from roc.reporting.step_log_sink import drain_step_logs

    states = State.get_states()
    screen_vals, saliency_vals, features = _collect_screen_data(states)
    object_info, focus_points = _collect_object_data(states)

    # Compute attention spread: focus peaks vs unique glyphs on screen
    _inject_attention_spread(states, obs)

    step_data = StepData(
        step=loop_num,
        game_number=game_num,
        timestamp=time_ns(),
        screen=screen_vals,
        saliency=saliency_vals,
        features=features,
        object_info=object_info,
        focus_points=focus_points,
        attenuation=states.attenuation_data.val,
        resolution_metrics=states.resolution.val,
        graph_summary=_collect_graph_summary(),
        event_summary=_collect_event_summary(),
        game_metrics=game_metrics,
        logs=drain_step_logs(loop_num),
        intrinsics=_build_intrinsics_dict(states),
        significance=_build_significance_val(states),
        action_taken=_build_action_taken_dict(states),
        sequence_summary=_build_sequence_summary(states),
        transform_summary=_build_transform_summary(states),
        prediction=_build_prediction_data(states),
        message=_collect_message(states),
        phonemes=_collect_phonemes(states),
        inventory=inventory,
        saliency_cycles=states.saliency_cycles.val or None,
        resolution_cycles=states.resolution_cycles.val or None,
    )
    buf.push(step_data)


def _tty_chars_to_screen(tty_chars: Any) -> str:
    """Convert a 2D array of ASCII char codes to a printable screen string."""
    return "".join("".join(chr(ch) for ch in row) + "\n" for row in tty_chars)


def _export_graph_archive() -> None:
    """Export the full graph as node-link JSON to the run directory.

    Uses GraphDB.to_networkx() to get the full graph, then writes
    nx.node_link_data() to run_dir/graph.json. Skips silently if no
    DuckLakeStore is configured (no run directory available).
    """
    store = Observability.get_ducklake_store()
    if store is None:
        return

    G = GraphDB.to_networkx()
    data = nx.node_link_data(G, edges="links")
    graph_path = store.run_dir / "graph.json"
    with open(graph_path, "w") as f:
        json.dump(data, f, default=str)
    logger.info(f"Graph archive written to {graph_path}")


def _handle_game_over(obs: Any, game_num: int, done: bool, settings: Config) -> None:
    """Log game over info and optionally flush the graph database."""
    screen = _tty_chars_to_screen(obs["tty_chars"])
    logger.info(screen, death=True, game_num=game_num)
    logger.info(f"Game {game_num} completed, starting next game")
    if settings.graphdb_flush:
        GraphDB.flush()
    if settings.graphdb_export:
        GraphDB.export()
    _export_graph_archive()
    blstats = obs["blstats"]
    score = int(blstats[blstat_offsets.SCORE])
    outcome = "done" if done else "truncated"
    _emit_state_record(
        "roc.game_end",
        f'{{"game_number": {game_num}, "outcome": "{outcome}", "score": {score}}}',
    )


class blstat_offsets(IntEnum):
    """An enumeration of Nethack bottom line statistics (intelligence, strength,
    charisma, position, hit points, etc.)
    """

    # fmt: off
    X =         0
    Y =         1
    STR25 =     2
    STR125 =    3
    DEX =       4
    CON =       5
    INT =       6
    WIS =       7
    CHA =       8
    SCORE =     9
    HP =        10
    HPMAX =     11
    DEPTH =     12
    GOLD =      13
    ENE =       14
    ENEMAX =    15
    AC =        16
    HD =        17
    XP =        18
    EXP =       19
    TIME =      20
    HUNGER =    21
    CAP =       22
    DNUM =      23
    DLEVEL =    24
    CONDITION = 25
    ALIGN =     26
    # fmt: on


class condition_bits(IntEnum):
    """Bits for decoding the `CONDITION` bottomline stat to determin if the
    player is flying, deaf, food poisoned, etc.
    """

    # fmt: off
    STONE =    nle.nethack.BL_MASK_STONE
    SLIME =    nle.nethack.BL_MASK_SLIME
    STRINGL =  nle.nethack.BL_MASK_STRNGL
    FOODPOIS = nle.nethack.BL_MASK_FOODPOIS
    TERMILL =  nle.nethack.BL_MASK_TERMILL
    BLIND =    nle.nethack.BL_MASK_BLIND
    DEAF =     nle.nethack.BL_MASK_DEAF
    STUN =     nle.nethack.BL_MASK_STUN
    CONF =     nle.nethack.BL_MASK_CONF
    HALLU =    nle.nethack.BL_MASK_HALLU
    LEV =      nle.nethack.BL_MASK_LEV
    FLY =      nle.nethack.BL_MASK_FLY
    RIDE =     nle.nethack.BL_MASK_RIDE
    # fmt: on


class BottomlineStats(BaseModel):
    """A Pydantic model representing the Nethack bottom line statistics."""

    x: int
    y: int
    str25: int
    str125: int
    dex: int
    con: int
    intel: int = Field(alias="int")
    wis: int
    cha: int
    score: int
    hp: int
    hpmax: int
    depth: int
    gold: int
    ene: int
    enemax: int
    ac: int
    hd: int
    xp: int
    exp: int
    time: int
    hunger: int
    cap: int
    dnum: int
    dlevel: int
    condition: int
    align: int
    stone: bool
    slime: bool
    stringl: bool
    foodpois: bool
    termill: bool
    blind: bool
    deaf: bool
    stun: bool
    conf: bool
    hallu: bool
    lev: bool
    fly: bool
    ride: bool


class NethackGym(Gym):
    """Wrapper around the Gym class for driving the Nethack interface to the ROC
    agent. Decodes Nethack specific data and sends it to the agent as Events.
    """

    def __init__(self, *, gym_opts: dict[str, Any] | None = None) -> None:
        gym_opts = gym_opts or {}
        settings = Config.get()
        gym_opts["options"] = list(nle.nethack.NETHACKOPTIONS) + settings.nethack_extra_options
        gym_opts["max_episode_steps"] = settings.nethack_max_turns
        # XXX: note that 'gym_opts["character"]' sets the character type, not
        # the player name... player name is forced to be "Agent" by NLE

        # XXX: env name options include: "NetHack", "NetHackScore", "NetHackStaircase", "NetHackStaircasePet", "NetHackOracle", "NetHackGold", "NetHackEat", "NetHackScout", "NetHackChallenge"
        # see: https://github.com/heiner/nle/blob/731f2aaa94f6d67838228f9c9b5b04faa31cb862/nle/env/__init__.py#L9
        # and: https://github.com/heiner/nle/blob/731f2aaa94f6d67838228f9c9b5b04faa31cb862/nle/env/tasks.py
        # "NetHack" is the vanilla environment
        # "NetHackScore" and "NetHackChallenge" also appear to be interesting
        super().__init__("NetHack-v0", gym_opts=gym_opts)

    def config(self, env: gym.core.Env[Any, Any]) -> None:
        settings = Config.get()
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_count = int(self.env.action_space.n)

        settings.gym_actions = tuple(self.env.unwrapped.actions)  # type: ignore
        settings.observation_shape = nle.nethack.DUNGEON_SHAPE

    def send_obs(self, obs: Any) -> None:
        self.send_vision(obs)
        self.send_intrinsics(obs)
        self.send_auditory(obs)
        self.send_proprioceptive(obs)

    def get_action(self) -> Any:
        self.action_bus_conn.send(ActionRequest())

        # get result using cache
        assert self.action_bus_conn.attached_bus.cache is not None
        cache = self.action_bus_conn.attached_bus.cache
        a = list(filter(lambda e: isinstance(e.data, TakeAction), cache))[-1]
        assert isinstance(a.data, TakeAction)

        return a.data.action

    def send_vision(self, obs: Any) -> None:
        vd = VisionData.from_dict(obs)
        self.env_bus_conn.send(vd)

    def send_auditory(self, obs: Any) -> None:
        msg = "".join(chr(ch) for ch in obs["message"])
        ad = AuditoryData(msg)
        self.env_bus_conn.send(ad)

    def send_proprioceptive(self, obs: Any) -> None:
        pd = ProprioceptiveData.from_dict(obs)
        self.env_bus_conn.send(pd)

    def send_intrinsics(self, obs: Any) -> None:
        blstats = obs["blstats"]
        blstats_vals = {e.name.lower(): blstats[e.value] for e in blstat_offsets}
        for bit in condition_bits:
            blstats_vals[bit.name.lower()] = (
                True if blstats_vals["condition"] & bit.value else False
            )

        # TODO: remove BottomlineStats?
        bl = BottomlineStats(**blstats_vals)
        self.intrinsic_bus_conn.send(IntrinsicData(bl.dict()))


dump_env_file: Any = None


def _ascii_list(al: list[int]) -> str:
    result_string = "# "

    for ascii_value in al:
        result_string += chr(ascii_value)

    return result_string


def _print_screen(screen: list[list[int]]) -> None:
    global dump_env_file
    assert dump_env_file
    for row in screen:
        dump_env_file.write(_ascii_list(row) + "\n")


def _dump_env_start() -> None:
    settings = Config.get()
    if not settings.enable_gym_dump_env:
        return

    global dump_env_file
    dump_env_file = open(settings.dump_file, "w")
    dump_env_file.write("screens = [\n")


count = 0


def _dump_env_record(obs: Any, loop_num: int) -> None:
    settings = Config.get()
    if not settings.enable_gym_dump_env:
        return

    global dump_env_file
    assert dump_env_file

    global count
    count = count + 1
    settings = Config.get()
    if count >= settings.max_dump_frames:
        return

    _print_screen(obs["tty_chars"])
    dump_env_file.write("{ # screen" + str(loop_num) + "\n# fmt: off\n")
    dump_env_file.write(f'        "chars": {obs["chars"].tolist()},\n')
    dump_env_file.write(f'        "colors": {obs["colors"].tolist()},\n')
    dump_env_file.write(f'        "glyphs": {obs["glyphs"].tolist()},\n')
    dump_env_file.write(f'        "blstats": {obs["blstats"].tolist()},\n')
    dump_env_file.write("# fmt: on\n},\n")


def _dump_env_end() -> None:
    settings = Config.get()
    if not settings.enable_gym_dump_env:
        return

    logger.info("Completing game dump.")

    global dump_env_file
    assert dump_env_file
    dump_env_file.write("]\n")
    dump_env_file.flush()
    dump_env_file.close()


def _game_main(
    *,
    num_games: int,
    stop_event: threading.Event,
    on_run_name: Callable[[str], None],
) -> None:
    """Game loop entry point for thread-based execution.

    Called by GameManager on the game worker thread. Initializes all game-specific
    components, runs the game loop, and cleans up afterward. All Python objects
    (GraphCache, StepBuffer, etc.) are shared with the server thread via the
    process heap.

    Args:
        num_games: Number of games to play.
        stop_event: Set by the server to request cooperative shutdown.
        on_run_name: Callback to report the run name to GameManager.
    """
    from roc.framework.config import Config
    from roc.framework.component import Component
    from roc.framework.event import EventBus
    from roc.framework.expmod import ExpMod
    from roc.reporting import observability as obs_mod
    from roc.reporting.observability import Observability
    from roc.reporting.state import State

    # Reuse existing Config singleton (already initialized by the server process).
    Config.init(force=False)
    roc_logger.init()
    Observability.init(enable_parquet=True)

    # Report the run name so the server can set up the live session. Access
    # instance_id via the module attribute so we see the fresh value that
    # Observability.reset() writes between runs -- a bare `from ... import
    # instance_id` would snapshot the previous run's ID.
    on_run_name(obs_mod.instance_id)

    gym = NethackGym()
    Component.init()
    ExpMod.init()
    State.init()

    # Start in-process dashboard for standalone mode (no-op if already started).
    from roc.reporting.api_server import start_dashboard

    start_dashboard()

    try:
        gym.start(stop_event=stop_event)
    finally:
        # Clean up game-specific state so the next game run starts fresh.
        # Order matters: shut down components first (disposes bus subscriptions),
        # then clear bus names so new buses can be created on the next run,
        # then tear down the Observability singleton (which closes the
        # DuckLake store and regenerates instance_id for the next run).
        Component.reset()
        EventBus.clear_names()
        State.reset_init()
        Observability.reset()
        # GraphCache nodes/edges persist intentionally -- the server can still
        # query them after the game thread exits.
