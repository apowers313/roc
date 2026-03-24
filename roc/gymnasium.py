# pragma: no cover

"""This module is a wrapper around the Gym / Gymnasium interfaces and drives all
the interactions between the agent and the system, including the main event loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]
import gymnasium as gym
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


from .action import Action, ActionRequest, TakeAction
from .breakpoint import breakpoints
from .component import Component
from .config import Config
from .graphdb import GraphDB
from .intrinsic import Intrinsic, IntrinsicData
from .logger import logger
from .perception import AuditoryData, Perception, ProprioceptiveData, VisionData
from .reporting.metrics import RocMetrics
from .reporting.observability import Observability
from .reporting.screen_renderer import screen_to_html_vals
from .reporting.state import State, _emit_state_record


@dataclass
class GameLoopContext:
    """Invariant state for the game observation loop."""

    callback_url: str | None
    callback_ctx: Any
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
        self._server_stop = False

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
    def start(self) -> None:
        logger.debug("Starting NLE loop...")
        obs, _reset_info = self.env.reset()
        settings = Config.get()

        callback_url = settings.dashboard_callback_url
        callback_ctx = _setup_callback_context(callback_url, settings)

        _publish_action_map(settings.gym_actions, callback_url, callback_ctx)

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
        loop_ctx = GameLoopContext(callback_url, callback_ctx, observation_counter)

        # main environment loop
        while game_num <= settings.num_games:
            with Observability.tracer.start_as_current_span("observation"):
                obs, done, truncated, loop_num = self._run_observation_step(
                    obs, loop_num, game_num, loop_ctx
                )

                if done or truncated:
                    _handle_game_over(obs, game_num, done, settings)

                if done or truncated:
                    self.env.reset()
                    game_counter.add(1)
                    game_num += 1
                    _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')

                if self._server_stop:
                    logger.info("Server requested stop, exiting game loop.")
                    break

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
    ) -> tuple[Any, bool, bool, int]:
        """Execute a single observation-action-step cycle.

        Returns the new (obs, done, truncated, loop_num) tuple.
        """
        from roc.reporting.step_log_sink import set_current_step

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

        loop_ctx.observation_counter.add(1)
        loop_num += 1
        State.get_states().loop.set(loop_num)
        State.maybe_emit_snapshot(loop_num)
        if Config.get().emit_state:
            State.emit_state_logs()

        self._server_stop = _push_dashboard_data(
            obs, loop_num, game_num, loop_ctx.callback_url, loop_ctx.callback_ctx
        )

        return obs, done, truncated, loop_num


def _setup_callback_context(callback_url: str | None, settings: Config) -> Any:
    """Build an SSL context for the HTTP callback URL, if needed."""
    if not callback_url:
        return None
    import ssl

    if callback_url.startswith("https://"):
        ctx = ssl.create_default_context()
        if settings.ssl_certfile:
            ctx.load_verify_locations(settings.ssl_certfile)
    else:
        ctx = None
    logger.info("Step callback URL: {}", callback_url)
    return ctx


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
    callback_url: str | None,
    callback_ctx: Any,
) -> None:
    """Build and publish the action map to the server or to disk."""
    if not gym_actions:
        return

    action_map = _build_action_map(gym_actions)

    if callback_url:
        _post_action_map_to_server(action_map, callback_url, callback_ctx)
    else:
        _save_action_map_to_file(action_map)


def _post_action_map_to_server(
    action_map: list[dict[str, Any]],
    callback_url: str,
    callback_ctx: Any,
) -> None:
    """POST the action map to the dashboard server."""
    from roc.reporting.observability import Observability

    dl_store = Observability.get_ducklake_store()
    if dl_store is None:
        return
    try:
        import json
        import urllib.request

        run_name = dl_store.run_dir.name
        base = callback_url.rsplit("/api/internal/step", 1)[0]
        map_url = f"{base}/api/internal/action-map"
        payload = json.dumps(
            {"run_name": run_name, "action_map": action_map},
            separators=(",", ":"),
        ).encode()
        req = urllib.request.Request(
            map_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5, context=callback_ctx)
        logger.debug("Sent action map ({} entries) to server", len(action_map))
    except Exception:
        logger.opt(exception=True).debug("Failed to send action map")


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
    from roc.graphdb import Edge, Node

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
    from roc.event import Event

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


def _build_transform_summary(states: Any) -> dict[str, Any] | None:
    """Build transform summary from the current transform state."""
    if states.transform.val is None:
        return None
    t = states.transform.val.transform
    changes_list: list[dict[str, Any]] = []
    for edge in t.src_edges:
        dst = edge.dst
        ch: dict[str, Any] = {"description": str(dst)}
        if hasattr(dst, "name"):
            ch["type"] = type(dst).__name__
            ch["name"] = dst.name
        if hasattr(dst, "normalized_change"):
            ch["normalized_change"] = dst.normalized_change
        changes_list.append(ch)
    return {"count": len(changes_list), "changes": changes_list}


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
    from roc.component import ComponentName, ComponentType, loaded_components
    from roc.sequencer import Sequencer

    seq_comp = loaded_components.get((ComponentName("sequencer"), ComponentType("sequencer")))
    if not isinstance(seq_comp, Sequencer) or seq_comp.last_frame is None:
        return None
    return seq_comp.last_frame


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a single object to a summary dict."""
    od: dict[str, Any] = {"id": str(obj.id)[:8]}
    if hasattr(obj, "last_x") and hasattr(obj, "last_y"):
        od["x"] = obj.last_x
        od["y"] = obj.last_y
    if hasattr(obj, "resolve_count"):
        od["resolve_count"] = obj.resolve_count
    return od


def _frame_to_summary(frame: Any, states: Any) -> dict[str, Any]:
    """Build the summary dict from a frame and current states."""
    from roc.intrinsic import IntrinsicNode

    objs = frame.objects
    obj_dicts = [_obj_to_dict(obj) for obj in objs]
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
    from roc.predict import NoPrediction

    prediction: dict[str, Any] = {"made": not isinstance(states.predict.val, NoPrediction)}
    try:
        from roc.component import (
            ComponentName,
            ComponentType,
            loaded_components,
        )
        from roc.predict import Predict

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


def _push_step_to_server(
    step_data: Any,
    callback_url: str,
    callback_ctx: Any,
) -> bool:
    """POST step data to the dashboard server. Returns True if server requested stop."""
    try:
        import dataclasses
        import json
        import urllib.request

        payload = json.dumps(
            dataclasses.asdict(step_data), separators=(",", ":"), default=str
        ).encode()
        req = urllib.request.Request(
            callback_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=2, context=callback_ctx)
        resp_body = json.loads(resp.read())
        return bool(resp_body.get("stop"))
    except Exception:
        return False  # best-effort, don't break the game loop


def _push_dashboard_data(
    obs: Any,
    loop_num: int,
    game_num: int,
    callback_url: str | None,
    callback_ctx: Any,
) -> bool:
    """Collect step data and push to the dashboard buffer and/or server.

    Returns True if the server requested a stop.
    """
    import json as _json

    game_metrics = _extract_game_metrics(obs)
    RocMetrics.log_step(game_metrics)
    _emit_state_record(
        "roc.game_metrics",
        _json.dumps(game_metrics, separators=(",", ":")),
    )

    from roc.reporting.step_buffer import get_step_buffer

    buf = get_step_buffer()
    dashboard_active = buf is not None or bool(callback_url)

    inventory = _parse_inventory(obs) if dashboard_active else None

    if dashboard_active and inventory is not None:
        _emit_state_record(
            "roc.inventory",
            _json.dumps(inventory, separators=(",", ":")),
        )

    if not dashboard_active:
        return False

    from time import time_ns

    from roc.reporting.run_store import StepData
    from roc.reporting.step_log_sink import drain_step_logs

    states = State.get_states()
    screen_vals, saliency_vals, features = _collect_screen_data(states)
    object_info, focus_points = _collect_object_data(states)

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
        intrinsics=(
            {
                "raw": states.intrinsic.val.intrinsics,
                "normalized": states.intrinsic.val.normalized_intrinsics,
            }
            if states.intrinsic.val is not None
            else None
        ),
        significance=(
            states.significance.val.significance if states.significance.val is not None else None
        ),
        action_taken=_build_action_taken_dict(states),
        sequence_summary=_build_sequence_summary(states),
        transform_summary=_build_transform_summary(states),
        prediction=_build_prediction_data(states),
        message=_collect_message(states),
        phonemes=_collect_phonemes(states),
        inventory=inventory,
    )
    if buf is not None:
        buf.push(step_data)

    server_stop = False
    if callback_url:
        server_stop = _push_step_to_server(step_data, callback_url, callback_ctx)
    return server_stop


def _handle_game_over(obs: Any, game_num: int, done: bool, settings: Config) -> None:
    """Log game over info and optionally flush the graph database."""
    screen = ""
    for row in obs["tty_chars"]:
        for ch in row:
            screen += chr(ch)
        screen += "\n"
    logger.info(screen, death=True, game_num=game_num)
    logger.info(f"Game {game_num} completed, starting next game")
    if settings.graphdb_flush:
        GraphDB.flush()
    if settings.graphdb_export:
        GraphDB.export()
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
