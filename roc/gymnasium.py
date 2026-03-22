# pragma: no cover

"""This module is a wrapper around the Gym / Gymnasium interfaces and drives all
the interactions between the agent and the system, including the main event loop.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]
import gymnasium as gym
import nle
from pydantic import BaseModel, Field

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


class Gym(Component, ABC):
    """A wrapper around an OpenAI Gym / Farama Gymnasium that drives the event
    loop and interfaces to the ROC agent.
    """

    name: str = "gym"
    type: str = "game"

    def __init__(self, gym_id: str, *, gym_opts: dict[str, Any] | None = None) -> None:
        logger.debug(f"Initializing Gym...")
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
    def start(self) -> None:
        logger.debug("Starting NLE loop...")
        obs, reset_info = self.env.reset()
        settings = Config.get()

        # HTTP callback for pushing step data to an external dashboard server
        _callback_url = settings.dashboard_callback_url
        _callback_ctx: Any = None
        if _callback_url:
            import ssl
            import urllib.request

            # Skip SSL verification for localhost (self-signed certs)
            _callback_ctx = ssl.create_default_context()
            _callback_ctx.check_hostname = False
            _callback_ctx.verify_mode = ssl.CERT_NONE
            logger.info("Step callback URL: {}", _callback_url)

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

        # main environment loop
        while game_num <= settings.num_games:
            with Observability.tracer.start_as_current_span("observation"):
                # Tag log records with the step number that StepData will use.
                # loop_num is incremented later in this iteration, so +1 here
                # ensures logs match StepData.step.
                from roc.reporting.step_log_sink import set_current_step

                set_current_step(loop_num + 1)

                logger.trace(f"Sending observation: {obs}")
                breakpoints.check()

                State.get_states().screen.set(
                    {
                        "chars": obs["tty_chars"],
                        "colors": obs["tty_colors"],
                        "cursor": obs["tty_cursor"],
                    }
                )

                # do all the real work
                self.send_obs(obs)

                # get an action
                action = self.get_action()
                logger.trace(f"Doing action: {action}")

                # perform the action and get the next observation
                obs, reward, done, truncated, info = self.env.step(action)

                # optionally save the screen to file
                _dump_env_record(obs, loop_num)

                logger.trace(f"Main loop done: {done}, {truncated}")

                # set and save the loop number
                observation_counter.add(1)
                loop_num += 1
                State.get_states().loop.set(loop_num)
                State.maybe_emit_snapshot(loop_num)
                if Config.get().emit_state:
                    State.emit_state_logs()

                # Push StepData to live dashboard buffer (cheap -- just dict refs)
                from roc.reporting.step_buffer import get_step_buffer

                _buf = get_step_buffer()
                if _buf is not None or _callback_url:
                    from time import time_ns as _time_ns

                    from roc.event import Event
                    from roc.graphdb import Edge, Node
                    from roc.reporting.run_store import StepData

                    _states = State.get_states()
                    _screen_state = _states.screen.val
                    _screen_vals = (
                        screen_to_html_vals(_screen_state) if _screen_state is not None else None
                    )
                    _saliency_state = _states.salency.val
                    _saliency_vals = (
                        _saliency_state.to_html_vals() if _saliency_state is not None else None
                    )
                    _features = None
                    if _saliency_state is not None:
                        _feat_report = _saliency_state.feature_report()
                        _features = [_feat_report] if _feat_report else None
                    _object_info = None
                    if _states.object.val is not None:
                        _object_info = [{"raw": str(_states.object)}]
                    _focus_points = None
                    if _states.attention.val is not None:
                        _focus_points = [{"raw": str(_states.attention.val.focus_points)}]
                    _node_cache = Node.get_cache()
                    _edge_cache = Edge.get_cache()
                    _graph_summary = {
                        "node_count": _node_cache.currsize,
                        "node_max": _node_cache.maxsize,
                        "edge_count": _edge_cache.currsize,
                        "edge_max": _edge_cache.maxsize,
                    }
                    _step_counts = Event.get_step_counts()
                    _event_summary = [_step_counts] if _step_counts else None
                    _resolution_metrics = _states.resolution.val
                    _attenuation = _states.attenuation_data.val

                    # New pipeline state fields
                    _intrinsics = None
                    _significance = None
                    _action_taken = None
                    _transform_summary = None
                    _prediction = None
                    _message = None
                    _phonemes = None
                    _inventory = None

                    if _states.intrinsic.val is not None:
                        _intr = _states.intrinsic.val
                        _intrinsics = {
                            "raw": _intr.intrinsics,
                            "normalized": _intr.normalized_intrinsics,
                        }

                    if _states.significance.val is not None:
                        _significance = _states.significance.val.significance

                    if _states.action.val is not None:
                        _act_id = int(_states.action.val.action)
                        _action_taken_dict: dict[str, Any] = {"action_id": _act_id}
                        try:
                            _gym_actions = Config.get().gym_actions
                            if _gym_actions and _act_id < len(_gym_actions):
                                _act_enum = _gym_actions[_act_id]
                                _action_taken_dict["action_name"] = str(
                                    getattr(_act_enum, "name", _act_enum)
                                )
                        except Exception:
                            pass
                        _action_taken = _action_taken_dict

                    if _states.transform.val is not None:
                        _t = _states.transform.val.transform
                        _changes = [str(e.dst) for e in _t.src_edges]
                        _transform_summary = {
                            "count": len(_changes),
                            "changes": _changes,
                        }

                    if _states.predict.val is not None:
                        from roc.predict import NoPrediction as _NP

                        _prediction = {
                            "made": not isinstance(_states.predict.val, _NP),
                        }

                    if _states.message.val is not None:
                        _msg = _states.message.val.strip()
                        if _msg:
                            _message = _msg

                    _phonemes = (
                        [
                            {"word": pw.word, "phonemes": pw.phonemes, "is_break": pw.is_break}
                            for pw in _states.phonemes.val
                        ]
                        if _states.phonemes.val is not None
                        else None
                    )

                    # Inventory from obs
                    try:
                        _inv_strs = obs["inv_strs"]
                        _inv_letters = obs["inv_letters"]
                        _inv_glyphs = obs["inv_glyphs"]
                        _inv_items: list[dict[str, Any]] = []
                        for _i in range(len(_inv_strs)):
                            _item_str = "".join(chr(ch) for ch in _inv_strs[_i]).strip("\x00 ")
                            _glyph = int(_inv_glyphs[_i])
                            if not _item_str or _glyph == 5976:
                                continue
                            _inv_items.append(
                                {
                                    "letter": chr(int(_inv_letters[_i])),
                                    "item": _item_str,
                                    "glyph": _glyph,
                                }
                            )
                        if _inv_items:
                            _inventory = _inv_items
                    except Exception:
                        pass

                # Log per-tick game state to W&B
                blstats = obs["blstats"]
                game_metrics = {
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
                RocMetrics.log_step(game_metrics)

                # Emit game metrics as OTel log record for Parquet storage
                import json as _json

                _emit_state_record(
                    "roc.game_metrics",
                    _json.dumps(game_metrics, separators=(",", ":")),
                )

                # Emit inventory as OTel log record
                if (_buf is not None or _callback_url) and _inventory is not None:
                    _emit_state_record(
                        "roc.inventory",
                        _json.dumps(_inventory, separators=(",", ":")),
                    )

                # Push assembled StepData to live dashboard
                if _buf is not None or _callback_url:
                    from roc.reporting.step_log_sink import drain_step_logs

                    _step_logs = drain_step_logs(loop_num)

                    _step_data = StepData(
                        step=loop_num,
                        game_number=game_num,
                        timestamp=_time_ns(),
                        screen=_screen_vals,
                        saliency=_saliency_vals,
                        features=_features,
                        object_info=_object_info,
                        focus_points=_focus_points,
                        attenuation=_attenuation,
                        resolution_metrics=_resolution_metrics,
                        graph_summary=_graph_summary,
                        event_summary=_event_summary,
                        game_metrics=game_metrics,
                        logs=_step_logs,
                        intrinsics=_intrinsics,
                        significance=_significance,
                        action_taken=_action_taken,
                        transform_summary=_transform_summary,
                        prediction=_prediction,
                        message=_message,
                        phonemes=_phonemes,
                        inventory=_inventory,
                    )
                    if _buf is not None:
                        _buf.push(_step_data)

                    # Push step to external dashboard server via HTTP callback
                    if _callback_url:
                        try:
                            import dataclasses as _dc
                            import urllib.request

                            _payload = _json.dumps(
                                _dc.asdict(_step_data), separators=(",", ":"), default=str
                            ).encode()
                            _req = urllib.request.Request(
                                _callback_url,
                                data=_payload,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            urllib.request.urlopen(_req, timeout=2, context=_callback_ctx)
                        except Exception:
                            pass  # best-effort, don't break the game loop

                if done or truncated:
                    # log game over info
                    screen = ""
                    for row in obs["tty_chars"]:
                        for ch in row:
                            screen += chr(ch)
                        screen += "\n"
                    logger.info(screen, death=True, game_num=game_num)
                    logger.info(f"Game {game_num} completed, starting next game")
                    # flush cache to graphdb
                    if settings.graphdb_flush:
                        GraphDB.flush()
                    if settings.graphdb_export:
                        GraphDB.export()
                    # Buffer game-end data before flush so it's in the same step
                    blstats = obs["blstats"]
                    score = int(blstats[blstat_offsets.SCORE])
                    outcome = "done" if done else "truncated"
                    _emit_state_record(
                        "roc.game_end",
                        f'{{"game_number": {game_num}, "outcome": "{outcome}", "score": {score}}}',
                    )

                if done or truncated:
                    # restart and prepare to go again
                    self.env.reset()
                    game_counter.add(1)
                    game_num += 1
                    _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')

        logger.info("NLE loop done, exiting.")
        from roc.reporting.api_server import stop_dashboard

        stop_dashboard()
        Observability.shutdown()
        _dump_env_end()


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
        # print("inv_glyphs", obs["inv_glyphs"])
        # print("inv_strs", obs["inv_strs"])
        # print("inv_letters", obs["inv_letters"])
        # print("inv_oclasses", obs["inv_oclasses"])
        # for inv in obs["inv_strs"]:
        #     invline = "".join(chr(ch) for ch in inv)
        #     print(invline)
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


def _print_screen(screen: list[list[int]], *, as_int: bool = False) -> None:
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
