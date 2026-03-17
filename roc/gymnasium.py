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
from .reporting.screen_renderer import render_grid_html, screen_to_html_vals
from .reporting.state import State, _emit_state_record
from .reporting.wandb_reporter import WandbReporter


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
        WandbReporter.start_game(game_num)
        _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')

        # main environment loop
        while game_num <= settings.num_games:
            with Observability.tracer.start_as_current_span("observation"):
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
                if _buf is not None:
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

                # Push assembled StepData to live dashboard
                if _buf is not None:
                    _buf.push(
                        StepData(
                            step=loop_num,
                            game_number=game_num,
                            timestamp=_time_ns(),
                            screen=_screen_vals,
                            saliency=_saliency_vals,
                            features=_features,
                            object_info=_object_info,
                            focus_points=_focus_points,
                            graph_summary=_graph_summary,
                            event_summary=_event_summary,
                            game_metrics=game_metrics,
                        )
                    )

                # Log screen as rich media to W&B
                screen_state = State.get_states().screen.val
                if screen_state is not None:
                    screen_vals = screen_to_html_vals(screen_state)
                    screen_html = render_grid_html(screen_vals)
                    RocMetrics.log_media("screen", screen_html)

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
                    WandbReporter.end_game(
                        outcome=outcome,
                        final_score=score,
                    )
                    _emit_state_record(
                        "roc.game_end",
                        f'{{"game_number": {game_num}, "outcome": "{outcome}", "score": {score}}}',
                    )

                # Flush all buffered W&B data as one log call so all panels
                # (metrics, screen, saliency_map) share the same step counter.
                # end_game() buffers into the same step; start_game() buffers
                # into the next tick's step. Exactly one wandb.log() per tick.
                RocMetrics.flush_step()

                if done or truncated:
                    # restart and prepare to go again
                    self.env.reset()
                    game_counter.add(1)
                    game_num += 1
                    WandbReporter.start_game(game_num)
                    _emit_state_record("roc.game_start", f'{{"game_number": {game_num}}}')

        logger.info("NLE loop done, exiting.")
        WandbReporter.finish()
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
