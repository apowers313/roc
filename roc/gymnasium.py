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
from .intrinsic import Intrinsic
from .jupyter.state import print_state, states
from .logger import logger
from .reporting.observability import Observability
from .perception import Perception, VisionData


class Gym(Component, ABC):
    """A wrapper around an OpenAI Gym / Farama Gymnasium that drives the event
    loop and interfaces to the ROC agent.
    """

    def __init__(self, gym_id: str, *, gym_opts: dict[str, Any] | None = None) -> None:
        super().__init__()
        gym_opts = gym_opts or {}
        logger.info(f"Gym options: {gym_opts}")
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
    @Observability.get_tracer().start_as_current_span("start")
    def start(self) -> None:
        logger.info("Starting NLE loop...")
        obs, reset_info = self.env.reset()
        settings = Config.get()

        done = False
        truncated = False
        _dump_env_start()

        loop_num = 0
        game_num = 0
        game_counter = Observability.get_meter().create_counter(
            "roc.game_total", unit="games", description="total number of games completed"
        )
        observation_counter = Observability.get_meter().create_counter(
            "roc.obs_total", unit="observations", description="total number of observations"
        )
        game_counter.add(1)

        # main environment loop
        while game_num < settings.num_games:
            with Observability.get_tracer().start_as_current_span("observation"):
                logger.trace(f"Sending observation: {obs}")
                breakpoints.check()

                # save the current screen
                screen = nle.nethack.tty_render(
                    obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"]
                )
                states.screen.set(screen)

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
                states.loop.set(loop_num)
                if (loop_num % settings.status_update) == 0:
                    print_state()

                if done or truncated:
                    logger.info(f"Game {game_num} completed, starting next game")
                    self.env.reset()
                    game_counter.add(1)
                    game_num += 1

        logger.info("NLE loop done, exiting.")
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
        # msg = "".join(chr(ch) for ch in obs["message"])
        # print("message", msg)
        pass

    def send_proprioceptive(self) -> None:
        pass

    def send_intrinsics(self, obs: Any) -> None:
        blstats = obs["blstats"]
        blstats_vals = {e.name.lower(): blstats[e.value] for e in blstat_offsets}
        for bit in condition_bits:
            blstats_vals[bit.name.lower()] = (
                True if blstats_vals["condition"] & bit.value else False
            )

        bl = BottomlineStats(**blstats_vals)
        self.intrinsic_bus_conn.send(bl.dict())


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
