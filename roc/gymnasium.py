# pragma: no cover

"""This module is a wrapper around the Gym / Gymnasium interfaces and drives all
the interactions between the agent and the system, including the main event loop.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

import nle
from pydantic import BaseModel

# from roc import init as roc_init
from .action import ActionCount, action_bus
from .breakpoint import breakpoints
from .component import Component
from .config import Config
from .jupyter.state import print_state, states
from .location import TextGrid
from .logger import logger
from .perception import Perception, VisionData

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym


class Gym(Component, ABC):
    """A wrapper around an OpenAI Gym / Farama Gymnasium that drives the event
    loop and interfaces to the ROC agent.
    """

    def __init__(self, gym_id: str, *, gym_opts: dict[str, Any] | None = None) -> None:
        super().__init__()
        gym_opts = gym_opts or {}
        self.env = gym.make(gym_id, **gym_opts)

        # setup communications
        self.env_bus = Perception.bus
        self.action_bus = action_bus
        self.env_bus_conn = self.env_bus.connect(self)
        self.action_bus_conn = self.action_bus.connect(self)

        # config actions
        self.action_count = self.env.action_space.n
        self.config_actions(self.action_count)
        settings = Config.get()
        settings.action_count = self.action_count
        settings.observation_shape = self.env.observation_space["glyphs"].shape

        # TODO: config environment
        # setup which features detectors to use on each bus

    @abstractmethod
    def send_obs(self, obs: Any) -> None:
        pass

    @abstractmethod
    def config_actions(self, action_count: int) -> None:
        pass

    @logger.catch
    def start(self) -> None:
        obs = self.env.reset()
        settings = Config.get()

        done = False
        _dump_env_start()

        logger.info("Starting NLE loop...")
        loop_num = 0

        # main environment loop
        while not done:
            # logger.trace(f"Sending observation: {obs}")
            breakpoints.check()

            # save the current screen
            screen = nle.nethack.tty_render(obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"])
            states.screen.set(screen)

            # do all the real work
            self.send_obs(obs)

            # get an action
            action = self.await_action()
            logger.trace(f"Doing action: {action}")

            # perform the action and get the next observation
            step_res = self.env.step(action)
            obs = step_res[0]

            # optionally save the screen to file
            _dump_env_record(obs, loop_num)

            # check to see if we are done
            if len(step_res) == 5:
                done = step_res[2] or step_res[3]
            else:
                done = step_res[2]

            logger.trace(f"Main loop done: {done}")

            # set and save the loop number
            loop_num += 1
            states.loop.set(loop_num)
            if (loop_num % settings.status_update) == 0:
                print_state()

        logger.info("NLE loop done.")
        _dump_env_end()

    def decode_action(self, action: int) -> Any:
        return action

    def await_action(self) -> Any:
        # TODO: self.action_bus_conn.subject.first()

        # warnings.warn("await action not implemented, defaulting to '.' for every action")

        default_action = 19  # 19 = 46 = "." = do nothing
        action = self.decode_action(default_action)
        return action


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

    X: int
    Y: int
    STR25: int
    STR125: int
    DEX: int
    CON: int
    INT: int
    WIS: int
    CHA: int
    SCORE: int
    HP: int
    HPMAX: int
    DEPTH: int
    GOLD: int
    ENE: int
    ENEMAX: int
    AC: int
    HD: int
    XP: int
    EXP: int
    TIME: int
    HUNGER: int
    CAP: int
    DNUM: int
    DLEVEL: int
    CONDITION: int
    ALIGN: int


class NethackGym(Gym):
    """Wrapper around the Gym class for driving the Nethack interface to the ROC
    agent. Decodes Nethack specific data and sends it to the agent as Events.
    """

    def __init__(self, *, gym_opts: dict[str, Any] | None = None) -> None:
        gym_opts = gym_opts or {}
        super().__init__("NetHackScore-v0", **gym_opts)

    def config_actions(self, action_count: int) -> None:
        a = ActionCount(action_count=action_count)
        self.action_bus_conn.send(a)

    def send_obs(self, obs: Any) -> None:
        self.send_vision(obs)
        self.send_intrinsics(obs)

    def send_vision(self, obs: Any) -> None:
        vd = VisionData.from_dict(obs)
        # vd = VisionData.from_dict(
        #     {
        #         "chars": obs["chars"].copy(),
        #         "glyphs": obs["glyphs"].copy(),
        #         "colors": obs["colors"].copy(),
        #     }
        # )
        self.env_bus_conn.send(vd)

    def send_auditory(self) -> None:
        pass

    def send_proprioceptive(self) -> None:
        pass

    def send_intrinsics(self, obs: Any) -> None:
        pass
        # NOTE: obs["blstats"] is an ndarray object from numpy
        bl = obs["blstats"].tolist()
        blstat_args = {e.name: bl[e.value] for e in blstat_offsets}
        # print("blstat_args", blstat_args)

        blstats = BottomlineStats(**blstat_args)
        # print("blstats", blstats.model_dump())
        blstat_conds = {bit.name for bit in condition_bits if blstats.CONDITION & bit.value}
        # TODO: remove... just curious if conditions ever get set
        if len(blstat_conds):
            logger.warning(f"!!! FOUND CONDITIONS: {blstats.CONDITION} = {blstat_conds}")
            logger.warning(f"bltstats: {obs['blstats']} :: {blstat_args}")
            self.env.render()


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
    dump_env_file.write(f"        \"chars\": {obs['chars'].tolist()},\n")
    dump_env_file.write(f"        \"colors\": {obs['colors'].tolist()},\n")
    dump_env_file.write(f"        \"glyphs\": {obs['glyphs'].tolist()},\n")
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
