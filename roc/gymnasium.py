"""This module is a wrapper around the Gym / Gymnasium interfaces and drives all
the interactions between the agent and the system, including the main event loop."""

# pragma: no cover
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

from pydantic import BaseModel

# from roc import init as roc_init
from .action import ActionCount, action_bus
from .component import Component
from .config import Config
from .logger import logger
from .perception import VisionData, perception_bus

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym


class Gym(Component, ABC):
    def __init__(self, gym_id: str, *, gym_opts: dict[str, Any] | None = None) -> None:
        super().__init__()
        gym_opts = gym_opts or {}
        self.env = gym.make(gym_id, **gym_opts)

        # setup communications
        self.env_bus = perception_bus
        self.action_bus = action_bus
        self.env_bus_conn = self.env_bus.connect(self)
        self.action_bus_conn = self.action_bus.connect(self)

        # config actions
        self.action_count = self.env.action_space.n
        self.config_actions(self.action_count)

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

        done = False
        dump_env_start()

        # main environment loop
        while not done:
            # logger.trace(f"Sending observation: {obs}")
            self.send_obs(obs)
            action = self.await_action()
            logger.trace(f"Doing action: {action}")
            step_res = self.env.step(action)
            obs = step_res[0]
            dump_env_record(obs)

            if len(step_res) == 5:
                done = step_res[2] or step_res[3]
            else:
                done = step_res[2]

            # self.env.render()
            logger.trace(f"Main loop done: {done}")

        dump_env_end()

    def decode_action(self, action: int) -> Any:
        return action

    def await_action(self) -> Any:
        # TODO: self.action_bus_conn.subject.first()
        logger.warning("AWAIT ACTION NOT IMPLEMENTED")
        default_action = 19  # 19 = 46 = "." = do nothing
        action = self.decode_action(default_action)
        return action


class blstat_offsets(IntEnum):
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
    # fmt: off
    BAREH =     0x00000001
    BLIND =     0x00000002
    BUSY =      0x00000004
    CONF =      0x00000008
    DEAF =      0x00000010
    ELF_IRON =  0x00000020
    FLY =       0x00000040
    FOODPOIS =  0x00000080
    GLOWHANDS = 0x00000100
    GRAB =      0x00000200
    HALLU =     0x00000400
    HELD =      0x00000800
    ICY =       0x00001000
    INLAVA =    0x00002000
    LEV =       0x00004000
    PARLYZ =    0x00008000
    RIDE =      0x00010000
    SLEEPING =  0x00020000
    SLIME =     0x00040000
    SLIPPERY =  0x00080000
    STONE =     0x00100000
    STRNGL =    0x00200000
    STUN =      0x00400000
    SUBMERGED = 0x00800000
    TERMILL   = 0x01000000
    TETHERED =  0x02000000
    TRAPPED =   0x04000000
    UNCONSC =   0x08000000
    WOUNDEDL =  0x10000000
    HOLDING =   0x20000000
    # fmt: on


class BaselineStats(BaseModel):
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
        # spectrum = [obs["chars"], obs["colors"], obs["glyphs"]]

        self.env_bus_conn.send(VisionData(screen=obs["chars"]))

    def send_auditory(self) -> None:
        pass

    def send_proprioceptive(self) -> None:
        pass

    def send_intrinsics(self, obs: Any) -> None:
        pass
        # NOTE: obs["blstats"] is an ndarray object from numpy
        # bl = obs["blstats"].tolist()
        # blstat_args = {e.name: bl[e.value] for e in blstat_offsets}
        # print("blstat_args", blstat_args)

        # blstats = BaselineStats(**blstat_args)
        # print("blstats", blstats.model_dump())
        # blstat_conds = {bit.name for bit in condition_bits if blstats.CONDITION & bit.value}
        # # TODO: remove... just curious if conditions ever get set
        # if len(blstat_conds):
        #     logger.warning("!!! FOUND CONDITIONS", blstat_conds)


dump_env_file: Any = None


def ascii_list(al: list[int]) -> str:
    result_string = "# "

    for ascii_value in al:
        result_string += chr(ascii_value)

    return result_string


def print_screen(screen: list[list[int]], *, as_int: bool = False) -> None:
    global dump_env_file
    assert dump_env_file
    for row in screen:
        dump_env_file.write(ascii_list(row) + "\n")


def dump_env_start() -> None:
    if not Config.enable_gym_dump_env:
        return

    global dump_env_file
    dump_env_file = open("env_dump.py", "w")
    dump_env_file.write("[\n")


count = 0


def dump_env_record(obs: Any) -> None:
    if not Config.enable_gym_dump_env:
        return

    global dump_env_file
    assert dump_env_file

    global count
    count = count + 1
    if count >= Config.max_dump_frames:
        return

    print_screen(obs["tty_chars"])
    dump_env_file.write("{\n# fmt: off\n")
    dump_env_file.write(f"        \"chars\": {obs['chars'].tolist()},\n")
    dump_env_file.write(f"        \"colors\": {obs['colors'].tolist()},\n")
    dump_env_file.write(f"        \"glyphs\": {obs['glyphs'].tolist()},\n")
    dump_env_file.write("# fmt: on\n},\n")


def dump_env_end() -> None:
    if not Config.enable_gym_dump_env:
        return

    global dump_env_file
    assert dump_env_file
    dump_env_file.write("]\n")
    dump_env_file.close()
