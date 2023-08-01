import pprint
from typing import Any

import click
import gym
import nle  # noqa
from icecream import ic

from roc.component import Component
from roc.perception import perception_bus

pp = pprint.PrettyPrinter(width=41, compact=True)


def ascii_list(al: list[int]) -> str:
    result_string = ""

    for ascii_value in al:
        result_string += chr(ascii_value)

    return result_string


def int_list(al: list[int]) -> str:
    result_string = ""

    for ascii_value in al:
        result_string += str(ascii_value) + " "

    return result_string


def print_screen(screen: list[list[int]], *, as_int: bool = False) -> None:
    for row in screen:
        if not as_int:
            print(ascii_list(row))
        else:
            print(int_list(row))


class Environment(Component):
    def __init__(self) -> None:
        super().__init__("environment", "environment")
        # self.attach(perception_bus)
        perception_bus


@click.command
@click.option("--arg", default=1)
def cli(arg: Any) -> None:
    print("hello cli")
    print("arg is", arg)
    env = gym.make("NetHackScore-v0")
    print(repr(env.action_space))
    print(env.action_space)
    print(repr(env.observation_space))
    print(repr(env.reward_range))
    print(repr(env.action_space.sample()))
    print(repr(env.action_space.sample()))
    print(repr(env.action_space.sample()))
    env.print_action_meanings()
    (obs) = env.reset()
    pp.pprint(obs)
    print(ascii_list(obs["message"]))
    print("observation keys:", obs.keys())
    # print(obs["blstats"])
    # print(type(obs["blstats"]))
    # print(obs["blstats"].shape)
    ic(obs["tty_chars"])
    print_screen(obs["tty_chars"])
    ic(obs["screen_descriptions"])
    print(type(obs["screen_descriptions"]))
    print(obs["screen_descriptions"].shape)
    # for screen in obs["screen_descriptions"]:
    #     print("next screen")
    #     for row in screen:
    #         print(ascii_list(row))
    # env.render()

    # obs, reward, terminated, truncated = env.step(env.action_space.sample())
    # pp.pprint(obs)
    # pp.pprint(reward)
    # print("terminated:", terminated)
    # print("truncated:", truncated)
    print_screen(obs["chars"])
    print_screen(obs["colors"], as_int=True)
    print_screen(obs["inv_strs"])
    print(obs["glyphs"])
    print(type(obs["glyphs"]))
    print(obs["inv_oclasses"].shape)
    print(obs["inv_oclasses"])


if __name__ == "__main__":
    cli()
