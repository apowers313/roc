from typing import Any

import pprint

import click
import gym
import nle

from roc.component import Component
from roc.perception import perception_bus

pp = pprint.PrettyPrinter(width=41, compact=True)


def ascii_list(al: list[int]) -> str:
    result_string = ""

    for ascii_value in al:
        result_string += chr(ascii_value)

    return result_string


class Environment(Component):
    def __init__(self):
        super().__init__("environment", "environment")
        self.attach(perception_bus)


@click.command  # type: ignore
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
    obs, reward, terminated, truncated = env.step(env.action_space.sample())
    pp.pprint(obs)
    pp.pprint(reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    # print("info", info)
    # print("done", done)


if __name__ == "__main__":
    cli()
