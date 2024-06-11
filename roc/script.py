"""This is a Python script that runs the Gym and agent, typically from the
command-line. See the Makefile for how to run this script."""

import pprint
from typing import Any

import click
import nle  # noqa

import roc

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
            print(ascii_list(row))  # noqa: T201
        else:
            print(int_list(row))  # noqa: T201


@click.command
@click.option("--arg", default=1)
def cli(arg: Any) -> None:
    roc.init()
    roc.start()


if __name__ == "__main__":
    cli()
