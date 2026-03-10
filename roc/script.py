"""This is a Python script that runs the Gym and agent, typically from the
command-line. See the Makefile for how to run this script.
"""

import json
import pprint
from types import UnionType
from typing import Any, get_args, get_origin

import click

import roc
from roc.config import Config

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


class JsonParam(click.ParamType):
    """Click parameter type that parses JSON strings for complex config values."""

    name = "JSON"

    def convert(self, value: Any, param: Any, ctx: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                self.fail(f"'{value}' is not valid JSON", param, ctx)
        return value


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional[X] (i.e. X | None) to get X."""
    origin = get_origin(annotation)
    if origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


_SIMPLE_TYPES = {bool, int, float, str}


@click.command
@click.pass_context
def cli(ctx: click.Context, **kwargs: Any) -> None:
    """Run the ROC agent."""
    config = {k: v for k, v in kwargs.items() if v is not None}
    roc.init(config=config if config else None)
    roc.start()


# Auto-generate click options from Config model fields
for _field_name, _field_info in Config.model_fields.items():
    _annotation = _field_info.annotation
    if _annotation is None:
        continue

    _annotation = _unwrap_optional(_annotation)
    _opt_name = f"--{_field_name.replace('_', '-')}"

    if _annotation is bool:
        _no_opt = f"--no-{_field_name.replace('_', '-')}"
        cli = click.option(
            _opt_name + "/" + _no_opt,
            _field_name,
            default=None,
            show_default=False,
        )(cli)
    elif _annotation in _SIMPLE_TYPES:
        cli = click.option(
            _opt_name,
            _field_name,
            type=_annotation,
            default=None,
            show_default=False,
        )(cli)
    else:
        # Complex types (list, dict, tuple, etc.) -- accept as JSON string
        cli = click.option(
            _opt_name,
            _field_name,
            type=JsonParam(),
            default=None,
            show_default=False,
        )(cli)


if __name__ == "__main__":
    cli()
