import re
from typing import Any

import pytest


def normalize_whitespace(s: str) -> str:
    s = s.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def assert_similar(expected: str, actual: str, changes: list[tuple[str, str]]) -> None:
    match_str = re.escape(expected)
    for change in changes:
        # print(f"replacing '{change[0]}' with '{change[1]}'")
        match_str = match_str.replace(change[0], change[1])
    assert re.search(match_str, actual), f"expected '{expected}' to be similar to '{actual}'"


class FakeData:
    def __init__(self, foo: str, baz: int):
        self.foo = foo
        self.baz = baz


def component_response_args(
    # name of the component to test
    name: str,
    # type of the component to test
    type: str,
    # name of the class attribute that is the bus to send events on
    input_conn_attr: str,
    # values to send to the component as events
    vals: list[Any],
    *,
    # optional class attribute of where the result events can be found
    # defaults to input bus
    output_conn_attr: str | None = None,
) -> Any:
    # component_name, component_type, input_conn_attr, output_conn_attr, val =
    # request.params
    arg_tuple = (name, type, input_conn_attr, output_conn_attr, vals)
    return pytest.mark.parametrize("component_response", [arg_tuple], indirect=True)
