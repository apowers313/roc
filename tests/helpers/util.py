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
    name: str,
    type: str,
    input_conn_attr: str,
    val: Any,
    *,
    output_conn_attr: str | None = None,
) -> Any:
    # component_name, component_type, input_conn_attr, output_conn_attr, val =
    # request.params
    arg_tuple = (name, type, input_conn_attr, output_conn_attr, val)
    return pytest.mark.parametrize("component_response", [arg_tuple], indirect=True)
