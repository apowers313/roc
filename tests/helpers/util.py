import re
from typing import Any, Callable, Generic, TypeVar, cast
from unittest.mock import MagicMock

import pytest

from roc.component import Component
from roc.event import BusConnection, Event, EventBus, EventFilter


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


OutputDataType = TypeVar("OutputDataType")
InputDataType = TypeVar("InputDataType")


class StubComponent(Component, Generic[OutputDataType, InputDataType]):
    """A dummy component for testing communications and capturing the results for inspection"""

    name: str = "testing-stub"
    type: str = "stub"

    def __init__(
        self,
        input_bus: EventBus[InputDataType],
        output_bus: EventBus[OutputDataType],
        *,
        name: str = "testing-stub",
        filter: Callable[[Event[OutputDataType]], bool] | None = None,
    ):
        self.name = name
        super().__init__()

        # setup output filter
        def pass_filter(e: Event[OutputDataType]) -> bool:
            return True

        if filter is not None:
            self.filter: EventFilter[OutputDataType] = filter
        else:
            self.filter = pass_filter

        # setup output
        self.output_bus = output_bus
        self.output_conn = self.connect_bus(output_bus)
        self.output = MagicMock(spec=lambda *args, **kwargs: None, name="name")
        self.output_conn.listen(self.output, filter=self.filter)
        # self.output_conn.listen(self.output)

        # setup input
        self.input_bus = input_bus
        if input_bus is not output_bus:
            self.input_conn = self.connect_bus(input_bus)
        else:
            self.input_conn = cast(BusConnection[InputDataType], self.output_conn)


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
