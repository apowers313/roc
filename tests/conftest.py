# mypy: disable-error-code="no-untyped-def"
import gc
from typing import Any, Generator, cast
from unittest.mock import MagicMock

import pytest
from helpers.util import FakeData

from roc.action import ActionData, action_bus
from roc.component import Component, register_component
from roc.config import Config
from roc.event import BusConnection, EventBus
from roc.graphdb import Edge, GraphDB, Node
from roc.perception import PerceptionData, perception_bus


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None, None, None]:
    yield

    node_cache = Node.get_cache()
    edge_cache = Edge.get_cache()
    for n in node_cache:
        node_cache[n]._no_save = True
    for e in edge_cache:
        edge_cache[e]._no_save = True

    Node.get_cache().clear()
    Edge.get_cache().clear()


@pytest.fixture
def new_edge() -> tuple[Edge, Node, Node]:
    src = Node(labels=["TestNode"])
    dst = Node(labels=["TestNode"])
    e = Node.connect(src, dst, "Test")
    return (e, src, dst)


@pytest.fixture
def eb_reset() -> None:
    EventBus.clear_names()


@pytest.fixture(scope="function", autouse=True)
def do_init() -> None:
    Config.init()


@pytest.fixture(scope="session", autouse=True)
def clear_db() -> Generator[None, None, None]:
    yield

    db = GraphDB.singleton()
    # delete all test nodes (which may have edges that need to be detached)
    db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
    # delete all nodes without relationships
    db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")


@pytest.fixture
def registered_test_component() -> Generator[tuple[str, str], None, None]:
    n = "foo"
    t = "bar"

    @register_component(n, t)
    class Bar(Component):
        """This is a Bar doc"""

        def shutdown(self) -> None:
            pass

    yield (n, t)

    Component.deregister(n, t)


@pytest.fixture
def fake_component(registered_test_component) -> Generator[Component, None, None]:
    n, t = registered_test_component
    c = Component.get(n, t)

    yield c

    c.shutdown()


@pytest.fixture
def fake_bus() -> EventBus[FakeData]:
    return EventBus[FakeData]("fake")


@pytest.fixture
def empty_components() -> None:
    Component.reset()
    gc.collect(2)


@pytest.fixture
def env_bus_conn(fake_component) -> BusConnection[PerceptionData]:
    return perception_bus.connect(fake_component)


@pytest.fixture
def action_bus_conn(fake_component) -> BusConnection[ActionData]:
    return action_bus.connect(fake_component)


@pytest.fixture
def testing_args(request) -> str:
    print("request", request)
    print("request.param", request.param)
    return f"ret {request.param}"


@pytest.fixture
def component_response(request, mocker, fake_component) -> MagicMock:
    print("request", request)
    # print("component_response args:", request.param)
    component_name, component_type, input_conn_attr, output_conn_attr, val = request.param
    c = Component.get(component_name, component_type)
    print("got component", c.name, c.type)
    if output_conn_attr is None:
        output_conn_attr = input_conn_attr

    assert hasattr(c, input_conn_attr)
    assert hasattr(c, output_conn_attr)

    input_conn = getattr(c, input_conn_attr)
    output_conn = getattr(c, output_conn_attr)
    assert isinstance(input_conn, BusConnection)
    assert isinstance(output_conn, BusConnection)

    fake_conn_send = fake_component.connect_bus(input_conn.attached_bus)
    if input_conn is not output_conn:
        fake_conn_recv = fake_component.connect_bus(output_conn.attached_bus)
    else:
        fake_conn_recv = fake_conn_send

    stub = mocker.stub()
    fake_conn_recv.listen(stub, filter=lambda e: e.data is not val)
    # print("sending", val)
    fake_conn_send.send(val)

    return cast(MagicMock, stub)


def pytest_emoji_passed(config: Any) -> tuple[str, str]:
    return "✅ ", "PASSED ✅ "


def pytest_emoji_failed(config: Any) -> tuple[str, str]:
    return "🔥 ", "FAILED 🔥"


def pytest_emoji_skipped(config: Any) -> tuple[str, str]:
    return "☁️ ", "SKIPPED ☁️ "


def pytest_emoji_error(config: Any) -> tuple[str, str]:
    return "💩 ", "ERROR 💩 "


def pytest_emoji_xfailed(config: Any) -> tuple[str, str]:
    return "⁉️ ", "XFAIL ⁉️ "


def pytest_emoji_xpassed(config: Any) -> tuple[str, str]:
    return "☘️ ", "XPASS ☘️ "
