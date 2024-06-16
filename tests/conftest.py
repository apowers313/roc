# mypy: disable-error-code="no-untyped-def"
import gc
from importlib import import_module
from typing import Any, Generator

import pytest
from helpers.util import FakeData

import roc.logger as logger
from roc.action import ActionData, action_bus
from roc.component import Component, register_component
from roc.config import Config
from roc.event import BusConnection, EventBus
from roc.graphdb import Edge, GraphDB, Node
from roc.perception import Perception, PerceptionData


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
def do_init() -> Generator[None, None, None]:
    Config.reset()
    Config.init()
    logger.init()

    yield

    # cleanup for clear_db fixture
    Config.reset()
    Config.init()


@pytest.fixture(scope="session", autouse=True)
def close_db() -> Generator[None, None, None]:
    """Closes the graph database and deletes all data that was created by
    tests"""

    yield

    db = GraphDB.singleton()
    # delete all test nodes (which may have edges that need to be detached)
    db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
    # delete all nodes without relationships
    db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")

    db.close()


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
def empty_components() -> Generator[None, None, None]:
    Component.reset()
    gc.collect(2)
    assert Component.get_component_count() == 0

    yield

    Component.reset()
    gc.collect(2)
    assert Component.get_component_count() == 0


@pytest.fixture
def env_bus_conn(fake_component) -> BusConnection[PerceptionData]:
    return Perception.bus.connect(fake_component)


@pytest.fixture
def action_bus_conn(fake_component) -> BusConnection[ActionData]:
    return action_bus.connect(fake_component)


@pytest.fixture
def testing_args(request) -> str:
    return f"ret {request.param}"


@pytest.fixture
def requires_module(request) -> None:
    """loads Python modules for their side-effects"""

    mods: list[str] | str = request.param
    if isinstance(mods, str):
        mods = [mods]

    for mod in mods:
        import_module(mod)


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
