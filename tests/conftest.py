from typing import Any, Generator

import pytest

from roc.action import ActionData, action_bus
from roc.component import Component, register_component
from roc.config import load_config
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
    load_config()


@pytest.fixture(scope="session", autouse=True)
def clear_db() -> Generator[None, None, None]:
    yield

    db = GraphDB.singleton()
    # delete all test nodes (which may have edges that need to be detached)
    db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
    # delete all nodes without relationships
    db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")


@pytest.fixture(scope="function", autouse=True)
def clear_components() -> None:
    Component.clear_registry()


@pytest.fixture
def fake_component() -> Component:
    @register_component("fake", "thing")
    class FakeComponent(Component):
        pass

    return Component.get("fake", "thing")


@pytest.fixture
def env_bus_conn() -> BusConnection[PerceptionData]:
    c = Component()
    return perception_bus.connect(c)


@pytest.fixture
def action_bus_conn() -> BusConnection[ActionData]:
    c = Component()
    return action_bus.connect(c)


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
