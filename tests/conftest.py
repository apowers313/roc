# mypy: disable-error-code="no-untyped-def"

import gc
import os
from importlib import import_module
from typing import Any, Generator

import psutil
import pytest
from helpers.util import FakeData

from roc.action import Action, ActionData
from roc.component import Component, component_set
from roc.config import Config
from roc.event import BusConnection, EventBus
from roc.graphdb import Edge, GraphDB, Node
from roc.logger import init as logger_init
from roc.perception import Perception, PerceptionData
from roc.reporting.observability import Observability


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


@pytest.fixture(autouse=True)
def restore_registries() -> Generator[None, None, None]:
    from roc.graphdb import edge_registry, node_label_registry, node_registry

    orig_node_registry = node_registry.copy()
    orig_node_label_registry = node_label_registry.copy()
    orig_edge_registry = edge_registry.copy()

    yield

    node_registry.clear()
    for nk, nv in orig_node_registry.items():
        node_registry[nk] = nv

    node_label_registry.clear()
    for nlk, nlv in orig_node_label_registry.items():
        node_label_registry[nlk] = nlv

    edge_registry.clear()
    for ek, ev in orig_edge_registry.items():
        edge_registry[ek] = ev


@pytest.fixture
def clear_registries(restore_registries) -> None:
    from roc.graphdb import edge_registry, node_label_registry, node_registry

    edge_registry.clear()
    node_registry.clear()
    node_label_registry.clear()


@pytest.fixture
def strict_schema() -> Generator[None, None, None]:
    settings = Config.get()
    db = GraphDB.singleton()
    orig = db.strict_schema
    settings.db_strict_schema = db.strict_schema = True

    yield

    settings.db_strict_schema = db.strict_schema = orig


@pytest.fixture
def no_strict_schema() -> Generator[None, None, None]:
    settings = Config.get()
    db = GraphDB.singleton()
    orig = db.strict_schema
    settings.db_strict_schema = db.strict_schema = False

    yield

    settings.db_strict_schema = db.strict_schema = orig


@pytest.fixture
def strict_schema_warns() -> Generator[None, None, None]:
    settings = Config.get()
    db = GraphDB.singleton()
    orig = db.strict_schema_warns
    settings.db_strict_schema_warns = db.strict_schema_warns = True

    yield

    settings.db_strict_schema_warns = db.strict_schema_warns = orig


@pytest.fixture
def new_edge(no_strict_schema) -> tuple[Edge, Node, Node]:
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
    Observability.init()
    settings = Config.get()
    settings.observation_shape = (21, 79)
    settings.gym_actions = (
        ord("j"),
        ord("k"),
        ord("h"),
        ord("l"),
        ord("y"),
        ord("u"),
        ord("b"),
        ord("n"),
        ord("e"),
        ord("."),
    )
    logger_init()

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

    class Bar(Component):
        """This is a Bar doc"""

        name: str = n
        type: str = t

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
    if Component.get_component_count() != 0:
        for c in component_set:
            print("ERROR: loaded component:", c)  # noqa: T201
            print("referred to by:", gc.get_referrers(c))  # noqa: T201
    assert Component.get_component_count() == 0


@pytest.fixture
def env_bus_conn(fake_component) -> BusConnection[PerceptionData]:
    return Perception.bus.connect(fake_component)


@pytest.fixture
def action_bus_conn(fake_component) -> BusConnection[ActionData]:
    return Action.bus.connect(fake_component)


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


@pytest.fixture
def test_tree(no_strict_schema) -> dict[str, Any]:
    root = Node(labels=["TestNode"])
    n1 = Node(labels=["TestNode"])
    n2 = Node(labels=["TestNode"])
    n3 = Node(labels=["TestNode"])
    n4 = Node(labels=["TestNode"])
    n5 = Node(labels=["TestNode"])
    n6 = Node(labels=["TestNode"])
    n7 = Node(labels=["TestNode"])
    n8 = Node(labels=["TestNode"])
    n9 = Node(labels=["TestNode"])
    n10 = Node(labels=["TestNode"])
    e1 = Node.connect(root, n1, "Foo")
    e2 = Node.connect(root, n2, "Test")
    e3 = Node.connect(n2, n3, "Test")
    e4 = Node.connect(n2, n4, "Test")
    e5 = Node.connect(n5, n2, "Foo")
    e6 = Node.connect(n6, root, "Foo")
    e7 = Node.connect(n6, n7, "Test")
    e8 = Node.connect(n8, n6, "Test")
    e9 = Node.connect(n9, n6, "Test")
    e10 = Node.connect(n10, root, "Test")

    return {
        "root": root,
        "nodes": [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10],
        "edges": [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10],
    }


@pytest.fixture
def memory_profile() -> Generator[None, None, None]:
    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    start_mem = process_memory()
    print(f"Starting memory: {start_mem / 1000000:1.3f} MB")  # noqa: T201
    print("Node Cache:", Node.get_cache())  # noqa: T201
    print("Edge Cache:", Edge.get_cache())  # noqa: T201

    yield

    end_mem = process_memory()
    print(  # noqa: T201
        f"Ending memory: {end_mem / 1000000:1.3f} MB  [{(end_mem - start_mem) / 1000000:1.3f} MB]"
    )
    print("Node Cache:", Node.get_cache())  # noqa: T201
    print("Edge Cache:", Edge.get_cache())  # noqa: T201


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
