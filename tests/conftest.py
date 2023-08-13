from typing import Any, Generator

import pytest

from roc.event import EventBus
from roc.graphdb import CacheControl, Edge, GraphDB, Node


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None, None, None]:
    yield

    node_cache = CacheControl.node_cache_control.cache
    edge_cache = CacheControl.edge_cache_control.cache
    for n in node_cache:
        node_cache[n]._no_save = True
    for e in edge_cache:
        edge_cache[e]._no_save = True

    CacheControl.node_cache_control.clear()
    CacheControl.edge_cache_control.clear()


@pytest.fixture
def new_edge() -> tuple[Edge, Node, Node]:
    src = Node(labels=["TestNode"])
    dst = Node(labels=["TestNode"])
    e = Node.connect(src, dst, "Test")
    return (e, src, dst)


@pytest.fixture
def eb_reset() -> None:
    EventBus.clear_names()


@pytest.fixture(scope="session", autouse=True)
def clear_db() -> Generator[None, None, None]:
    yield

    db = GraphDB()
    # delete all test nodes (which may have edges that need to be detached)
    db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
    # delete all nodes without relationships
    db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")


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


# def pytest_emoji_passed(config: Any) -> tuple[str, str]:
#     return "\U00002705 ", "PASSED \U00002705 "

# def pytest_emoji_failed(config: Any) -> tuple[str, str]:
#     return "\U0001F525 ", "FAILED \U0001F525"

# def pytest_emoji_skipped(config: Any) -> tuple[str, str]:
#     return "\U00002601 ", "SKIPPED \U00002601 "

# def pytest_emoji_error(config: Any) -> tuple[str, str]:
#     return "\U0001F4A9 ", "ERROR \U0001F4A9 "

# def pytest_emoji_xfailed(config: Any) -> tuple[str, str]:
#     return "⁉\U00002049 ", "XFAIL \U00002049 "

# def pytest_emoji_xpassed(config: Any) -> tuple[str, str]:
#     return "\U00002618 ", "XPASS \U00002618 "
