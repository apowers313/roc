from typing import Generator

import pytest

from roc.event import EventBus
from roc.graphdb import CacheControl, Edge, GraphDB, Node


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None, None, None]:
    yield

    node_cache = CacheControl.node_cache_control.cache
    edge_cache = CacheControl.edge_cache_control.cache
    for n in node_cache:
        node_cache[n].no_save = True
    for e in edge_cache:
        edge_cache[e].no_save = True

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
