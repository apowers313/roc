import pytest

from roc.event import EventBus
from roc.graphdb import Edge, GraphDB, Node


@pytest.fixture(autouse=True)
def clear_cache():
    yield

    node_cache = Node.cache_control.cache
    edge_cache = Edge.cache_control.cache
    for n in node_cache:
        node_cache[n].no_save = True
    for e in edge_cache:
        edge_cache[e].no_save = True

    Node.cache_control.clear()
    Edge.cache_control.clear()


@pytest.fixture
def eb_reset():
    EventBus.clear_names()


@pytest.fixture(scope="session", autouse=True)
def clear_db():
    yield

    db = GraphDB()
    # delete all test nodes (which may have edges that need to be detached)
    db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
    # delete all nodes without relationships
    db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")
