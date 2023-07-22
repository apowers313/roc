from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
from helpers.db_record import clear_current_test_record, do_recording, get_query_record

from roc.event import EventBus
from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = False
RECORD_DB = False

if RECORD_DB:
    do_recording()


def mock_raw_fetch(
    db: Any, query: str, *, params: dict[str, Any] | None = None
) -> Iterator[Any] | None:
    return get_query_record(query)


@pytest.fixture
def clear_cache():
    Node.cache_control.clear()
    Edge.cache_control.clear()


@pytest.fixture
def mock_db(clear_cache):
    if not LIVE_DB:
        with mock.patch.object(GraphDB, "raw_fetch", new=mock_raw_fetch):
            yield
    else:
        if RECORD_DB:
            clear_current_test_record()
        yield


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
