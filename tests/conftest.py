from types import SimpleNamespace
from typing import Any, Literal, Type

import json
import re
from collections.abc import Iterator
from unittest import mock

import pytest
from helpers.db_record import clear_current_test_record, do_recording, get_query_record

from roc.event import EventBus
from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = False
RECORD_DB = False

if RECORD_DB:
    do_recording()


def mock_raw_query(db: Any, query: str, *, fetch: bool) -> Iterator[Any]:
    return get_query_record(query)


@pytest.fixture
def clear_cache():
    Node.get_cache_control().clear()
    Edge.get_cache_control().clear()


@pytest.fixture
def mock_db(clear_cache):
    if not LIVE_DB:
        with mock.patch.object(GraphDB, "raw_query", new=mock_raw_query):
            yield
    else:
        if RECORD_DB:
            clear_current_test_record()
        yield


@pytest.fixture
def eb_reset():
    EventBus.clear_names()
