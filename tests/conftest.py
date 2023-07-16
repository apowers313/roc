from types import SimpleNamespace
from typing import Any, Literal, Type

import json
import re
from collections.abc import Iterator
from unittest import mock

import pytest
from helpers.db_data import db_query_mapping, normalize_whitespace
from helpers.db_record import do_recording

from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = False
RECORD_DB = False

if RECORD_DB:
    do_recording()


def mock_raw_query(db: Any, query: str, *, fetch: bool) -> Iterator[Any]:
    query = normalize_whitespace(query)
    print(f"\nmock raw query: '{query}'")

    try:
        # ret = db_query_mapping[query]()
        # print("returning", list(ret))
        return db_query_mapping[query]()
    except KeyError:
        raise NotImplementedError(f"mock raw query not implemented: '{query}'")


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
        yield
