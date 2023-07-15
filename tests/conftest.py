from types import SimpleNamespace
from typing import Any, Literal, Type

import json
import re
from collections.abc import Iterator
from unittest import mock

import pytest
from helpers.db_data import db_query_mapping, normalize_whitespace

from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = True
RECORD_DB = True


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
        if RECORD_DB:
            print("DOING RECORD_DB")

            def _debug_record_graphdb_raw_query(query: str, res: Iterator[Any], tag: str) -> None:
                print("GLOBAL _debug_record_graphdb_raw_query")
                # write json
                # write query and json location to query string
                pass

            GraphDB().set_record_callback(_debug_record_graphdb_raw_query)

            # TODO on exit, save data

        yield
