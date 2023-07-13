from types import SimpleNamespace
from typing import Any, Literal, Type

import json
import re
from collections.abc import Iterator

import pytest
from helpers.db_data import db_query_mapping, normalize_whitespace

from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = False


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
