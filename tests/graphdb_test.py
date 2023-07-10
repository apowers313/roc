from typing import Any, cast

from collections import namedtuple

import pytest
from cachetools import Cache
from icecream import ic

from roc.graphdb import Edge, GraphDB, Node


@pytest.mark.skip(reason="skip until mocks are added")
def test_graphdb_connect():
    db = GraphDB()
    db.connect()
    res = db.raw_query(
        """
        MATCH (n)-[e]-(m) WHERE id(n) = 0
        RETURN n, e
        """,
        fetch=True,
    )
    print("!!! RES:", res)
    print("!!! REPR:", repr(res))
    assert res != None
    for row in res:
        print("!!! ROW:", repr(row))


def test_node_cache_control():
    cc = Node.get_cache_control()
    # assert cc.info() == (0, 0, 4096, 0)
    ci = cc.info()
    assert ci.hits == 0
    assert ci.misses == 0
    assert ci.maxsize == 4096
    assert ci.currsize == 0
    assert isinstance(cc.cache, Cache)


def test_edge_cache_control():
    cc = Edge.get_cache_control()
    # assert cc.info() == (0, 0, 4096, 0)
    ci = cc.info()
    assert ci.hits == 0
    assert ci.misses == 0
    assert ci.maxsize == 4096
    assert ci.currsize == 0
    assert isinstance(cc.cache, Cache)


def test_node_save():
    pass


def test_edge_save():
    pass


def test_node_connect():
    pass
