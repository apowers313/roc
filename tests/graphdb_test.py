from typing import Any, cast

# from gqlalchemy.exceptions import GQLAlchemySubclassNotFoundWarning
import warnings

# warnings.filterwarnings("ignore", category=GQLAlchemySubclassNotFoundWarning, module="dynconf")
warnings.filterwarnings("ignore", module="gqlalchemy")


from collections import namedtuple

import pytest
from cachetools import Cache
from icecream import ic

from roc.graphdb import Edge, GraphDB, Node


class TestGraphDB:
    # @pytest.mark.skip(reason="skip until mocks are added")
    def test_graphdb_connect(self):
        db = GraphDB()
        res = list(
            db.raw_query(
                """
            MATCH (n)-[e]-(m) WHERE id(n) = 0
            RETURN n, e
            """,
                fetch=True,
            )
        )
        assert len(res) == 3
        # print("!!! RES:", res)
        # print("!!! REPR:", repr(res))
        # assert res != None
        # for row in res:
        #     print("!!! ROW:", repr(row))


class TestNode:
    def test_node_cache(self, clear_cache):
        cc = Node.get_cache_control()
        db = GraphDB()
        assert cc.info().hits == 0
        assert cc.info().misses == 0
        n1 = Node.get(0)
        assert cc.info().hits == 0
        assert cc.info().misses == 1
        n2 = Node.get(0)
        assert cc.info().hits == 1
        assert cc.info().misses == 1
        assert id(n1) == id(n2)

    def test_node_cache_control(self, clear_cache):
        cc = Node.get_cache_control()
        # assert cc.info() == (0, 0, 4096, 0)
        ci = cc.info()
        assert ci.hits == 0
        assert ci.misses == 0
        assert ci.maxsize == 4096
        assert ci.currsize == 0
        assert isinstance(cc.cache, Cache)

    def test_node_save(self):
        pass

    def test_node_connect(self):
        pass


class TestEdge:
    def test_edge_cache_control(clear_cache):
        cc = Edge.get_cache_control()
        # assert cc.info() == (0, 0, 4096, 0)
        ci = cc.info()
        assert ci.hits == 0
        assert ci.misses == 0
        assert ci.maxsize == 4096
        assert ci.currsize == 0
        assert isinstance(cc.cache, Cache)

    def test_edge_save(self):
        pass
