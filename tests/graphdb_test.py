from typing import Any, cast

from collections import namedtuple
from unittest import mock

import pytest
from cachetools import Cache
from helpers.db_data import mock_raw_query
from icecream import ic

from roc.graphdb import Edge, GraphDB, Node

# from gqlalchemy.exceptions import GQLAlchemySubclassNotFoundWarning
# import warnings

# warnings.filterwarnings("ignore", category=GQLAlchemySubclassNotFoundWarning, module="dynconf")
# warnings.filterwarnings("ignore", module="gqlalchemy")


class TestGraphDB:
    @pytest.mark.skip(reason="add assertions")
    def test_graphdb_connect(self, mock_db):
        db = GraphDB()
        res = list(
            db.raw_query(
                """
                MATCH (n)-[e]-(m) WHERE id(n) = 0
                RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
                """,
                fetch=True,
            )
        )
        assert len(res) == 3
        print("!!! RES:", res)
        print("!!! REPR:", repr(res))
        # assert res != None
        for row in res:
            print("!!! ROW:", repr(row))


class TestNode:
    @mock.patch.object(GraphDB, "raw_query", new=mock_raw_query)
    def test_node_get(self):
        n = Node.get(0)
        assert n.id == 0
        assert len(n.src_edges) == 2
        assert len(n.dst_edges) == 1
        assert n.data == {"name": "Waymar Royce"}
        assert n.labels == {"Character"}

    @mock.patch.object(GraphDB, "raw_query", new=mock_raw_query)
    def test_node_cache(self, clear_cache):
        cc = Node.get_cache_control()
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

    @pytest.mark.skip("pending")
    def test_node_save(self):
        pass

    @pytest.mark.skip("pending")
    def test_node_connect(self):
        pass


class TestEdgeList:
    @mock.patch.object(GraphDB, "raw_query", new=mock_raw_query)
    def test_get_edge(self, clear_cache):
        n = Node.get(0)
        e0 = n.src_edges[0]
        e1 = n.src_edges[1]
        e11 = n.dst_edges[0]
        # Edge 0
        assert e0.id == 0
        assert e0.data == {}
        assert e0.type == "LOYAL_TO"
        assert e0.src_id == 0
        assert e0.dst_id == 6
        # Edge 1
        assert e1.id == 1
        assert e1.data == {}
        assert e1.type == "VICTIM_IN"
        assert e1.src_id == 0
        assert e1.dst_id == 453
        # Edge 11
        assert e11.id == 11
        assert e11.data == {"count": 1, "method": "Ice sword"}
        assert e11.type == "KILLED"
        assert e11.src_id == 2
        assert e11.dst_id == 0


class TestEdge:
    def test_edge_cache_control(self, clear_cache):
        cc = Edge.get_cache_control()
        # assert cc.info() == (0, 0, 4096, 0)
        ci = cc.info()
        assert ci.hits == 0
        assert ci.misses == 0
        assert ci.maxsize == 4096
        assert ci.currsize == 0
        assert isinstance(cc.cache, Cache)

    @pytest.mark.skip("pending")
    def test_edge_cache(self):
        pass

    @pytest.mark.skip("pending")
    def test_edge_save(self):
        pass
