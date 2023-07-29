import re
from unittest.mock import MagicMock

import pytest
from cachetools import Cache
from helpers.util import normalize_whitespace

from roc.graphdb import Edge, EdgeNotFound, GraphDB, Node, NodeNotFound


class TestGraphDB:
    @pytest.mark.skip(reason="add assertions")
    def test_graphdb_connect(self):
        db = GraphDB()
        res = list(
            db.raw_fetch(
                """
                MATCH (n)-[e]-(m) WHERE id(n) = 0
                RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
                """
            )
        )
        assert len(res) == 3
        print("!!! RES:", res)
        print("!!! REPR:", repr(res))
        # assert res != None
        for row in res:
            print("!!! ROW:", repr(row))

    def test_is_singleton(self):
        db1 = GraphDB()
        db2 = GraphDB()
        assert not db2.port == 1111
        assert id(db1) == id(db2)
        db1.port = 1111
        assert db2.port == 1111

    def test_singleton_doesnt_double_init(self):
        db1 = GraphDB()
        db1.port = 1111
        assert db1.port == 1111
        db2 = GraphDB()
        assert db2.port == 1111

    @pytest.mark.skip("screws up recording tests")
    def test_set_record_callback(self, mocker):
        stub = mocker.stub()
        db = GraphDB()
        db.record_callback = stub
        db.raw_fetch("MATCH (n)-[e]-{m) WHERE id(n) = 0 RETURN n")
        assert stub.call_count == 1

    @pytest.mark.slow
    def test_walk(self):
        cnt = 0
        cache: set[int] = set()
        cc = Node.cache_control
        maxsize = cc.info().maxsize
        if maxsize:
            max = maxsize + 100
        else:
            max = 4196

        def walk_node(id: int) -> None:
            nonlocal cnt, max, cache

            if id in cache:
                return
            else:
                cache.add(id)
            print("walking node", id)
            print(f"*** MAX {cnt}/{max}")

            cnt = cnt + 1
            n = Node.get(id)
            src_edges = n.src_edges
            dst_edges = n.dst_edges
            del n

            for e in src_edges:
                if cnt > max:
                    return

                print(f"+++ id:{id} --> dst:{e.dst.id}")
                walk_node(e.dst.id)

            for e in dst_edges:
                if cnt > max:
                    return

                print(f"--- id{id} <-- {e.src.id}")
                walk_node(e.src.id)

        walk_node(0)
        print("CNT", cnt)
        print("MAX", max)
        print("MAXSIZE", maxsize)
        i = Node.cache_control.info()
        print("HITS", i.hits)
        print("MISSES", i.misses)
        print("CURRENT", i.currsize)
        assert cnt == i.misses
        assert i.currsize == i.maxsize


class TestNode:
    def test_node_get(self):
        n = Node.get(0)
        assert n.id == 0
        assert len(n.src_edges) == 2
        assert len(n.dst_edges) == 1
        assert n.data == {"name": "Waymar Royce"}
        assert n.labels == ["Character"]
        assert not n.new

    def test_node_cache(self):
        cc = Node.cache_control
        assert cc.info().hits == 0
        assert cc.info().misses == 0
        n1 = Node.get(0)
        assert cc.info().hits == 0
        assert cc.info().misses == 1
        n2 = Node.get(0)
        assert cc.info().hits == 1
        assert cc.info().misses == 1
        assert id(n1) == id(n2)

    def test_node_new_in_cache(self, clear_cache):
        n = Node()
        n_dupe = Node.get(n.id)
        assert id(n) == id(n_dupe)

    def test_node_create_on_delete(self, mocker, clear_cache):
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["TestNode"], data={"testname": "test_node_save_on_delete"})

        del n
        Node.cache_control.clear()

        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:TestNode $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {"testname": "test_node_save_on_delete"}}

    def test_node_update_on_delete(self, mocker):
        n = Node(labels=["TestNode"], data={"testname": "test_node_save_on_delete"})
        Node.save(n)
        assert not n.new
        assert n.id > 0
        n.data = {"foo": "bar"}
        n.labels.append("Bob")
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        del n
        Node.cache_control.clear()

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2813 SET n:Bob, n = $props")
        match_str = esc_str.replace("2813", "\d+")
        assert re.search(match_str, spy.call_args[0][1])
        assert spy.call_args[1]["params"] == {"props": {"foo": "bar"}}

    def test_node_cache_control(self, clear_cache):
        cc = Node.cache_control
        # assert cc.info() == (0, 0, 4096, 0)
        ci = cc.info()
        assert ci.hits == 0
        assert ci.misses == 0
        assert ci.maxsize == 2048
        assert ci.currsize == 0
        assert isinstance(cc.cache, Cache)

    @pytest.mark.skip("pending")
    def test_node_save(self):
        pass

    def test_node_new(self):
        n = Node()

        assert n.id < 0
        assert len(n.src_edges) == 0
        assert len(n.dst_edges) == 0
        assert n.data == dict()
        assert n.labels == list()
        assert n.new

    def test_node_create(self, mocker):
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node()
        pre_id = n.id

        Node.create(n)
        assert not n.new
        assert n.id != pre_id
        spy.assert_called_once()

        assert spy.call_args[0][1] == "CREATE (n $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {}}
        # MATCH (n) WHERE size(labels(n)) = 0 DELETE n

    def test_node_create_with_label(self, mocker):
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["Foo"])

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Foo $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {}}
        # MATCH (n:Foo) DELETE n

    def test_node_create_with_multiple_labels(self, mocker):
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["Foo", "Bar"])

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Foo:Bar $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {}}

    def test_node_create_with_data(self, mocker):
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["Foo"], data={"answer": 42})

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Foo $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {"answer": 42}}

    def test_node_update(self, mocker):
        n = Node.create(Node())
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.append("TestNode")
        n.data = {"beer": "yum", "number": 42}
        Node.update(n)

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2746 SET n:TestNode, n = $props ")
        match_str = esc_str.replace("2746", "\d+")
        assert re.search(match_str, spy.call_args[0][1])
        assert spy.call_args[1]["params"] == {"props": {"beer": "yum", "number": 42}}

    def test_node_update_add_label(self, mocker):
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.append("Foo")
        Node.update(n)

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2746 SET n:Foo, n = $props ")
        match_str = esc_str.replace("2746", "\d+")
        assert re.search(match_str, spy.call_args[0][1])

    def test_node_update_remove_label(self, mocker):
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.clear()
        Node.update(n)

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2746 SET n = $props REMOVE n:TestNode")
        match_str = esc_str.replace("2746", "\d+")
        assert re.search(match_str, spy.call_args[0][1])

    def test_node_update_add_and_remove_label(self, mocker):
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.clear()
        n.labels.append("Foo")
        Node.update(n)

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2746 SET n:Foo, n = $props REMOVE n:TestNode")
        match_str = esc_str.replace("2746", "\d+")
        assert re.search(match_str, spy.call_args[0][1])

    def test_node_update_properties(self, mocker):
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.data = {"foo": "bar", "baz": "bat"}
        Node.update(n)

        spy.assert_called_once()
        esc_str = re.escape("MATCH (n) WHERE id(n) = 2746 SET n = $props ")
        match_str = esc_str.replace("2746", "\d+")
        assert re.search(match_str, spy.call_args[0][1])
        assert spy.call_args[1]["params"] == {"props": {"foo": "bar", "baz": "bat"}}

    def test_node_connect(self):
        n1 = Node()
        n2 = Node()

        e = Node.connect(n1, n2, "Test")

        assert len(n1.src_edges) == 1
        assert len(n1.dst_edges) == 0
        assert e in n1.src_edges
        assert len(n2.src_edges) == 0
        assert len(n2.dst_edges) == 1
        assert e in n2.dst_edges
        assert e.src_id == n1.id
        assert e.dst_id == n2.id

    def test_node_create_updates_edge_src(self):
        n1 = Node(labels=["TestNode"])
        old_id = n1.id
        n2 = Node(labels=["TestNode"])
        e = Node.connect(n1, n2, "Test")
        Node.save(n1)

        # edge is still new
        assert e.new
        assert e.id < 0
        # but src node id has been updated
        assert old_id != n1.id
        # and edge src has been updated
        assert e.src_id == n1.id

    def test_node_create_updates_edge_dst(self):
        n1 = Node(labels=["TestNode"])
        n2 = Node(labels=["TestNode"])

        n2_id = n2.id
        e = Node.connect(n1, n2, "Test")
        Node.save(n2)

        # edge is still new
        assert e.new
        assert e.id < 0
        # but src node id has been updated
        assert n2_id != n2.id
        # and edge dst has been updated
        assert e.dst_id == n2.id

    def test_node_create_updates_cache(self):
        cc = Node.cache_control
        n = Node(labels=["TestNode"])
        old_id = n.id

        Node.create(n)

        assert cc.info().hits == 0
        assert cc.info().misses == 0
        # old ID doesn't exist in cache
        with pytest.raises(NodeNotFound):
            Node.get(old_id)
        assert cc.info().hits == 0
        assert cc.info().misses == 1
        # new ID does exist in cache
        Node.get(n.id)
        assert cc.info().hits == 1
        assert cc.info().misses == 1


class TestEdgeList:
    def test_get_edge(self):
        n = Node.get(0)
        e0 = n.src_edges[0]
        e1 = n.src_edges[1]
        e11 = n.dst_edges[0]
        assert isinstance(e0, Edge)
        assert isinstance(e1, Edge)
        assert isinstance(e11, Edge)
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

    def test_iter(self):
        n = Node.get(0)
        for e in n.src_edges:
            assert isinstance(e, Edge)

    # test_add
    # test_add_duplicate
    # test_discard

    def test_contains(self):
        n = Node.get(0)
        e = n.src_edges[0]

        assert e in n.src_edges
        assert e.id in n.src_edges
        assert "bob" not in n.src_edges  # type: ignore


class TestEdge:
    def test_edge_cache_control(self):
        cc = Edge.cache_control
        # assert cc.info() == (0, 0, 4096, 0)
        ci = cc.info()
        assert ci.hits == 0
        assert ci.misses == 0
        assert ci.maxsize == 2048
        assert ci.currsize == 0
        assert isinstance(cc.cache, Cache)

    def test_src(self):
        n0 = Node.get(0)
        n2 = Node.get(2)
        e0 = n0.src_edges[0]
        e1 = n0.src_edges[1]
        e11 = n0.dst_edges[0]
        assert isinstance(e0.src, Node)
        assert isinstance(e1.src, Node)
        assert isinstance(e11.src, Node)
        assert id(e0.src) == id(n0)
        assert id(e1.src) == id(n0)
        assert id(e11.src) == id(n2)

    def test_dst(self):
        n0 = Node.get(0)
        n6 = Node.get(6)
        n453 = Node.get(453)
        e0 = n0.src_edges[0]
        e1 = n0.src_edges[1]
        e11 = n0.dst_edges[0]
        assert isinstance(e0.dst, Node)
        assert isinstance(e1.dst, Node)
        assert isinstance(e11.dst, Node)
        assert id(e11.dst) == id(n0)
        assert id(e0.dst) == id(n6)
        assert id(e1.dst) == id(n453)

    def test_edge_create(self, mocker):
        e = Edge.create(Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test"))
        e_id = e.id

        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        e = Edge.create(e)

        # saved both new nodes
        assert not e.src.new
        assert not e.dst.new
        assert not e.new
        # created an ID
        assert e.id != e_id
        assert e.id > 0

        assert spy.call_count == 1
        query = normalize_whitespace(spy.call_args[0][1])
        esc_str = re.escape(
            "MATCH (src), (dst) WHERE id(src) = 3102 AND id(dst) = 3103 CREATE (src)-[e:Test $props]->(dst) RETURN id"  # noqa: E501
        )
        match_str = esc_str.replace("3102", "\d+")
        match_str = match_str.replace("3103", "\d+")
        assert re.search(match_str, query)

    def test_edge_create_with_data(self, mocker):
        e = Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test")
        e_id = e.id
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        e.data = {"name": "bob", "fun": False}
        e = Edge.create(e)

        # saved both new nodes
        assert not e.src.new
        assert not e.dst.new
        assert not e.new
        # created an ID
        assert e.id != e_id
        assert e.id > 0

        assert spy.call_count == 3
        query = normalize_whitespace(spy.call_args[0][1])
        esc_str = re.escape(
            "MATCH (src), (dst) WHERE id(src) = 3102 AND id(dst) = 3103 CREATE (src)-[e:Test $props]->(dst) RETURN id"  # noqa: E501
        )
        match_str = esc_str.replace("3102", "\d+")
        match_str = match_str.replace("3103", "\d+")
        assert re.search(match_str, query)
        assert spy.call_args[1]["params"] == {"props": {"name": "bob", "fun": False}}

    def test_edge_create_updates_cache(self, clear_cache):
        cc = Edge.cache_control
        assert cc.info().hits == 0
        assert cc.info().misses == 0
        e = Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test")
        old_id = e.id

        Edge.create(e)

        # NOTE: Node.create() is called by Edge.create() if the nodes are new
        # Node.create() updates the edge.src and edge.dst, so it hits the cache twice
        assert cc.info().hits == 2
        assert cc.info().misses == 0
        # old ID doesn't exist in cache
        with pytest.raises(EdgeNotFound):
            Edge.get(old_id)
        assert cc.info().hits == 2
        assert cc.info().misses == 1
        # new ID does exist in cache
        Edge.get(e.id)
        assert cc.info().hits == 3
        assert cc.info().misses == 1

    def test_edge_create_updates_node_edges(self):
        src_n = Node(labels=["TestNode"])
        dst_n = Node(labels=["TestNode"])
        e = Node.connect(src_n, dst_n, "Test")
        old_id = e.id
        e.data = {"name": "bob", "fun": False}

        e = Edge.create(e)

        assert e.id != old_id
        assert e.id in src_n.src_edges
        assert e.id in dst_n.dst_edges

    def test_edge_immutable_properties(self):
        e = Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test")
        # orig_src_id = e.src_id
        # orig_dst_id = e.dst_id
        orig_src = e.src
        orig_dst = e.dst
        orig_type = e.type

        # e.src_id = -666
        # e.dst_id = -666
        with pytest.raises(AttributeError):
            e.src = Node()  # type: ignore
        with pytest.raises(AttributeError):
            e.dst = Node()  # type: ignore
        with pytest.raises(AttributeError):
            e.type = "jackedup"  # type: ignore

        # assert e.src_id == orig_src_id
        # assert e.dst_id == orig_dst_id
        assert id(e.src) == id(orig_src)
        assert id(e.dst) == id(orig_dst)
        assert e.type == orig_type

    # test_edge_update
    def test_edge_update(self, mocker):
        e = Edge.create(Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test"))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        e.data = {"wine": "cab", "more": True}
        Edge.update(e)

        spy.assert_called_once()
        esc_str = re.escape("MATCH ()-[e]->() WHERE id(e) = 2746 SET e = $props")
        match_str = esc_str.replace("2746", "\d+")
        print("CALL:", spy.call_args[0][1])
        assert re.search(match_str, spy.call_args[0][1])
        assert spy.call_args[1]["params"] == {"props": {"wine": "cab", "more": True}}

    def test_edge_src_after_node_save(self):
        src = Node(labels=["TestNode"])
        dst = Node(labels=["TestNode"])
        old_id = src.id
        e = Edge.create(Node.connect(src, dst, "Test"))

        Node.save(src)

        assert src.id != old_id
        assert e.src_id == src.id
        assert id(e.src) == id(src)

    def test_edge_dst_after_node_save(self):
        src = Node(labels=["TestNode"])
        dst = Node(labels=["TestNode"])
        old_id = dst.id
        e = Edge.create(Node.connect(src, dst, "Test"))

        Node.save(dst)

        assert dst.id != old_id
        assert e.dst_id == dst.id
        assert id(e.dst) == id(dst)

    @pytest.mark.skip("pending")
    def test_edge_cache(self):
        pass

    @pytest.mark.skip("pending")
    def test_edge_save(self):
        pass
