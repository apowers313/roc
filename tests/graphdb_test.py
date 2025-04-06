# mypy: disable-error-code="no-untyped-def"

from abc import ABC, abstractmethod
from typing import cast
from unittest.mock import MagicMock

import pytest
from cachetools import Cache
from helpers.dot import dot_node1, dot_schema1
from helpers.mermaid import mermaid_schema1
from helpers.schema import GotCharacter, GotSeason
from helpers.util import assert_similar, normalize_whitespace
from pydantic import Field, ValidationError

from roc.graphdb import (
    Edge,
    EdgeConnectionsList,
    EdgeId,
    EdgeList,
    EdgeNotFound,
    GraphDB,
    Node,
    NodeId,
    NodeList,
    NodeNotFound,
    Schema,
    SchemaValidationError,
    StrictSchemaWarning,
    _EdgeDescription,
    _NodeDescription,
    edge_registry,
    node_label_registry,
    node_registry,
)


class TestGraphDB:
    @pytest.mark.skip(reason="add assertions")
    def test_graphdb_connect(self) -> None:
        db = GraphDB.singleton()
        res = list(
            db.raw_fetch(
                """
                MATCH (n)-[e]-(m) WHERE id(n) = 0
                RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
                """
            )
        )
        assert len(res) == 3
        print("!!! RES:", res)  # noqa: T201
        print("!!! REPR:", repr(res))  # noqa: T201
        for row in res:
            print("!!! ROW:", repr(row))  # noqa: T201

    def test_singleton(self) -> None:
        db1 = GraphDB.singleton()
        db2 = GraphDB.singleton()
        assert not db2.port == 1111
        assert id(db1) == id(db2)
        db1.port = 1111
        assert db2.port == 1111

    def test_singleton_doesnt_double_init(self) -> None:
        db1 = GraphDB.singleton()
        db1.port = 1111
        assert db1.port == 1111
        db2 = GraphDB.singleton()
        assert db2.port == 1111

    @pytest.mark.slow
    def test_walk(self) -> None:
        cnt = 0
        cache: set[int] = set()
        c = Node.get_cache()
        maxsize = c.maxsize
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
            print("walking node", id)  # noqa: T201
            print(f"*** MAX {cnt}/{max}")  # noqa: T201

            cnt = cnt + 1
            n = Node.get(cast(NodeId, id))
            src_edges = n.src_edges
            dst_edges = n.dst_edges
            del n

            for e in src_edges:
                if cnt > max:
                    return

                print(f"+++ id:{id} --> dst:{e.dst.id}")  # noqa: T201
                walk_node(e.dst.id)

            for e in dst_edges:
                if cnt > max:
                    return

                print(f"--- id{id} <-- {e.src.id}")  # noqa: T201
                walk_node(e.src.id)

        walk_node(0)
        print("CNT", cnt)  # noqa: T201
        print("MAX", max)  # noqa: T201
        print("MAXSIZE", maxsize)  # noqa: T201
        c = Node.get_cache()
        print("HITS", c.hits)  # noqa: T201
        print("MISSES", c.misses)  # noqa: T201
        print("CURRENT", c.currsize)  # noqa: T201
        assert cnt == c.misses
        assert c.currsize == c.maxsize

    @pytest.mark.slow
    def test_to_networkx(self) -> None:
        node_ids = Node.all_ids()
        G = GraphDB.to_networkx(node_ids=node_ids)
        assert G
        assert len(node_ids) == G.number_of_nodes()


class TestNode:
    def test_node_get(self) -> None:
        n = Node.get(cast(NodeId, 0))
        assert n.id == 0
        assert len(n.src_edges) == 2
        assert len(n.dst_edges) == 1
        assert Node.to_dict(n) == {"name": "Waymar Royce"}
        assert n.labels == {"Character"}
        assert not n.new
        assert n.id in Node.get_cache()

    def test_node_get_many(self) -> None:
        node_ids = {
            cast(NodeId, 0),
            cast(NodeId, 1),
            cast(NodeId, 2),
            cast(NodeId, 3),
            cast(NodeId, 4),
        }
        nodes = Node.get_many(node_ids)
        assert len(nodes) == len(node_ids)
        c = Node.get_cache()
        for n in nodes:
            assert n.id in c

    def test_node_get_many_with_edges(self) -> None:
        node_ids = {
            cast(NodeId, 0),
            cast(NodeId, 1),
            cast(NodeId, 2),
            cast(NodeId, 3),
            cast(NodeId, 4),
        }
        nodes = Node.get_many(node_ids, load_edges=True)
        assert len(nodes) == len(node_ids)
        nc = Node.get_cache()
        ec = Edge.get_cache()
        for n in nodes:
            assert n.id in nc
            for e in n.src_edges:
                assert e.id in ec
            for e in n.dst_edges:
                assert e.id in ec

    def test_node_find(self) -> None:
        node_cache = Node.get_cache()
        edge_cache = Edge.get_cache()
        assert 4 not in node_cache
        assert len(edge_cache) == 0

        nodes = Node.find("src.name = 'Winter Is Coming'")
        assert len(nodes) == 1
        assert nodes[0].id == 4
        assert nodes[0].name == "Winter Is Coming"  # type: ignore
        assert 4 in node_cache
        assert len(node_cache) == 1
        assert len(edge_cache) == 0

    def test_node_find_with_params(self) -> None:
        node_cache = Node.get_cache()
        edge_cache = Edge.get_cache()
        assert 4 not in node_cache
        assert len(edge_cache) == 0

        nodes = Node.find("src.name = $title", params={"title": "Winter Is Coming"})
        assert len(nodes) == 1
        assert nodes[0].id == 4
        assert nodes[0].name == "Winter Is Coming"  # type: ignore
        assert 4 in node_cache
        assert len(node_cache) == 1
        assert len(edge_cache) == 0

    def test_node_find_with_alt_names(self) -> None:
        cache = Node.get_cache()
        assert 4 not in cache

        nodes = Node.find(
            "n.name = 'Zalla' AND type(e) = 'LOYAL_TO'",
            src_node_name="n",
            edge_name="e",
        )
        assert len(nodes) == 1
        assert nodes[0].id == 295
        assert nodes[0].name == "Zalla"  # type: ignore

    def test_node_find_not_found(self) -> None:
        cache = Node.get_cache()
        assert len(cache) == 0

        nodes = Node.find("src.name = 'asdfasdfasdfasdfasdfasdf'")
        assert len(nodes) == 0
        assert len(cache) == 0

    def test_node_find_label(self) -> None:
        cache = Node.get_cache()
        assert len(cache) == 0

        nodes = Node.find("src.name =~ 'B.*'", src_labels={"Location"})
        assert len(nodes) == 2
        assert set([n.id for n in nodes]) == {1, 310}

    def test_node_find_single_node_with_no_relationships(self) -> None:
        cache = Node.get_cache()
        n = Node(
            labels=["TestNode", "Foo"], testname="test_node_find_single_node_with_no_relationships"
        )
        Node.save(n)
        old_id = n.id
        cache.clear()
        assert len(cache) == 0

        n2 = Node.get(old_id)
        assert n2.id == old_id
        assert n2.testname == "test_node_find_single_node_with_no_relationships"  # type: ignore
        assert n2.labels == {"TestNode", "Foo"}

    def test_node_find_multiple_labels(self) -> None:
        cache = Node.get_cache()
        n = Node(labels=["TestNode", "Foo"], testname="test_node_find_multiple_labels")
        Node.save(n)
        old_id = n.id
        cache.clear()
        assert len(cache) == 0

        nodes = Node.find(
            "src.testname = 'test_node_find_multiple_labels'", src_labels={"TestNode", "Foo"}
        )
        assert len(nodes) == 1
        assert nodes[0].id == old_id
        assert nodes[0].testname == "test_node_find_multiple_labels"  # type: ignore
        assert nodes[0].labels == {"TestNode", "Foo"}

    def test_node_find_edge_type(self) -> None:
        cache = Node.get_cache()
        assert len(cache) == 0

        nodes = Node.find("src.name =~ 'Z.*'", edge_type="VICTIM_IN")
        assert len(nodes) == 1
        assert nodes[0].id == 295

    def test_node_find_cached(self) -> None:
        cache = Node.get_cache()
        assert 4 not in cache
        n1 = Node.get(NodeId(4))
        assert 4 in cache

        nodes = Node.find("src.name = 'Winter Is Coming'")
        assert len(nodes) == 1
        assert nodes[0].id == 4
        assert 4 in cache
        assert len(cache) == 1
        assert n1 is nodes[0]

    def test_node_find_adds_to_cache(self) -> None:
        cache = Node.get_cache()
        assert 4 not in cache
        nodes = Node.find("src.name = 'Winter Is Coming'")
        assert 4 in cache

        n1 = Node.get(NodeId(4))
        assert 4 in cache
        assert len(cache) == 1
        assert n1 is nodes[0]

    def test_node_find_multiple_some_cached(self) -> None:
        cache = Node.get_cache()
        assert 4 not in cache
        assert len(cache) == 0
        n1 = Node.get(NodeId(349))
        assert 349 in cache
        n2 = Node.get(NodeId(266))
        assert 266 in cache
        n3 = Node.get(NodeId(362))
        assert 362 in cache
        assert len(cache) == 3

        nodes = Node.find("src.name =~ 'E.*'")
        assert len(nodes) == 5
        assert len(cache) == 5
        node_ids = {n.id for n in nodes}
        assert node_ids == {425, 349, 266, 362, 37}
        for nid in node_ids:
            assert nid in cache
        new_n1 = next((n for n in nodes if n.id == 349), None)
        new_n2 = next((n for n in nodes if n.id == 266), None)
        new_n3 = next((n for n in nodes if n.id == 362), None)
        assert new_n1 is n1
        assert new_n2 is n2
        assert new_n3 is n3

    def test_node_find_and_load_edges(self) -> None:
        node_cache = Node.get_cache()
        edge_cache = Edge.get_cache()
        assert len(node_cache) == 0
        assert len(edge_cache) == 0

        nodes = Node.find("src.name = 'Winter Is Coming'", load_edges=True)
        assert len(nodes) == 1
        assert len(node_cache) == 1
        assert nodes[0].id == 4
        assert len(node_cache) == 1
        assert len(edge_cache) == 8
        src_edge_ids = {e.id for e in nodes[0].src_edges}
        assert src_edge_ids == {17}
        dst_edge_ids = {e.id for e in nodes[0].dst_edges}
        assert dst_edge_ids == {5295, 5298, 5301, 5304, 5307, 5310, 5313}

    def test_node_find_and_load_edges_with_different_edge_name(self) -> None:
        node_cache = Node.get_cache()
        edge_cache = Edge.get_cache()
        assert len(node_cache) == 0
        assert len(edge_cache) == 0

        nodes = Node.find("src.name = 'Winter Is Coming'", load_edges=True, edge_name="bob")
        assert len(nodes) == 1
        assert len(node_cache) == 1
        assert nodes[0].id == 4
        assert len(node_cache) == 1
        assert len(edge_cache) == 8
        src_edge_ids = {e.id for e in nodes[0].src_edges}
        assert src_edge_ids == {17}
        dst_edge_ids = {e.id for e in nodes[0].dst_edges}
        assert dst_edge_ids == {5295, 5298, 5301, 5304, 5307, 5310, 5313}

    # TODO: find node with one cached edge and one uncached edge, and load edges

    def test_node_cache(self) -> None:
        c = Node.get_cache()
        assert c.hits == 0
        assert c.misses == 0
        n1 = Node.get(cast(NodeId, 0))
        assert c.hits == 0
        assert c.misses == 1
        n2 = Node.get(cast(NodeId, 0))
        assert c.hits == 1
        assert c.misses == 1
        assert id(n1) == id(n2)

    def test_node_new_in_cache(self, clear_cache) -> None:
        n = Node()
        n_dupe = Node.get(n.id)
        assert id(n) == id(n_dupe)

    def test_node_create_on_delete(self, mocker, clear_cache) -> None:
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["TestNode"], testname="test_node_save_on_delete")

        del n
        Node.get_cache().clear()

        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:TestNode $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {"testname": "test_node_save_on_delete"}}

    def test_node_update_on_delete(self, mocker) -> None:
        n = Node(labels={"TestNode"}, testname="test_node_update_on_delete")
        Node.save(n)
        assert not n.new
        assert n.id > 0
        n.foo = "bar"  # type: ignore
        del n.testname  # type: ignore
        n.labels.add("Bob")
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        del n
        Node.get_cache().clear()

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2813 SET n:Bob, n = $props",
            spy.call_args[0][1],
            [("2813", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"foo": "bar"}}

    def test_node_cache_control(self, clear_cache) -> None:
        c = Node.get_cache()
        assert c.hits == 0
        assert c.misses == 0
        assert c.currsize == 0
        assert isinstance(c, Cache)

    @pytest.mark.skip("pending")
    def test_node_save(self) -> None:
        pass

    def test_node_new(self) -> None:
        n = Node()

        assert n.id < 0
        assert len(n.src_edges) == 0
        assert len(n.dst_edges) == 0
        assert Node.to_dict(n) == dict()
        assert n.labels == set()
        assert n.new

    def test_node_create(self, mocker) -> None:
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

    def test_node_create_with_label(self, mocker) -> None:
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["Foo"])

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Foo $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {}}
        # MATCH (n:Foo) DELETE n

    def test_node_create_with_multiple_labels(self, mocker) -> None:
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels={"Foo", "Bar"})

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Bar:Foo $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {}}

    def test_node_create_with_data(self, mocker) -> None:
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        n = Node(labels=["Foo"], answer=42)

        Node.create(n)
        spy.assert_called_once()
        assert spy.call_args[0][1] == "CREATE (n:Foo $props) RETURN id(n) as id"
        assert spy.call_args[1]["params"] == {"props": {"answer": 42}}

    def test_node_update(self, mocker) -> None:
        n = Node.create(Node())
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.add("TestNode")
        n.beer = "yum"  # type: ignore
        n.number = 42  # type: ignore
        Node.update(n)

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 SET n:TestNode, n = $props ",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"beer": "yum", "number": 42}}

    def test_node_update_add_label(self, mocker) -> None:
        n = Node.create(Node(labels={"TestNode"}))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.add("Foo")
        Node.update(n)

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 SET n:Foo, n = $props ",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )

    def test_node_update_remove_label(self, mocker) -> None:
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.clear()
        Node.update(n)

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 SET n = $props REMOVE n:TestNode",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )

    def test_node_update_add_and_remove_label(self, mocker) -> None:
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.labels.clear()
        n.labels.add("Foo")
        Node.update(n)

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 SET n:Foo, n = $props REMOVE n:TestNode",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )

    def test_node_update_properties(self, mocker) -> None:
        n = Node.create(Node(labels=["TestNode"]))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        n.foo = "bar"  # type: ignore
        n.baz = "bat"  # type: ignore
        Node.update(n)

        spy.assert_called_once()
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 SET n = $props ",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"foo": "bar", "baz": "bat"}}

    def test_node_connect(self, no_strict_schema) -> None:
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

    def test_node_create_updates_edge_src(self, no_strict_schema) -> None:
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

    def test_node_create_updates_edge_dst(self, no_strict_schema) -> None:
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

    def test_node_create_updates_cache(self) -> None:
        c = Node.get_cache()
        n = Node(labels=["TestNode"])
        old_id = n.id

        Node.create(n)

        assert c.hits == 0
        assert c.misses == 0
        # old ID doesn't exist in cache
        with pytest.raises(NodeNotFound, match=f"Couldn't find node IDs: {old_id}"):
            Node.get(old_id)
        assert c.hits == 0
        assert c.misses == 1
        # new ID does exist in cache
        Node.get(n.id)
        assert c.hits == 1
        assert c.misses == 1

    def test_node_delete_new(self) -> None:
        n = Node(labels=["TestNode"])
        old_id = n.id

        Node.delete(n)

        assert n._deleted is True
        assert n._no_save is True
        assert old_id not in Node.get_cache()

    def test_node_delete_existing(self, mocker) -> None:
        n = Node(labels=["TestNode"])
        Node.save(n)
        old_id = n.id

        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")
        Node.delete(n)

        spy.assert_called_once()
        assert old_id not in Node.get_cache()
        with pytest.raises(NodeNotFound, match=f"Couldn't find node IDs: {old_id}"):
            Node.get(old_id)
        assert_similar(
            "MATCH (n) WHERE id(n) = 2746 DELETE n",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )

    def test_node_walk_src(self, test_tree) -> None:
        node_list: list[Node] = list()
        root = test_tree["root"]
        Node.walk(root, mode="src", node_callback=lambda n: node_list.append(n))
        assert len(node_list) == 5
        nodes = test_tree["nodes"]
        assert root in node_list
        assert nodes[0] in node_list
        assert nodes[1] in node_list
        assert nodes[2] in node_list
        assert nodes[3] in node_list

    def test_node_walk_dst(self, test_tree) -> None:
        node_list: list[Node] = list()
        root = test_tree["root"]
        Node.walk(root, mode="dst", node_callback=lambda n: node_list.append(n))
        assert len(node_list) == 5
        nodes = test_tree["nodes"]
        assert root in node_list
        assert nodes[5] in node_list
        assert nodes[7] in node_list
        assert nodes[8] in node_list
        assert nodes[9] in node_list

    def test_node_walk_both(self, test_tree) -> None:
        node_list: list[Node] = list()
        root = test_tree["root"]
        Node.walk(root, mode="both", node_callback=lambda n: node_list.append(n))
        assert len(node_list) == 11
        nodes = test_tree["nodes"]
        assert root in node_list
        assert nodes[0] in node_list
        assert nodes[1] in node_list
        assert nodes[2] in node_list
        assert nodes[3] in node_list
        assert nodes[4] in node_list
        assert nodes[5] in node_list
        assert nodes[6] in node_list
        assert nodes[7] in node_list
        assert nodes[8] in node_list
        assert nodes[9] in node_list

    def test_node_walk_filtered_edges(self, test_tree) -> None:
        node_list: list[Node] = list()
        root = test_tree["root"]
        nodes = test_tree["nodes"]
        Node.walk(
            root,
            mode="both",
            node_callback=lambda n: node_list.append(n),
            edge_filter=lambda e: e.type == "Test",  # type: ignore
        )
        assert len(node_list) == 5

        # walked
        assert root in node_list
        assert nodes[1] in node_list
        assert nodes[2] in node_list
        assert nodes[3] in node_list
        assert nodes[9] in node_list

        # not walked
        assert nodes[0] not in node_list  # is Foo edge
        assert nodes[4] not in node_list  # is Foo edge
        assert nodes[5] not in node_list  # is Foo edge
        assert nodes[6] not in node_list  # is related through nodes[5]
        assert nodes[7] not in node_list  # is related through nodes[5]
        assert nodes[8] not in node_list  # is related through nodes[5]

    def test_node_walk_filtered_nodes(self, test_tree) -> None:
        node_list: list[Node] = list()
        root = test_tree["root"]
        nodes = test_tree["nodes"]
        Node.walk(
            root,
            mode="both",
            node_callback=lambda n: node_list.append(n),
            node_filter=lambda n: n not in [nodes[1], nodes[8]],
        )

        assert len(node_list) == 6

        # walked
        assert root in node_list
        assert nodes[0] in node_list
        assert nodes[5] in node_list
        assert nodes[6] in node_list
        assert nodes[7] in node_list
        assert nodes[9] in node_list

        # not walked
        assert nodes[1] not in node_list  # is in filter
        assert nodes[2] not in node_list  # is related through nodes[1]
        assert nodes[3] not in node_list  # is related through nodes[1]
        assert nodes[4] not in node_list  # is related through nodes[1]
        assert nodes[8] not in node_list  # is in filter

    def test_to_dict(self) -> None:
        n = Node(labels={"TestNode"}, foo="bar")
        d = Node.to_dict(n)
        assert len(d.keys()) == 1
        assert "foo" in d.keys()
        assert d["foo"] == "bar"

    def test_to_dict_include_labels(self) -> None:
        n = Node(labels={"TestNode"}, foo="bar")
        d = Node.to_dict(n, include_labels=True)
        assert len(d.keys()) == 2
        assert "foo" in d.keys()
        assert d["foo"] == "bar"
        assert "labels" in d.keys()
        assert isinstance(d["labels"], set)
        assert len(d["labels"]) == 1
        assert "TestNode" in d["labels"]

    def test_to_dict_include_labels_no_labels(self) -> None:
        n = Node(labels={"TestNode"}, foo="bar")
        del n.labels
        d = Node.to_dict(n, include_labels=True)
        assert len(d.keys()) == 1
        assert "foo" in d.keys()
        assert d["foo"] == "bar"
        assert "labels" not in d.keys()

    def test_all_ids(self) -> None:
        node_set = Node.all_ids()
        assert isinstance(node_set, set)
        assert len(node_set) > 0

    def test_labels(self) -> None:
        class Foo(Node):
            labels: set[str] = {"NotFoo"}

        f = Foo()

        assert f.labels == {"NotFoo"}

    def test_labels_from_field(self) -> None:
        class Foo(Node):
            labels: set[str] = Field(default_factory=lambda: {"OtherNotFoo"})

        f = Foo()

        assert f.labels == {"OtherNotFoo"}

    def test_default_labels(self) -> None:
        class Foo(Node):
            pass

        class Bar(Foo):
            pass

        f = Foo()
        b = Bar()

        assert f.labels == {"Foo"}
        assert b.labels == {"Foo", "Bar"}

    def test_default_labels_with_mixin(self) -> None:
        class Mixin(ABC):
            @abstractmethod
            def bar(self) -> None: ...

        class Foo(Node, Mixin):
            def bar(self) -> None:
                pass

        f = Foo()

        assert f.labels == {"Foo"}

    def test_register_node(self) -> None:
        class Foo(Node):
            name: str

        k = frozenset(["Foo"])
        assert k in node_label_registry
        assert node_label_registry[k] is Foo
        assert "Foo" in node_registry
        assert node_registry["Foo"] is Foo

    def test_register_node_duplicate(self) -> None:
        class Foo(Node):
            name: str

        with pytest.raises(
            Exception,
            match="node_register can't register labels 'Foo' because they have already been registered",
        ):

            class Bar(Node):
                labels: set[str] = {"Foo"}

    def test_resolve_registered_node(self) -> None:
        class Foo(Node):
            name: str

        f = Foo(name="bar")
        Node.save(f)
        old_id = f.id
        del f
        node_cache = Node.get_cache()
        node_cache.clear()
        f2 = Node.get(old_id)

        assert isinstance(f2, Foo)

    def test_registered_node_sets_labels(self) -> None:
        class Foo(Node):
            name: str

        f = Foo(name="bar")
        Node.save(f)
        old_id = f.id
        del f
        node_cache = Node.get_cache()
        node_cache.clear()
        f2 = Node.get(old_id)

        assert isinstance(f2, Foo)
        assert f2.labels == {"Foo"}
        assert f2.name == "bar"

    def test_predecessors(self) -> None:
        n = Node.get(NodeId(0))

        assert len(n.predecessors) == 1
        assert n.predecessors[0].id == NodeId(2)

    def test_successors(self) -> None:
        n = Node.get(NodeId(0))

        assert len(n.successors) == 2
        assert n.successors[0].id == NodeId(6)
        assert n.successors[1].id == NodeId(453)

    def test_neighbors(self) -> None:
        n = Node.get(NodeId(0))

        assert len(n.neighbors) == 3
        assert n.neighbors[0].id == NodeId(6)
        assert n.neighbors[1].id == NodeId(453)
        assert n.neighbors[2].id == NodeId(2)

    def test_neighborhood_zero(self) -> None:
        n = Node.get(NodeId(0))

        neighborhood = n.neighborhood(depth=0)

        assert len(neighborhood) == 1
        assert neighborhood.ids == {0}

    def test_neighborhood_one(self) -> None:
        n = Node.get(NodeId(0))

        neighborhood = n.neighborhood(depth=1)

        assert len(neighborhood) == 4
        assert neighborhood.ids == {0, 2, 6, 453}

    def test_neighborhood_two(self) -> None:
        n = Node.get(NodeId(0))

        neighborhood = n.neighborhood(depth=2)

        assert len(neighborhood) == 43
        assert neighborhood.ids == {
            0,
            2,
            219,
            3,
            336,
            453,
            454,
            674,
            975,
            982,
            1101,
            1103,
            7,
            337,
            49,
            371,
            374,
            6,
            8,
            75,
            77,
            105,
            111,
            114,
            134,
            187,
            189,
            190,
            191,
            192,
            220,
            265,
            266,
            270,
            287,
            289,
            293,
            330,
            366,
            367,
            4,
            5,
            1,
        }

    def test_connections(self) -> None:
        n = Node.get(NodeId(0))
        neighborhood = n.neighborhood(depth=2)

        conns = neighborhood.connections

        assert len(conns) == 93
        assert conns.ids == {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            3084,
            17,
            18,
            19,
            20,
            3086,
            7241,
            7247,
            1109,
            3681,
            3684,
            1163,
            1164,
            3081,
            1167,
            1166,
            1170,
            3082,
            1171,
            1173,
            1174,
            1172,
            1181,
            5295,
            5296,
            5297,
            5298,
            5299,
            5300,
            3770,
            1272,
            1273,
            1274,
            1275,
            1276,
            1277,
            1278,
            1280,
            3328,
            3341,
            3342,
            3343,
            3347,
            3357,
            3361,
            3873,
            3875,
            3899,
            3902,
            5960,
            3408,
            3413,
            3420,
            3423,
            2407,
            3431,
            3436,
            2421,
            2434,
            2964,
            2967,
            919,
            2969,
            2973,
            2974,
            2975,
            2977,
            2980,
            940,
            941,
            942,
            943,
            944,
            945,
            2496,
            2499,
        }


# deletes edges


class TestEdgeList:
    def test_get_edge(self) -> None:
        n = Node.get(cast(NodeId, 0))
        e0 = n.src_edges[0]
        e1 = n.src_edges[1]
        e11 = n.dst_edges[0]
        assert isinstance(e0, Edge)
        assert isinstance(e1, Edge)
        assert isinstance(e11, Edge)
        # Edge 0
        assert e0.id == 0
        assert Edge.to_dict(e0) == {}
        assert e0.type == "LOYAL_TO"
        assert e0.src_id == 0
        assert e0.dst_id == 6
        # Edge 1
        assert e1.id == 1
        assert Edge.to_dict(e1) == {}
        assert e1.type == "VICTIM_IN"
        assert e1.src_id == 0
        assert e1.dst_id == 453
        # Edge 11
        assert e11.id == 11
        assert Edge.to_dict(e11) == {"count": 1, "method": "Ice sword"}
        assert e11.type == "KILLED"
        assert e11.src_id == 2
        assert e11.dst_id == 0

    def test_iter(self) -> None:
        n = Node.get(cast(NodeId, 0))
        for e in n.src_edges:
            assert isinstance(e, Edge)

    # test_add
    # test_add_duplicate
    # test_discard

    def test_contains(self) -> None:
        n = Node.get(cast(NodeId, 0))
        e = n.src_edges[0]

        assert e in n.src_edges
        assert e.id in n.src_edges
        assert "bob" not in n.src_edges  # type: ignore

    def test_select(self) -> None:
        n = Node.get(cast(NodeId, 2))
        src_edges = n.src_edges.select()

        assert isinstance(src_edges, EdgeList)
        assert len(src_edges) == 15

    def test_select_by_type(self, no_strict_schema) -> None:
        n = Node.get(cast(NodeId, 2))
        src_edges = n.src_edges.select(type="LOYAL_TO")

        assert isinstance(src_edges, EdgeList)
        assert len(src_edges) == 2

    def test_select_by_id(self) -> None:
        GraphDB.singleton().strict_schema = False
        n = Node.get(cast(NodeId, 2))
        src_edges = n.src_edges.select(id=EdgeId(2))

        assert isinstance(src_edges, EdgeList)
        assert len(src_edges) == 1
        assert src_edges[0].type == "LOYAL_TO"
        assert src_edges[0].src_id == 2
        assert src_edges[0].dst_id == 3

    def test_concat(self) -> None:
        list1 = EdgeList([EdgeId(1), EdgeId(2)])
        list2 = EdgeList([EdgeId(3), EdgeId(4)])

        new_list = list1 + list2

        assert len(new_list) == 4
        assert EdgeId(1) in new_list
        assert EdgeId(2) in new_list
        assert EdgeId(3) in new_list
        assert EdgeId(4) in new_list


class TestEdge:
    def test_edge_cache_control(self) -> None:
        c = Edge.get_cache()
        c.clear()
        assert c.hits == 0
        assert c.misses == 0
        assert c.currsize == 0
        assert isinstance(c, Cache)

    def test_src(self) -> None:
        n0 = Node.get(cast(NodeId, 0))
        n2 = Node.get(cast(NodeId, 2))
        e0 = n0.src_edges[0]
        e1 = n0.src_edges[1]
        e11 = n0.dst_edges[0]
        assert isinstance(e0.src, Node)
        assert isinstance(e1.src, Node)
        assert isinstance(e11.src, Node)
        assert id(e0.src) == id(n0)
        assert id(e1.src) == id(n0)
        assert id(e11.src) == id(n2)

    def test_dst(self) -> None:
        n0 = Node.get(cast(NodeId, 0))
        n6 = Node.get(cast(NodeId, 6))
        n453 = Node.get(cast(NodeId, 453))
        e0 = n0.src_edges[0]
        e1 = n0.src_edges[1]
        e11 = n0.dst_edges[0]
        assert isinstance(e0.dst, Node)
        assert isinstance(e1.dst, Node)
        assert isinstance(e11.dst, Node)
        assert id(e11.dst) == id(n0)
        assert id(e0.dst) == id(n6)
        assert id(e1.dst) == id(n453)

    def test_edge_create(self, no_strict_schema, mocker, new_edge, clear_cache) -> None:  # type: ignore
        e, src, dst = new_edge
        e_id = e.id
        Node.save(src)
        Node.save(dst)

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
        assert_similar(
            "MATCH (src), (dst) WHERE id(src) = 3102 AND id(dst) = 3103 CREATE (src)-[e:Test $props]->(dst) RETURN id",
            normalize_whitespace(spy.call_args[0][1]),
            [("3102", r"\d+"), ("3103", r"\d+")],
        )

    def test_edge_create_with_data(self, mocker, new_edge) -> None:
        e, _, _ = new_edge
        e_id = e.id
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        e.name = "bob"
        e.fun = False
        e = Edge.create(e)

        # saved both new nodes
        assert not e.src.new
        assert not e.dst.new
        assert not e.new
        # created an ID
        assert e.id != e_id
        assert e.id > 0

        assert spy.call_count == 3
        assert_similar(
            "MATCH (src), (dst) WHERE id(src) = 3102 AND id(dst) = 3103 CREATE (src)-[e:Test $props]->(dst) RETURN id",
            normalize_whitespace(spy.call_args[0][1]),
            [("3102", r"\d+"), ("3103", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"name": "bob", "fun": False}}

    def test_edge_create_updates_cache(self, new_edge, clear_cache) -> None:
        c = Edge.get_cache()
        assert c.hits == 0
        assert c.misses == 0
        e, _, _ = new_edge
        old_id = e.id

        Edge.create(e)

        # NOTE: Node.create() is called by Edge.create() if the nodes are new
        # Node.create() updates the edge.src and edge.dst, so it hits the cache twice
        assert c.hits == 2
        assert c.misses == 0
        # old ID doesn't exist in cache
        with pytest.raises(EdgeNotFound):
            Edge.get(old_id)
        assert c.hits == 2
        assert c.misses == 1
        # new ID does exist in cache
        Edge.get(e.id)
        assert c.hits == 3
        assert c.misses == 1

    def test_edge_create_updates_node_edges(self, new_edge) -> None:
        e, src, dst = new_edge
        old_id = e.id
        e.name = "bob"
        e.fun = False

        e = Edge.create(e)

        assert e.id != old_id
        assert e.id in src.src_edges
        assert e.id in dst.dst_edges

    def test_edge_create_on_delete(self, no_strict_schema, mocker) -> None:
        e = Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test")
        e.foo = "deleting-edge"  # type: ignore
        Edge.create(e)
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")

        del e
        Edge.get_cache().clear()

        spy.assert_called_once()
        assert_similar(
            "MATCH ()-[e]->() WHERE id(e) = 11928 SET e = $props",
            spy.call_args[0][1],
            [("11928", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"foo": "deleting-edge"}}

    def test_edge_immutable_properties(self, new_edge) -> None:
        e, _, _ = new_edge
        orig_src = e.src
        orig_dst = e.dst
        orig_type = e.type

        with pytest.raises(AttributeError):
            e.src = Node()
        with pytest.raises(AttributeError):
            e.dst = Node()
        assert id(e.src) == id(orig_src)
        assert id(e.dst) == id(orig_dst)
        assert e.type == orig_type

    # test_edge_update
    def test_edge_update(self, no_strict_schema, mocker) -> None:
        e = Edge.create(Node.connect(Node(labels=["TestNode"]), Node(labels=["TestNode"]), "Test"))
        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")
        e.wine = "cab"  # type: ignore
        e.more = True  # type: ignore
        Edge.update(e)

        spy.assert_called_once()
        assert_similar(
            "MATCH ()-[e]->() WHERE id(e) = 2746 SET e = $props",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )
        assert spy.call_args[1]["params"] == {"props": {"wine": "cab", "more": True}}

    def test_edge_src_after_node_save(self, new_edge) -> None:
        e, src, dst = new_edge
        old_id = src.id
        e = Edge.create(e)

        Node.save(src)

        assert src.id != old_id
        assert e.src_id == src.id
        assert id(e.src) == id(src)

    def test_edge_dst_after_node_save(self, new_edge) -> None:
        e, src, dst = new_edge
        old_id = dst.id
        e = Edge.create(e)

        Node.save(dst)

        assert dst.id != old_id
        assert e.dst_id == dst.id
        assert id(e.dst) == id(dst)

    def test_edge_delete_new(self, new_edge) -> None:
        e, src, dst = new_edge
        e = Edge.create(e)
        old_id = e.id

        Edge.delete(e)

        assert e._deleted is True
        assert e._no_save is True
        assert old_id not in Edge.get_cache()

    def test_edge_delete_existing(self, mocker, new_edge) -> None:
        e, src, dst = new_edge
        e = Edge.create(e)
        old_id = e.id

        spy: MagicMock = mocker.spy(GraphDB, "raw_execute")
        Edge.delete(e)

        spy.assert_called_once()
        assert old_id not in Edge.get_cache()
        with pytest.raises(EdgeNotFound):
            Edge.get(old_id)
        assert_similar(
            "MATCH ()-[e]->() WHERE id(e) = 2746 DELETE e",
            spy.call_args[0][1],
            [("2746", r"\d+")],
        )

    def test_to_dict(self, new_edge) -> None:
        e, _, _ = new_edge
        e.foo = "blah"
        d = Edge.to_dict(e)
        assert len(d.keys()) == 1
        assert "foo" in d.keys()
        assert d["foo"] == "blah"

    def test_to_dict_include_type(self, new_edge) -> None:
        e, _, _ = new_edge
        e.foo = "blah"
        assert e.type == "Test"

        d = Edge.to_dict(e, include_type=True)
        assert len(d.keys()) == 2
        assert "foo" in d.keys()
        assert d["foo"] == "blah"
        assert "type" in d.keys()
        assert d["type"] == "Test"

    def test_to_dict_include_type_no_type(self, new_edge) -> None:
        e, _, _ = new_edge
        e.foo = "blah"
        del e.type

        d = Edge.to_dict(e, include_type=True)
        assert len(d.keys()) == 1
        assert "foo" in d.keys()
        assert d["foo"] == "blah"
        assert "type" not in d.keys()

    def test_to_dict_exclude_allowed_connections(self) -> None:
        class NewEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("Testy", "Testy")]

        class Testy(Node):
            pass

        n1 = Testy()
        n2 = Testy()
        e = NewEdge.connect(n1, n2)

        res = Edge.to_dict(e)

        assert res == {}

    @pytest.mark.skip("pending")
    def test_edge_cache(self) -> None:
        pass

    @pytest.mark.skip("pending")
    def test_edge_save(self) -> None:
        pass

    def test_edge_connect(self, no_strict_schema) -> None:
        n1 = Node()
        n2 = Node()

        f = Edge.connect(n1, n2, "Foo")
        assert f.type == "Foo"

    def test_type(self, no_strict_schema) -> None:
        class Foo(Edge):
            type: str = "NotFoo"

        f = Foo.connect(Node(), Node())

        assert f.type == "NotFoo"

    def test_type_from_field(self, no_strict_schema) -> None:
        class Foo(Edge):
            type: str = Field(default_factory=lambda: "OtherNotFoo")

        f = Foo.connect(Node(), Node())

        assert f.type == "OtherNotFoo"

    def test_default_type(self, no_strict_schema) -> None:
        class Foo(Edge):
            pass

        class Bar(Foo):
            pass

        f = Foo.connect(Node(), Node())
        b = Bar.connect(Node(), Node())

        assert f.type == "Foo"
        assert b.type == "Bar"

    def test_edge_connect_no_type(self) -> None:
        n1 = Node()
        n2 = Node()

        with pytest.raises(Exception):
            Edge.connect(n1, n2)

    def test_edge_type_lookup(self, no_strict_schema) -> None:
        n1 = Node()
        n2 = Node()

        class Foo(Edge):
            name: str

        f = Edge.connect(n1, n2, "Foo", name="bar")
        assert f.type == "Foo"
        assert "Foo" in edge_registry
        assert edge_registry["Foo"] is Foo

    def test_register_edge(self, no_strict_schema) -> None:
        class Foo(Edge):
            name: str

        f = Foo.connect(Node(), Node(), name="bar")
        assert f.type == "Foo"
        assert "Foo" in edge_registry
        assert edge_registry["Foo"] is Foo

    def test_register_edge_duplicate(self) -> None:
        class Foo(Edge):
            type: str = "Foo"

        with pytest.raises(
            Exception,
            match="edge_register can't register type 'Foo' because it has already been registered",
        ):

            class Bar(Edge):
                type: str = "Foo"

    def test_resolve_registered_edge(self, no_strict_schema) -> None:
        class Foo(Edge):
            name: str

        f = Foo.connect(Node(), Node(), name="bar")
        Edge.save(f)
        old_id = f.id
        del f
        node_cache = Node.get_cache()
        node_cache.clear()
        f2 = Edge.get(old_id)

        assert isinstance(f2, Foo)

    def test_registered_node_sets_type(self, no_strict_schema) -> None:
        n1 = Node()
        n2 = Node()

        class Foo(Edge):
            name: str

        f = Foo.connect(n1, n2, name="bar")
        Edge.save(f)
        old_id = f.id
        del f
        node_cache = Node.get_cache()
        node_cache.clear()
        f2 = Edge.get(old_id)

        assert isinstance(f2, Foo)
        assert f2.type == "Foo"
        assert f2.name == "bar"

    def test_connection_allowed(self) -> None:
        class Foo(Edge):
            allowed_connections: EdgeConnectionsList = [("Node", "Node")]
            name: str

        f = Foo.connect(Node(), Node(), name="bar")
        assert f.type == "Foo"

    def test_connection_allowed_to_parent(self) -> None:
        class Foo(Node):
            pass

        class Parent(Node):
            pass

        class Bar(Parent):
            pass

        n1 = Foo()
        n2 = Bar()

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Parent")]
            name: str

        f = Link.connect(n1, n2, name="bar")
        assert f.type == "Link"

    def test_connection_not_allowed(self, no_strict_schema) -> None:
        class Foo(Edge):
            allowed_connections: EdgeConnectionsList = [("Node", "Bar")]
            name: str

        with pytest.raises(
            Exception,
            match="attempting to connect edge 'Foo' from 'Node' to 'Node' not in allowed connections list",
        ):
            Foo.connect(Node(), Node(), name="bar")

    def test_strict_schema(self, strict_schema) -> None:
        class Foo(Edge):
            name: str

        with pytest.raises(
            Exception, match="allowed_connections missing in 'Foo' and strict_schema is set"
        ):
            Foo.connect(Node(), Node(), name="bar")

    def test_strict_schema_warns(self, strict_schema, strict_schema_warns) -> None:
        class Foo(Edge):
            name: str

        with pytest.warns(
            StrictSchemaWarning,
            match="allowed_connections missing in 'Foo' and strict_schema is set",
        ):
            Foo.connect(Node(), Node(), name="bar")


class TestNodeList:
    def test_get_node(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)])
        n = node_list[0]

        assert n.id == 2
        assert n.labels == {"Character"}

    def test_iter(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)])
        n = node_list[0]

        assert n.id == 2
        assert n.labels == {"Character"}

    # test_add
    # test_add_duplicate
    # test_discard

    def test_contains(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)])
        n = node_list[0]

        assert n in node_list
        assert n.id in node_list
        assert "bob" not in n.src_edges  # type: ignore

    def test_select(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)])
        ret = node_list.select()

        assert isinstance(ret, NodeList)
        assert len(ret) == 3

    def test_select_by_labels(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)]).select(labels={"Character"})

        assert isinstance(node_list, NodeList)
        assert len(node_list) == 2
        assert node_list[0].id == NodeId(2)
        assert node_list[1].id == NodeId(0)

    def test_select_by_exact_labels(self) -> None:
        n1 = Node(labels={"TestNode", "Foo", "Bar"})
        n2 = Node(labels={"TestNode", "Foo"})
        n3 = Node(labels={"TestNode", "Foo"})
        nl = NodeList([n1.id, n2.id, n3.id])
        node_list = nl.select(labels={"TestNode", "Foo", "Bar"})
        assert len(node_list) == 1
        assert n1.id in node_list
        assert n2.id not in node_list
        assert n3.id not in node_list

        node_list = nl.select(labels={"TestNode", "Foo"})
        assert len(node_list) == 2
        assert n1.id not in node_list
        assert n2.id in node_list
        assert n3.id in node_list

    def test_select_by_partial_labels(self) -> None:
        n1 = Node(labels={"TestNode", "Foo", "Bar"})
        n2 = Node(labels={"TestNode", "Foo"})
        n3 = Node(labels={"TestNode", "Baz"})
        nl = NodeList([n1.id, n2.id, n3.id])
        node_list = nl.select(partial_labels={"Foo"})
        assert len(node_list) == 2
        assert n1.id in node_list
        assert n2.id in node_list
        assert n3.id not in node_list

        node_list = nl.select(partial_labels={"Baz"})
        assert len(node_list) == 1
        assert n1.id not in node_list
        assert n2.id not in node_list
        assert n3.id in node_list

        node_list = nl.select(partial_labels={"TestNode"})
        assert len(node_list) == 3
        assert n1.id in node_list
        assert n2.id in node_list
        assert n3.id in node_list

    def test_select_by_function(self) -> None:
        node_list = NodeList([NodeId(2), NodeId(1), NodeId(0)]).select(
            filter_fn=lambda n: n.id == NodeId(1)
        )

        assert isinstance(node_list, NodeList)
        assert len(node_list) == 1
        assert node_list[0].id == NodeId(1)

    def test_concat(self) -> None:
        list1 = NodeList([NodeId(1), NodeId(2)])
        list2 = NodeList([NodeId(3), NodeId(4)])

        new_list = list1 + list2

        assert len(new_list) == 4
        assert NodeId(1) in new_list
        assert NodeId(2) in new_list
        assert NodeId(3) in new_list
        assert NodeId(4) in new_list

    def test_to_dot(self) -> None:
        class Character(Node):
            name: str

        class Allegiance(Node):
            name: str

        class Death(Node):
            order: int

        n = Node.get(NodeId(0))

        dot_str = n.neighborhood(depth=1).to_dot()

        assert dot_str == dot_node1


class TestTypes:
    def test_get(self):
        c = GotCharacter.get(cast(NodeId, 0))

        assert isinstance(c, GotCharacter)
        assert isinstance(c, Node)

    def test_parse_fail(self):
        with pytest.raises(ValidationError):
            GotSeason.get(cast(NodeId, 0))

    def test_same_cache(self):
        c = GotCharacter.get(cast(NodeId, 0))
        n = Node.get(cast(NodeId, 0))

        assert id(n) == id(c)
        assert id(Node.get_cache()) == id(GotCharacter.get_cache())

    def test_missing_args(self):
        GotCharacter(name="bob", labels=set())
        GotCharacter(name="bob", id=None)
        GotCharacter(name="bob")


class TestSchema:
    def test_validate(self, clear_registries) -> None:
        class Foo(Node):
            pass

        class Bar(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Bar")]

        Schema.validate()

    def test_validate_fails(self, clear_registries) -> None:
        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Bar")]

        expected_error = (
            "Error validating schema:\n"
            + "\t0: Edge 'Link' requires src Node 'Foo', which is not registered\n"
            + "\t1: Edge 'Link' requires dst Node 'Bar', which is not registered\n"
        )

        with pytest.raises(SchemaValidationError, match=expected_error):
            Schema.validate()

    def test_validate_no_allowed_connections(self, clear_registries) -> None:
        class Foo(Node):
            pass

        class Bar(Node):
            pass

        class Link(Edge):
            pass

        Schema.validate()

    def test_create(self, clear_registries) -> None:
        class Foo(Node):
            pass

        class Bar(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Bar")]

        schema = Schema()
        assert schema.edge_names == {"Link"}
        assert schema.node_names == {"Foo", "Bar"}

    def test_mermaid(self, clear_registries) -> None:
        class Bar(Node):
            weight: float

            def print_weight(self) -> None:
                pass

        class Foo(Bar):
            name: str = Field(default="Bob")

            def set_name(self, name: str = "Uggo") -> str:
                self.name = name
                return self.name

        class Baz(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Baz")]

        schema = Schema()

        assert schema.to_mermaid() == mermaid_schema1

    def test_dot(self, clear_registries) -> None:
        class Bar(Node):
            weight: float

            def print_weight(self) -> None:
                pass

        class Foo(Bar):
            name: str = Field(default="Bob")

            def set_name(self, name: str = "Uggo") -> str:
                self.name = name
                return self.name

        class Baz(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Baz")]

        schema = Schema()

        assert schema.to_dot() == dot_schema1


class TestEdgeDescription:
    def test_create(self) -> None:
        class Foo(Node):
            pass

        class Bar(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Foo", "Bar"), ("Bar", "Foo")]
            name: str = "Bob"

        ed = _EdgeDescription(Link)
        assert ed.edge_cls is Link
        assert ed.allowed_connections == [("Foo", "Bar"), ("Bar", "Foo")]
        assert ed.related_nodes == {"Foo", "Bar"}
        assert ed.name == "Link"
        assert ed.edgetype == "Link"
        assert ed.resolved_name == "Link"
        assert ed.related_nodes == {"Foo", "Bar"}
        assert str(ed) == "EdgeDesc(Link)"
        assert len(ed.fields) == 5
        assert ed.fields[0].name == "allowed_connections"
        assert ed.fields[1].name == "dst_id"
        assert ed.fields[2].name == "name"
        assert ed.fields[3].name == "src_id"
        assert ed.fields[4].name == "type"
        assert ed.parent_class_names == set()
        assert ed.parents == []
        assert ed.method_names == set()
        assert ed.methods == []


class TestNodeDescription:
    def test_create(self) -> None:
        class Bar(Node):
            weight: float

            def print_weight(self) -> None:
                pass

        class Foo(Bar):
            name: str = Field(default="Bob")

            def set_name(self, name: str = "Uggo") -> str:
                self.name = name
                return self.name

        nd = _NodeDescription(Foo)

        assert nd.name == "Foo"
        assert len(nd.fields) == 3
        assert nd.fields[0].name == "labels"
        assert nd.fields[1].name == "name"
        assert nd.fields[2].name == "weight"
        assert nd.parent_class_names == {"Bar"}
        assert len(nd.parents) == 1
        assert nd.parents[0].name == "Bar"
        assert nd.method_names == {"set_name", "print_weight"}
        assert len(nd.methods) == 2
        assert nd.methods[0].name == "print_weight"
        assert nd.methods[1].name == "set_name"
