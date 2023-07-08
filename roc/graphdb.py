from typing import Any, Dict

from collections.abc import Iterator

from gqlalchemy import Memgraph

from roc.config import settings


class GraphDB:
    def __init__(self):
        self.host = settings.db_host
        self.port = settings.db_port
        self.db = None

    def connect(self):
        """Connects to the database. The host and port for the database are specified through the config variables 'db_host' and 'db_port' (respectively).

        Example:
            >>> db = GraphDB()
            >>> db.connect()
        """
        self.db = Memgraph(host=self.host, port=self.port)

    def query(self, query: str, *, fetch: bool = True) -> Iterator[dict[str, Any]] | None:
        if not self.db:
            raise Exception("database not connected")

        if fetch:
            return self.db.execute_and_fetch(query)  # type: ignore
        else:
            self.db.execute(query)
            return None


import gc
import weakref
from functools import lru_cache

node_counter = 0
edge_counter = 0


@lru_cache(maxsize=4)
def get_node(id):
    return Node(id)


def new_node():
    global node_counter
    n = get_node(node_counter)
    node_counter += 1
    return n


def reset_counters():
    global node_counter
    global edge_counter
    node_counter = 0
    edge_counter = 0


class Node:
    def __init__(self, id):
        global node_counter
        if not id:
            self._id = node_counter
        else:
            self._id = id
        print(">>> created node", self._id)
        # node_counter += 1
        self._src_edges = []
        self._dst_edges = []

    def __del__(self):
        print("deleting node:", self.id)
        # print("deleting node")

    @property
    def id(self):
        return f"node-{self._id}"

    @staticmethod
    def load(id):
        return Node(id)


class Edge:
    def __init__(self, src, dst):
        global edge_counter
        self._id = edge_counter
        print("+++ created edge", self._id)
        edge_counter += 1

        # make weak connections
        self._src = weakref.proxy(src)
        src._dst_edges.append(weakref.proxy(self))
        self._dst = weakref.proxy(dst)
        dst._src_edges.append(weakref.proxy(self))

        # finalizers (for status)
        def nop():
            pass

        self._src_finalizer = weakref.finalize(src, nop)
        self._dst_finalizer = weakref.finalize(dst, nop)

        print("Edge __init__ returning for", self.id)

    def __del__(self):
        print("deleting edge:", self.id)
        # print("deleting edge")

    @property
    def id(self):
        return f"edge-{self._id}"


def test_cache():
    # e = Edge(new_node(), new_node())
    # print("cache stats:", get_node.cache_info())

    # create chain
    on = None
    nn = new_node()
    first_node = nn
    edge_list = []
    for i in range(10):
        print("cache stats:", get_node.cache_info())
        print(i)
        on = nn
        nn = new_node()
        e = Edge(on, nn)
        # XXX: only save strong references to edges; most nodes will get dropped
        edge_list.append(e)

    # these should still have strong references
    print("first node:", first_node)
    print("first node id:", first_node.id)
    print(nn)
    print(nn.id)

    for e in edge_list:
        print(f"src [{e._src_finalizer.alive}] --[{e.id}]--> [{e._dst_finalizer.alive}]")


test_cache()


def test_chain():
    old_node = None
    new_node = Node()
    first_node = new_node
    for i in range(10):
        print(i)
        old_node = new_node
        new_node = Node()
        Edge(old_node, new_node)

    print(first_node)
    print(first_node.id)
    print(new_node)
    print(new_node.id)

    n = first_node
    for _ in range(10):
        e = n._dst_edges[0]
        print(f"src [{n.id}] --[{e.id}]--> [{e._dst.id}]")
        n = e._dst


# test_chain()


def test_delete_proxy():
    # n1 = Node()
    # n2 = Node()
    # e = Edge(n1, n2)
    e = Edge(Node(), Node())
    # del e
    print("e wr count", weakref.getweakrefcount(e))
    # print("wr count", weakref.getweakrefcount(n1))
    # print("wr count", weakref.getweakrefcount(n2))
    # assert weakref.getweakrefcount(e) == 1

    # del n1
    # del n2
    gc.collect()
    print("e wr count", weakref.getweakrefcount(e))

    assert weakref.getweakrefcount(e) == 0


# test_delete_proxy()


def test_node():
    n = Node()

    assert n.id == "node-0"


def test_edge():
    n1 = Node()
    n2 = Node()
    e = Edge(n1, n2)

    assert e.id == "edge-0"

    assert e._src == n1
    assert e._dst == n2
    assert n1._dst_edges[0] == e
    assert n2._src_edges[0] == e
