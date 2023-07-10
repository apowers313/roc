from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Literal, NamedTuple, Type, TypeVar, overload

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from threading import Lock

from cachetools import Cache, LRUCache, cached
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

    @overload
    def raw_query(self, query: str, *, fetch: Literal[True]) -> Iterator[dict[str, Any]]:
        ...

    @overload
    def raw_query(self, query: str, *, fetch: Literal[False]) -> None:
        ...

    def raw_query(self, query: str, *, fetch: bool = True) -> Iterator[dict[str, Any]] | None:
        if not self.db:
            raise Exception("database not connected")

        if fetch:
            return self.db.execute_and_fetch(query)  # type: ignore
        else:
            self.db.execute(query)
            return None


db = GraphDB()
cache_size = 2**8


RefType = TypeVar("RefType")


class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class CacheControl(Generic[RefType]):
    def __init__(self, cache_fn: Any):
        self.cache: Cache[int, RefType] = cache_fn.cache
        self.key: Callable[[Any, Any], tuple[Any]] = cache_fn.cache_key
        self.lock: Lock | None = cache_fn.cache_lock
        self.clear: Callable[[None], None] = cache_fn.cache_clear
        self.info: Callable[[], CacheInfo] = cache_fn.cache_info


class Edge:
    def __init__(
        self, id: int, src_id: int, dst_id: int, *, data: dict[Any, Any] | None = None, label: str | None = None
    ):
        self.id = id
        self.label = label
        self.data = data
        self.src_id = src_id
        self.dst_id = dst_id

    @staticmethod
    @cached(cache=LRUCache(settings.edge_cache_size), info=True)
    def get(id: int) -> Edge:
        return Edge.load(id)

    @staticmethod
    def load(id: int) -> Edge:
        edge_list = [e for e in db.raw_query(f"MATCH [e] WHERE id(e) = {id}", fetch=True)]
        # reveal_type(edge_list)
        if not len(edge_list) == 1:
            raise Exception(f"Couldn't find edge ID: {id}")

        e = edge_list[0]
        # reveal_type(e)
        return Edge(id, e["start"], e["end"], data=e["properties"], label=e["label"])

    @property
    def src(self) -> Node:
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        return Node.get(self.dst_id)

    @classmethod
    def get_cache_control(self) -> CacheControl[Edge]:
        return CacheControl[Edge](Edge.get)


class EdgeList(Mapping[int, Edge]):
    __edges: list[Callable[[None], Edge]]

    def __init__(self, ids: list[int]):
        for i in range(len(ids)):
            self.__edges[i] = functools.partial(Edge.get, id)

    def __iter__(self):
        if not self.__edges:
            return iter([])

        return iter(self.__edges)

    def __getitem__(self, key):
        if not key in self.__edges:
            raise KeyError(f"Key not found: {key}")

        return self.__edges[key]()

    def __len__(self):
        if not self.__edges:
            return 0

        return len(self.__edges)


class Node:
    def __init__(
        self, id: int, edges: EdgeList, *, data: dict[Any, Any] | None = None, labels: list[str] | None = None
    ):
        self.id = id
        self.data = data
        self.labels = labels
        self.edge_list = None

    @staticmethod
    def load(id: int) -> Node:
        node_list = [n for n in db.raw_query(f"MATCH (n) WHERE id(n) = {id}", fetch=True)]
        # reveal_type(node_list)
        if not len(node_list) == 1:
            raise Exception(f"Couldn't find edge ID: {id}")

        n = node_list[0]
        # reveal_type(n)
        return Node(
            id,
            EdgeList([]),  # TODO: edges
            data=n["properties"],
            labels=n["labels"],
        )

    @cached(cache=LRUCache(settings.node_cache_size), info=True)
    @staticmethod
    def get(id: int) -> Node:
        return Node.load(id)

    @classmethod
    def get_cache_control(self) -> CacheControl[Node]:
        return CacheControl[Node](Node.get)


# class DbWeakref(ABC, Generic[RefType]):
#     def __init__(self, id: int):
#         self.id = id
#         self.value: RefType | None = None
#         self.ref = None

#     @property
#     @abstractmethod
#     def cache(self) -> Cache[int, RefType]:
#         pass

#     def get_value(self) -> RefType:
#         try:
#             return self.cache[self.id]
#         except KeyError:
#             return self.load()

#     @abstractmethod
#     def load(self) -> RefType:
#         # call DB
#         pass


# # class cached_class(Generic[RefType]):
# #     # def __init__(self, wrapped: Callable[[Any, Any], Any], cache: Cache[int, RefType]):
# #     def __init__(self, wrapped: Any):
# #         # self.cache = cache
# #         self.wrapped = wrapped

# #     def __call__(self, *args, **kwargs):
# #         return self.wrapped(*args, **kwargs)


# # node_cache: LRUCache[int, Node]

# from functools import lru_cache


# # @cached_class(cache=node_cache)
# @lru_cache
# class Node:
#     def __init__(self, id: int, *, data: dict[Any, Any] | None = None, labels: list[str] | None = None):
#         self.id = id
#         self.data = data
#         self.labels = labels
#         self.edge_list = None
#         node_cache[id] = self


# node_cache = LRUCache[int, Node](2**8)

# # edge_class: LRUCache[int, Edge]


# @lru_cache
# class Edge:
#     def __init__(
#         self, id: int, src_id: int, dst_id: int, *, data: dict[Any, Any] | None = None, label: str | None = None
#     ):
#         self.id = id
#         self.label = label
#         self.data = data
#         self.src_id = src_id
#         self.dst_id = dst_id
#         self.__src = NodeWeakref(src_id)
#         self.__dst = NodeWeakref(dst_id)
#         edge_cache[id] = self

#     @property
#     def src(self) -> Node:
#         return self.__src.get_value()

#     @property
#     def dst(self) -> Node:
#         return self.__dst.get_value()


# edge_cache = LRUCache[int, Edge](2**8)


# class EdgeWeakref(DbWeakref[Edge]):
#     @property
#     def cache(self):
#         return edge_cache

#     def load(self) -> Edge:
#         edge_list = [e for e in db.query(f"MATCH [e] WHERE e.id = {self.id}", fetch=True)]
#         # reveal_type(edge_list)
#         if not len(edge_list) == 1:
#             raise Exception(f"Couldn't find edge ID: {self.id}")

#         e = edge_list[0]
#         # reveal_type(e)
#         return Edge(self.id, e["start"], e["end"], data=e["properties"], label=e["label"])


# class NodeWeakref(DbWeakref[Node]):
#     @property
#     def cache(self):
#         return node_cache

#     def load(self) -> Node:
#         node_list = [n for n in db.query(f"MATCH (n) WHERE n.id = {self.id}", fetch=True)]
#         # reveal_type(node_list)
#         if not len(node_list) == 1:
#             raise Exception(f"Couldn't find edge ID: {self.id}")

#         n = node_list[0]
#         # reveal_type(n)
#         return Node(self.id, data=n["properties"], labels=n["labels"])


# class EdgeList(Mapping[int, Edge]):
#     __edges: list[EdgeWeakref] | None

#     def set_ids(self, ids: list[int]) -> None:
#         self.__edges = []
#         for i in range(len(ids)):
#             self.__edges[i] = EdgeWeakref(ids[i])

#     def __iter__(self):
#         if not self.__edges:
#             return iter([])

#         return iter(self.__edges)

#     def __getitem__(self, key):
#         if not self.__edges:
#             raise KeyError(f"Key not found: {key}")

#         return self.__edges[key].get_value()

#     def __len__(self):
#         if not self.__edges:
#             return 0

#         return len(self.__edges)


# import gc
# import weakref
# from functools import lru_cache

# node_counter = 0
# edge_counter = 0


# @lru_cache(maxsize=4)
# def get_node(id):
#     return Node(id)


# def new_node():
#     global node_counter
#     n = get_node(node_counter)
#     node_counter += 1
#     return n


# def reset_counters():
#     global node_counter
#     global edge_counter
#     node_counter = 0
#     edge_counter = 0


# class Node:
#     def __init__(self, id):
#         global node_counter
#         if not id:
#             self._id = node_counter
#         else:
#             self._id = id
#         print(">>> created node", self._id)
#         # node_counter += 1
#         self._src_edges = []
#         self._dst_edges = []

#     def __del__(self):
#         print("deleting node:", self.id)
#         # print("deleting node")

#     @property
#     def id(self):
#         return f"node-{self._id}"

#     @staticmethod
#     def load(id):
#         return Node(id)


# class Edge:
#     def __init__(self, src, dst):
#         global edge_counter
#         self._id = edge_counter
#         print("+++ created edge", self._id)
#         edge_counter += 1

#         # make weak connections
#         self._src = weakref.proxy(src)
#         src._dst_edges.append(weakref.proxy(self))
#         self._dst = weakref.proxy(dst)
#         dst._src_edges.append(weakref.proxy(self))

#         # finalizers (for status)
#         def nop():
#             pass

#         self._src_finalizer = weakref.finalize(src, nop)
#         self._dst_finalizer = weakref.finalize(dst, nop)

#         print("Edge __init__ returning for", self.id)

#     def __del__(self):
#         print("deleting edge:", self.id)
#         # print("deleting edge")

#     @property
#     def id(self):
#         return f"edge-{self._id}"


# def test_cache():
#     # e = Edge(new_node(), new_node())
#     # print("cache stats:", get_node.cache_info())

#     # create chain
#     on = None
#     nn = new_node()
#     first_node = nn
#     edge_list = []
#     for i in range(10):
#         print("cache stats:", get_node.cache_info())
#         print(i)
#         on = nn
#         nn = new_node()
#         e = Edge(on, nn)
#         # XXX: only save strong references to edges; most nodes will get dropped
#         edge_list.append(e)

#     # these should still have strong references
#     print("first node:", first_node)
#     print("first node id:", first_node.id)
#     print(nn)
#     print(nn.id)

#     for e in edge_list:
#         print(f"src [{e._src_finalizer.alive}] --[{e.id}]--> [{e._dst_finalizer.alive}]")


# test_cache()


# def test_chain():
#     old_node = None
#     new_node = Node()
#     first_node = new_node
#     for i in range(10):
#         print(i)
#         old_node = new_node
#         new_node = Node()
#         Edge(old_node, new_node)

#     print(first_node)
#     print(first_node.id)
#     print(new_node)
#     print(new_node.id)

#     n = first_node
#     for _ in range(10):
#         e = n._dst_edges[0]
#         print(f"src [{n.id}] --[{e.id}]--> [{e._dst.id}]")
#         n = e._dst


# # test_chain()


# def test_delete_proxy():
#     # n1 = Node()
#     # n2 = Node()
#     # e = Edge(n1, n2)
#     e = Edge(Node(), Node())
#     # del e
#     print("e wr count", weakref.getweakrefcount(e))
#     # print("wr count", weakref.getweakrefcount(n1))
#     # print("wr count", weakref.getweakrefcount(n2))
#     # assert weakref.getweakrefcount(e) == 1

#     # del n1
#     # del n2
#     gc.collect()
#     print("e wr count", weakref.getweakrefcount(e))

#     assert weakref.getweakrefcount(e) == 0


# # test_delete_proxy()


# def test_node():
#     n = Node()

#     assert n.id == "node-0"


# def test_edge():
#     n1 = Node()
#     n2 = Node()
#     e = Edge(n1, n2)

#     assert e.id == "edge-0"

#     assert e._src == n1
#     assert e._dst == n2
#     assert n1._dst_edges[0] == e
#     assert n2._src_edges[0] == e
