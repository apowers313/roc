from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from threading import Lock
from typing import Any, Callable, Generic, Literal, NamedTuple, TypeVar, overload

from cachetools import Cache, LRUCache, cached
from gqlalchemy import Memgraph

from roc.config import settings

RecordFn = Callable[[str, Iterator[Any]], None]


class GraphDB:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance.__isinitialized = False  # type: ignore
        return cls.instance

    def __init__(self):
        if self.__isinitialized:  # type: ignore
            return

        self.__isinitialized = True
        self.host = settings.db_host
        self.port = settings.db_port
        self.db = Memgraph(host=self.host, port=self.port)
        self.record_callback: RecordFn | None = None

    @overload
    def raw_query(self, query: str, *, fetch: Literal[True]) -> Iterator[dict[str, Any]]:
        ...

    @overload
    def raw_query(self, query: str, *, fetch: Literal[False]) -> None:
        ...

    def raw_query(self, query: str, *, fetch: bool = True) -> Iterator[dict[str, Any]] | None:
        print(f"raw_query: '{query}'")

        if fetch:
            if self.record_callback:
                self.record_callback(query, self.db.execute_and_fetch(query))

            ret = self.db.execute_and_fetch(query)
            return ret  # type: ignore
        else:
            self.db.execute(query)
            return None


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
        self.clear: Callable[[], None] = cache_fn.cache_clear
        self.info: Callable[[], CacheInfo] = cache_fn.cache_info


class Edge:
    def __init__(
        self,
        id: int,
        src_id: int,
        dst_id: int,
        *,
        data: dict[Any, Any] | None = None,
        type: str | None = None,
    ):
        self.id = id
        self.type = type
        self.data = data or {}
        self.src_id = src_id
        self.dst_id = dst_id

    @staticmethod
    @cached(cache=LRUCache(settings.edge_cache_size), info=True)
    def get(id: int) -> Edge:
        return Edge.load(id)

    @staticmethod
    def load(id: int) -> Edge:
        db = GraphDB()
        edge_list = list(db.raw_query(f"MATCH (n)-[e]-(m) WHERE id(e) = {id} RETURN e LIMIT 1", fetch=True))
        if not len(edge_list) == 1:
            raise Exception(f"Couldn't find edge ID: {id}")

        e = edge_list[0]["e"]
        props = None
        if hasattr(e, "_properties"):
            props = e._properties
        return Edge(id, e._start_node_id, e._end_node_id, data=props, type=e._type)

    @property
    def src(self) -> Node:
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        return Node.get(self.dst_id)

    @classmethod
    def get_cache_control(self) -> CacheControl[Edge]:
        return CacheControl[Edge](Edge.get)


class EdgeFetchIterator:
    def __init__(self, edgeFetchList: list[Callable[[], Edge]]):
        self.__edgeFetchList = edgeFetchList
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur >= len(self.__edgeFetchList):
            raise StopIteration

        fetcher = self.__edgeFetchList[self.cur]
        self.cur = self.cur + 1
        return fetcher()


class EdgeList(Mapping[int, Edge]):
    def __init__(self, ids: list[int]):
        self.__edges: list[Callable[[], Edge]] = []
        for id in ids:
            self.__edges.append(functools.partial(Edge.get, id))

    def __iter__(self):
        return EdgeFetchIterator(self.__edges)

    def __getitem__(self, key):
        return self.__edges[key]()

    def __len__(self):
        return len(self.__edges)


class Node:
    def __init__(
        self,
        id: int,
        src_edges: EdgeList,
        dst_edges: EdgeList,
        *,
        data: dict[Any, Any] | None = None,
        labels: set[str] | list[str] | None = None,
    ):
        self.id = id
        self.data = data or {}
        if isinstance(labels, list):
            labels = set(labels)
        self.labels = labels or set()
        self.src_edges = src_edges
        self.dst_edges = dst_edges

    @staticmethod
    def load(id: int) -> Node:
        db = GraphDB()
        res = list(
            db.raw_query(
                f"""
            MATCH (n)-[e]-(m) WHERE id(n) = {id}
            RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
            """,
                fetch=True,
            )
        )

        if not len(res) >= 1:
            raise Exception(f"Couldn't find node ID: {id}")

        n = res[0]["n"]
        edges = list(map(lambda r: {"id": r["e_id"], "start": r["e_start"], "end": r["e_end"]}, res))
        src_edges = list(map(lambda e: e["id"], filter(lambda e: e["start"] == id, edges)))
        dst_edges = list(map(lambda e: e["id"], filter(lambda e: e["end"] == id, edges)))
        # reveal_type(n)
        return Node(
            id,
            EdgeList(src_edges),
            EdgeList(dst_edges),
            data=n._properties,
            labels=n._labels,
        )

    @cached(cache=LRUCache(settings.node_cache_size), info=True)
    @staticmethod
    def get(id: int) -> Node:
        return Node.load(id)

    @classmethod
    def get_cache_control(self) -> CacheControl[Node]:
        return CacheControl[Node](Node.get)
