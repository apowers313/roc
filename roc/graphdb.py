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
