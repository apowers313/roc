from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableSet
from threading import Lock
from typing import Any, Callable, Generic, Literal, NamedTuple, TypeVar, overload

from cachetools import Cache, LRUCache, cached
from gqlalchemy import Memgraph

from roc.config import settings

RecordFn = Callable[[str, Iterator[Any]], None]


class GraphDB:
    """
    A graph database singleton. Settings for the graph database come from the config module.
    """

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
    def raw_query(
        self, query: str, *, params: dict[str, Any] | None = None, fetch: Literal[True]
    ) -> Iterator[dict[str, Any]]:
        ...

    @overload
    def raw_query(
        self, query: str, *, params: dict[str, Any] | None = None, fetch: Literal[False]
    ) -> None:
        ...

    def raw_query(
        self, query: str, *, params: dict[str, Any] | None = None, fetch: bool = True
    ) -> Iterator[dict[str, Any]] | None:
        print(f"raw_query: '{query}'")
        params = params or {}

        if fetch:
            if self.record_callback:
                self.record_callback(query, self.db.execute_and_fetch(query, parameters=params))

            ret = self.db.execute_and_fetch(query, parameters=params)
            return ret  # type: ignore
        else:
            self.db.execute(query, parameters=params)
            return None


RefType = TypeVar("RefType")


class CacheInfo(NamedTuple):
    """
    Information about the cache: hits, misses, max size, and current size.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class CacheControl(Generic[RefType]):
    """
    For controlling the Node and Edge caches, such as clearing them, getting their current or max
    size, adding items to the cache (outside the automatic methods for adding cached items), etc.
    """

    def __init__(self, cache_fn: Any):
        self.cache: Cache[int, RefType] = cache_fn.cache
        self.key: Callable[[Any, Any], tuple[Any]] = cache_fn.cache_key
        self.lock: Lock | None = cache_fn.cache_lock
        self.clear: Callable[[], None] = cache_fn.cache_clear
        self.info: Callable[[], CacheInfo] = cache_fn.cache_info


next_new_edge = -1


class EdgeMeta(type):
    @property
    def cache_control(cls):
        return CacheControl[Edge](Edge.get)


class Edge(metaclass=EdgeMeta):
    """
    An edge (a.k.a. Relationship or Connection) between two Nodes. An edge obect automatically
    implements all phases of CRUD in the underlying graph database. This is a directional
    relationship with a "source" and "destination". The source and destination properties
    are dynamically loaded through property getters when they are called, and may trigger
    a graph database query if they don't already exist in the edge cache.
    """

    def __init__(
        self,
        src_id: int,
        dst_id: int,
        *,
        id: int | None = None,
        data: dict[Any, Any] | None = None,
        type: str | None = None,
    ):
        self.new = False

        if id is None:
            global next_new_edge
            id = next_new_edge
            next_new_edge = next_new_edge - 1
            self.new = True
            Edge.cache_control.cache[id] = self

        self.id = id
        self.type = type
        self.data = data or {}
        self.src_id = src_id
        self.dst_id = dst_id

    def __del__(self):
        Edge.save(self)

    @property
    def src(self) -> Node:
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        return Node.get(self.dst_id)

    @staticmethod
    @cached(cache=LRUCache(settings.edge_cache_size), key=lambda id: id, info=True)
    def get(id: int) -> Edge:
        return Edge.load(id)

    @staticmethod
    def load(id: int) -> Edge:
        db = GraphDB()
        edge_list = list(
            db.raw_query(f"MATCH (n)-[e]-(m) WHERE id(e) = {id} RETURN e LIMIT 1", fetch=True)
        )
        if not len(edge_list) == 1:
            raise Exception(f"Couldn't find edge ID: {id}")

        e = edge_list[0]["e"]
        props = None
        if hasattr(e, "_properties"):
            props = e._properties
        return Edge(e._start_node_id, e._end_node_id, id=id, data=props, type=e._type)

    @staticmethod
    def save(e: Edge) -> Edge:
        if e.new:
            return Edge.create(e)
        else:
            return Edge.update(e)

    @staticmethod
    def create(e: Edge) -> Edge:
        return e
        # Node.save(e.src)
        # Node.save(e.dst)
        # db.raw_query()

    @staticmethod
    def update(e: Edge) -> Edge:
        return e

    @staticmethod
    def to_id(e: Edge | int) -> int:
        if isinstance(e, Edge):
            return e.id
        else:
            return e


class EdgeFetchIterator:
    """
    The implementation of an iterator for an EdgeList. Only intended to be used internally by
    EdgeList.
    """

    def __init__(self, edge_set: list[int]):
        self.__edge_set = edge_set
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur >= len(self.__edge_set):
            raise StopIteration

        id = self.__edge_set[self.cur]
        self.cur = self.cur + 1
        return Edge.get(id)


class EdgeList(MutableSet[Edge | int], Mapping[int, Edge]):
    """
    A list of Edges that is used by Node for keeping track of the connections it has.
    Implements interfaces for both a MutableSet (i.e. set()) and a Mapping (i.e. read-only list())
    """

    def __init__(self, ids: list[int] | set[int]):
        self.__edges: list[int] = list(ids)

    def __iter__(self):
        return EdgeFetchIterator(self.__edges)

    def __getitem__(self, key: int) -> Edge:
        return Edge.get(self.__edges[key])

    def __len__(self):
        return len(self.__edges)

    def add(self, e: Edge | int) -> None:
        e_id = Edge.to_id(e)

        if e_id in self.__edges:
            return

        self.__edges.append(e_id)

    def discard(self, e: Edge | int) -> None:
        e_id = Edge.to_id(e)

        self.__edges.remove(e_id)

    def __contains__(self, e: Any) -> bool:
        if isinstance(e, Edge) or type(e) == int:
            e_id = Edge.to_id(e)
        else:
            return False

        return e_id in self.__edges


next_new_node = -1


class NodeMeta(type):
    @property
    def cache_control(cls):
        return CacheControl[Node](Node.get)


class Node(metaclass=NodeMeta):
    """
    An graph database node that automatically handles CRUD for the underlying graph database objects
    """

    def __init__(
        self,
        *,
        id: int | None = None,
        src_edges: EdgeList | None = None,
        dst_edges: EdgeList | None = None,
        data: dict[Any, Any] | None = None,
        labels: set[str] | list[str] | None = None,
    ):
        self.new = False

        if id is None:
            global next_new_node
            id = next_new_node
            next_new_node = next_new_node - 1
            self.new = True
            Node.cache_control.cache[id] = self

        self.id = id
        self.data = data or {}
        if isinstance(labels, set):
            labels = list(labels)
        self.labels = labels or list()
        self._orig_labels = set(self.labels)
        self.src_edges = src_edges or EdgeList([])
        self.dst_edges = dst_edges or EdgeList([])

    def __del__(self):
        Node.save(self)

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

        # print("RES", res)

        if not len(res) >= 1:
            raise Exception(f"Couldn't find node ID: {id}")

        n = res[0]["n"]
        edges = list(
            map(lambda r: {"id": r["e_id"], "start": r["e_start"], "end": r["e_end"]}, res)
        )
        src_edges = list(map(lambda e: e["id"], filter(lambda e: e["start"] == id, edges)))
        dst_edges = list(map(lambda e: e["id"], filter(lambda e: e["end"] == id, edges)))
        # reveal_type(n)
        return Node(
            id=id,
            src_edges=EdgeList(src_edges),
            dst_edges=EdgeList(dst_edges),
            data=n._properties,
            labels=n._labels,
        )

    @cached(cache=LRUCache(settings.node_cache_size), key=lambda id: id, info=True)
    @staticmethod
    def get(id: int) -> Node:
        return Node.load(id)

    @staticmethod
    def save(n: Node) -> Node:
        if n.new:
            return Node.create(n)
        else:
            return Node.update(n)

    @staticmethod
    def update(n: Node) -> Node:
        db = GraphDB()

        orig_labels = n._orig_labels
        curr_labels = set(n.labels)
        new_labels = curr_labels - orig_labels
        rm_labels = orig_labels - curr_labels
        set_label_str = Node.mklabels(list(new_labels))
        if set_label_str:
            set_query = f"SET n{set_label_str}, n = $props"
        else:
            set_query = "SET n = $props"
        rm_label_str = Node.mklabels(list(rm_labels))
        if rm_label_str:
            rm_query = f"REMOVE n{rm_label_str}"
        else:
            rm_query = ""

        params = {"props": n.data}

        db.raw_query(
            f"MATCH (n) WHERE id(n) = {n.id} {set_query} {rm_query}", params=params, fetch=False
        )

        return n

    @staticmethod
    def create(n: Node) -> Node:
        db = GraphDB()

        label_str = Node.mklabels(n.labels)
        params = {"props": n.data}

        res = list(
            db.raw_query(
                f"CREATE (n{label_str} $props) RETURN id(n) as id", params=params, fetch=True
            )
        )

        if not len(res) >= 1:
            raise Exception(f"Couldn't find node ID: {id}")
        new_id = res[0]["id"]
        n.id = new_id
        n.new = False
        # TODO: update edges with new ID

        return n

    @staticmethod
    def connect(src: int | Node, dst: int | Node) -> Edge:
        if isinstance(src, Node):
            src_id = src.id
        else:
            src_id = src

        if isinstance(dst, Node):
            dst_id = dst.id
        else:
            dst_id = dst

        e = Edge(src_id, dst_id)
        src_node = Node.get(src_id)
        dst_node = Node.get(dst_id)
        src_node.src_edges.add(e)
        dst_node.dst_edges.add(e)
        return e

    @staticmethod
    def mklabels(labels: list[str]) -> str:
        "Converts a list of strings into proper Cypher syntax for a graph database query"
        label_str = ":".join([i for i in labels])
        if len(label_str) > 0:
            label_str = ":" + label_str
        return label_str
