from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableSet
from threading import Lock
from typing import Any, Callable, Generic, NamedTuple, NewType, TypeVar, cast

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

    def raw_fetch(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        params = params or {}
        print(f"raw_fetch: '{query}' *** with params: *** '{params}")

        if self.record_callback:
            self.record_callback(query, self.db.execute_and_fetch(query, parameters=params))

        ret = self.db.execute_and_fetch(query, parameters=params)
        return ret  # type: ignore

    def raw_execute(self, query: str, *, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        print(f"raw_execute: '{query}' *** with params: *** '{params}'")
        self.db.execute(query, parameters=params)


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


EdgeId = NewType("EdgeId", int)
NodeId = NewType("NodeId", int)
next_new_edge: EdgeId = cast(EdgeId, -1)


class EdgeNotFound(Exception):
    pass


class EdgeCreateFailed(Exception):
    pass


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
        src_id: NodeId,
        dst_id: NodeId,
        type: str,
        *,
        id: EdgeId | None = None,
        data: dict[Any, Any] | None = None,
    ):
        self.new = False
        self.no_save = False

        if id is None:
            global next_new_edge
            id = next_new_edge
            next_new_edge = cast(EdgeId, next_new_edge - 1)
            self.new = True
            Edge.cache_control.cache[id] = self

        self.id: EdgeId = id
        self.__type = type
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

    @property
    def type(self) -> str:
        return self.__type

    @staticmethod
    @cached(cache=LRUCache(settings.edge_cache_size), key=lambda id: id, info=True)
    def get(id: EdgeId) -> Edge:
        """Looks up an Edge based on it's ID. If the Edge is cached, the cached edge is returned;
        otherwise the Edge is queried from the graph database based the ID provided and a new
        Edge is returned and cached.

        Args:
            id (EdgeId): the unique identifier for the Edge

        Returns:
            Edge: returns the Edge requested by the id
        """
        return Edge.load(id)

    @staticmethod
    def load(id: EdgeId) -> Edge:
        """Loads an Edge from the graph database without attempting to check if the Edge
        already exists in the cache. Typically this is only called by Edge.get()

        Args:
            id (EdgeId): the unique identifier of the Edge to fetch

        Raises:
            EdgeNotFound: if the specified ID does not exist in the cache or the database

        Returns:
            Edge: returns the Edge requested by the id
        """
        db = GraphDB()
        edge_list = list(db.raw_fetch(f"MATCH (n)-[e]-(m) WHERE id(e) = {id} RETURN e LIMIT 1"))
        if not len(edge_list) == 1:
            raise EdgeNotFound(f"Couldn't find edge ID: {id}")

        e = edge_list[0]["e"]
        props = None
        if hasattr(e, "_properties"):
            props = e._properties
        return Edge(
            e._start_node_id,
            e._end_node_id,
            id=id,
            data=props,
            type=e._type,
        )

    @staticmethod
    def save(e: Edge) -> Edge:
        if e.new:
            return Edge.create(e)
        else:
            return Edge.update(e)

    @staticmethod
    def create(e: Edge) -> Edge:
        if e.no_save:
            return e

        db = GraphDB()
        old_id = e.id

        if e.src.new:
            Node.save(e.src)

        if e.dst.new:
            Node.save(e.dst)

        params = {"props": e.data}

        ret = list(
            db.raw_fetch(
                f"""
                MATCH (src), (dst)
                WHERE id(src) = {e.src_id} AND id(dst) = {e.dst_id} 
                CREATE (src)-[e:{e.type} $props]->(dst)
                RETURN id(e) as e_id
                """,
                params=params,
            )
        )

        if len(ret) != 1:
            raise EdgeCreateFailed("failed to create new edge")

        e.id = ret[0]["e_id"]
        e.new = False
        # update the cache; if being called during __del__ then the cache entry may not exist
        try:
            cache = Edge.cache_control.cache
            del cache[old_id]
            cache[e.id] = e
        except KeyError:
            pass
        # update references to edge id
        e.src.src_edges.replace(old_id, e.id)
        e.dst.dst_edges.replace(old_id, e.id)

        return e

    @staticmethod
    def update(e: Edge) -> Edge:
        if e.no_save:
            return e

        db = GraphDB()

        params = {"props": e.data}

        db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} SET e = $props", params=params)

        return e

    @staticmethod
    def to_id(e: Edge | EdgeId) -> EdgeId:
        if isinstance(e, Edge):
            return e.id
        else:
            return e


class EdgeFetchIterator:
    """
    The implementation of an iterator for an EdgeList. Only intended to be used internally by
    EdgeList.
    """

    def __init__(self, edge_list: list[EdgeId]):
        self.__edge_list = edge_list
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur >= len(self.__edge_list):
            raise StopIteration

        id = self.__edge_list[self.cur]
        self.cur = self.cur + 1
        return Edge.get(id)


class EdgeList(MutableSet[Edge | EdgeId], Mapping[EdgeId, Edge]):
    """
    A list of Edges that is used by Node for keeping track of the connections it has.
    Implements interfaces for both a MutableSet (i.e. set()) and a Mapping (i.e. read-only list())
    """

    def __init__(self, ids: list[EdgeId] | set[EdgeId]):
        self.__edges: list[EdgeId] = list(ids)

    def __iter__(self):
        return EdgeFetchIterator(self.__edges)

    def __getitem__(self, key: int) -> Edge:
        return Edge.get(self.__edges[key])

    def __len__(self):
        return len(self.__edges)

    def add(self, e: Edge | EdgeId) -> None:
        e_id = Edge.to_id(e)

        if e_id in self.__edges:
            return

        self.__edges.append(e_id)

    def discard(self, e: Edge | EdgeId) -> None:
        e_id = Edge.to_id(e)

        self.__edges.remove(e_id)

    def __contains__(self, e: Any) -> bool:
        if isinstance(e, Edge) or type(e) == int:
            e_id = Edge.to_id(e)  # type: ignore
        else:
            return False

        return e_id in self.__edges

    def replace(self, old: Edge | EdgeId, new: Edge | EdgeId) -> None:
        old_id = Edge.to_id(old)
        new_id = Edge.to_id(new)
        for i in range(len(self.__edges)):
            if self.__edges[i] == old_id:
                self.__edges[i] = new_id


next_new_node: NodeId = cast(NodeId, -1)


class NodeNotFound(Exception):
    pass


class NodeCreationFailed(Exception):
    pass


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
        id: NodeId | None = None,
        src_edges: EdgeList | None = None,
        dst_edges: EdgeList | None = None,
        data: dict[Any, Any] | None = None,
        labels: set[str] | list[str] | None = None,
    ):
        self.new = False
        self.no_save = False

        if id is None:
            global next_new_node
            id = next_new_node
            next_new_node = cast(NodeId, next_new_node - 1)
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
    def load(id: NodeId) -> Node:
        db = GraphDB()
        res = list(
            db.raw_fetch(
                f"""
                MATCH (n)-[e]-(m) WHERE id(n) = {id}
                RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
                """,
            )
        )

        # print("RES", res)

        if not len(res) >= 1:
            raise NodeNotFound(f"Couldn't find node ID: {id}")

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
    def get(id: NodeId) -> Node:
        return Node.load(id)

    @staticmethod
    def save(n: Node) -> Node:
        if n.new:
            return Node.create(n)
        else:
            return Node.update(n)

    @staticmethod
    def update(n: Node) -> Node:
        if n.no_save:
            return n

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

        db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} {set_query} {rm_query}", params=params)

        return n

    @staticmethod
    def create(n: Node) -> Node:
        if n.no_save:
            return n

        db = GraphDB()
        old_id = n.id

        label_str = Node.mklabels(n.labels)
        params = {"props": n.data}

        res = list(db.raw_fetch(f"CREATE (n{label_str} $props) RETURN id(n) as id", params=params))

        if not len(res) >= 1:
            raise NodeCreationFailed(f"Couldn't find node ID: {id}")

        new_id = res[0]["id"]
        n.id = new_id
        n.new = False
        # update the cache; if being called during __del__ then the cache entry may not exist
        try:
            cache = Node.cache_control.cache
            del cache[old_id]
            cache[new_id] = n
        except KeyError:
            pass

        for e in n.src_edges:
            assert e.src_id == old_id
            e.src_id = new_id

        for e in n.dst_edges:
            assert e.dst_id == old_id
            e.dst_id = new_id

        return n

    @staticmethod
    def connect(src: NodeId | Node, dst: NodeId | Node, type: str) -> Edge:
        if isinstance(src, Node):
            src_id = src.id
        else:
            src_id = src

        if isinstance(dst, Node):
            dst_id = dst.id
        else:
            dst_id = dst

        e = Edge(src_id, dst_id, type)
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
