from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableSet
from threading import Lock
from typing import Any, Callable, Generic, NamedTuple, NewType, TypeVar, cast

import mgclient
from cachetools import Cache, LRUCache, cached
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from roc.config import settings

RecordFn = Callable[[str, Iterator[Any]], None]


class GraphDB:
    """
    A graph database singleton. Settings for the graph database come from the config module.
    """

    def __new__(cls) -> GraphDB:
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance.__isinitialized = False  # type: ignore
        return cls.instance

    def __init__(self) -> None:
        if self.__isinitialized:  # type: ignore
            return

        self.__isinitialized = True
        self.host = settings.db_host
        self.port = settings.db_port
        self.encrypted = settings.db_conn_encrypted
        self.username = settings.db_username or ""
        self.password = settings.db_password or ""
        self.lazy = settings.db_lazy
        self.client_name = "roc-graphdb-client"
        self.db_conn = self.connect()
        # self.record_callback: RecordFn | None = None

    def raw_fetch(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        params = params or {}
        print(f"raw_fetch: '{query}' *** with params: *** '{params}")

        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield {
                dsc.name: _convert_memgraph_value(row[index])
                for index, dsc in enumerate(cursor.description)
            }

    def raw_execute(self, query: str, *, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        print(f"raw_execute: '{query}' *** with params: *** '{params}'")
        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        cursor.fetchall()

    def connected(self) -> bool:
        return self.db_conn is not None and self.db_conn.status == mgclient.CONN_STATUS_READY

    def connect(self) -> mgclient.Connection:
        sslmode = mgclient.MG_SSLMODE_REQUIRE if self.encrypted else mgclient.MG_SSLMODE_DISABLE
        connection = mgclient.connect(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            sslmode=sslmode,
            lazy=self.lazy,
            client_name=self.client_name,
        )
        connection.autocommit = True
        return connection


# XXX: copied from GQLAlchemy
def _convert_memgraph_value(value: Any) -> Any:
    """Converts Memgraph objects to custom Node/Relationship objects."""
    # if isinstance(value, mgclient.Relationship):
    #     return Relationship.parse_obj(
    #         {
    #             "_type": value.type,
    #             "_id": value.id,
    #             "_start_node_id": value.start_id,
    #             "_end_node_id": value.end_id,
    #             **value.properties,
    #         }
    #     )

    # if isinstance(value, mgclient.Node):
    #     return Node.parse_obj(
    #         {
    #             "_id": value.id,
    #             "_labels": set(value.labels),
    #             **value.properties,
    #         }
    #     )

    # if isinstance(value, mgclient.Path):
    #     return Path.parse_obj(
    #         {
    #             "_nodes": list([_convert_memgraph_value(node) for node in value.nodes]),
    #             "_relationships": list(
    #                 [_convert_memgraph_value(rel) for rel in value.relationships]
    #             ),
    #         }
    #     )

    return value


CacheType = TypeVar("CacheType")
CacheId = TypeVar("CacheId")


class CacheInfo(NamedTuple):
    """
    Information about the cache: hits, misses, max size, and current size.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class NodeCacheControlAttr:
    def __get__(self, instance: Any, owner: Any) -> CacheControl[Node, NodeId]:
        return CacheControl[Node, NodeId](Node.get)


class EdgeCacheControlAttr:
    def __get__(self, instance: Any, owner: Any) -> CacheControl[Edge, EdgeId]:
        return CacheControl[Edge, EdgeId](Edge.get)


class CacheControl(Generic[CacheType, CacheId]):
    """
    For controlling the Node and Edge caches, such as clearing them, getting their current or max
    size, adding items to the cache (outside the automatic methods for adding cached items), etc.
    """

    node_cache_control = NodeCacheControlAttr()
    edge_cache_control = EdgeCacheControlAttr()

    def __init__(self, cache_fn: Any):
        self.cache: Cache[CacheId, CacheType] = cache_fn.cache
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


def get_next_new_edge_id() -> EdgeId:
    global next_new_edge
    id = next_new_edge
    next_new_edge = cast(EdgeId, next_new_edge - 1)

    return id


class Edge(BaseModel, extra="allow"):
    """
    An edge (a.k.a. Relationship or Connection) between two Nodes. An edge obect automatically
    implements all phases of CRUD in the underlying graph database. This is a directional
    relationship with a "source" and "destination". The source and destination properties
    are dynamically loaded through property getters when they are called, and may trigger
    a graph database query if they don't already exist in the edge cache.
    """

    id: EdgeId = Field(exclude=True)
    type: str = Field(literal=True, exclude=True)
    src_id: NodeId = Field(literal=True, exclude=True)
    dst_id: NodeId = Field(literal=True, exclude=True)

    # TODO: lifecycle to privattr
    _new: bool = False
    _no_save: bool = False
    _deleted: bool = False

    @field_validator("id", mode="before")
    def default_id(cls, id: EdgeId | None) -> EdgeId:
        if isinstance(id, int):
            return id

        return get_next_new_edge_id()

    def __init__(
        self,
        src_id: NodeId,
        dst_id: NodeId,
        type: str,
        *,
        id: EdgeId | None = None,
        data: dict[Any, Any] | None = None,
    ):
        data = data or {}
        super().__init__(
            src_id=src_id,
            dst_id=dst_id,
            type=type,
            id=id,
            **data,
        )

        if self.id < 0:
            self._new = True
            CacheControl.edge_cache_control.cache[self.id] = self

    def __del__(self) -> None:
        Edge.save(self)

    @property
    def src(self) -> Node:
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        return Node.get(self.dst_id)

    @property
    def new(self) -> bool:
        return self._new

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
        if hasattr(e, "properties"):
            props = e.properties
        return Edge(
            e.start_id,
            e.end_id,
            id=id,
            data=props,
            type=e.type,
        )

    @staticmethod
    def save(e: Edge) -> Edge:
        if e._new:
            return Edge.create(e)
        else:
            return Edge.update(e)

    @staticmethod
    def create(e: Edge) -> Edge:
        if e._no_save:
            return e

        db = GraphDB()
        old_id = e.id

        if e.src._new:
            Node.save(e.src)

        if e.dst._new:
            Node.save(e.dst)

        params = {"props": e.model_dump()}

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
        e._new = False
        # update the cache; if being called during __del__ then the cache entry may not exist
        try:
            cache = CacheControl.edge_cache_control.cache
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
        if e._no_save:
            return e

        db = GraphDB()

        params = {"props": e.model_dump()}

        db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} SET e = $props", params=params)

        return e

    @staticmethod
    def delete(e: Edge) -> None:
        e._deleted = True
        e._no_save = True

        # remove e from src and dst nodes
        e.src.src_edges.discard(e)
        e.dst.dst_edges.discard(e)

        # remove from cache
        edge_cache = CacheControl.edge_cache_control.cache
        if e.id in edge_cache:
            del edge_cache[e.id]

        # delete from db
        if not e._new:
            db = GraphDB()
            db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} DELETE e")

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

    def __iter__(self) -> EdgeFetchIterator:
        return self

    def __next__(self) -> Edge:
        if self.cur >= len(self.__edge_list):
            raise StopIteration

        id = self.__edge_list[self.cur]
        self.cur = self.cur + 1
        return Edge.get(id)


class EdgeList(MutableSet[Edge | EdgeId], Mapping[int, Edge]):
    """
    A list of Edges that is used by Node for keeping track of the connections it has.
    Implements interfaces for both a MutableSet (i.e. set()) and a Mapping (i.e. read-only list())
    """

    def __init__(self, ids: list[EdgeId] | set[EdgeId]):
        self.__edges: list[EdgeId] = list(ids)

    def __iter__(self) -> EdgeFetchIterator:
        return EdgeFetchIterator(self.__edges)

    def __getitem__(self, key: int) -> Edge:
        return Edge.get(self.__edges[key])

    def __len__(self) -> int:
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


def get_next_new_node_id() -> NodeId:
    global next_new_node
    id = next_new_node
    next_new_node = cast(NodeId, next_new_node - 1)
    return id


class Node(BaseModel):
    # class Node(BaseModel):
    """
    An graph database node that automatically handles CRUD for the underlying graph database objects
    """

    model_config = ConfigDict(extra="allow")
    id: NodeId = Field(exclude=True)
    # TODO: set[str]
    labels: list[str] = Field(exclude=True)
    _new: bool = PrivateAttr(default=False)
    # new: bool = Field(default=False)
    _no_save: bool = PrivateAttr(default=False)
    _deleted: bool = PrivateAttr(default=False)

    @field_validator("id", mode="before")
    def default_id(cls, id: NodeId | None) -> NodeId:
        if isinstance(id, int):
            return id

        return get_next_new_node_id()

    @field_validator("labels", mode="before")
    def default_labels(cls, labels: list[str] | set[str] | None) -> list[str]:
        if not labels:
            return []

        if isinstance(labels, set):
            return list(labels)

        return labels

    def __init__(
        self,
        *,
        id: NodeId | None = None,
        data: dict[Any, Any] | None = None,
        labels: set[str] | list[str] | None = None,
        src_edges: EdgeList | None = None,
        dst_edges: EdgeList | None = None,
    ):
        data = data or {}
        super().__init__(
            id=id,
            labels=labels,
            **data,
        )

        if self.id < 0:
            self._new = True  # TODO: derived?
            CacheControl.node_cache_control.cache[self.id] = self

        self._orig_labels = set(self.labels)
        self._src_edges = src_edges or EdgeList([])
        self._dst_edges = dst_edges or EdgeList([])
        # TODO: ignore fields on save
        # self._ignored_fields = ["new", "no_save", "deleted"]

    @property
    def src_edges(self) -> EdgeList:
        return self._src_edges

    @property
    def dst_edges(self) -> EdgeList:
        return self._dst_edges

    @property
    def new(self) -> bool:
        return self._new

    def __del__(self) -> None:
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

        print("RES", res)

        if not len(res) >= 1:
            raise NodeNotFound(f"Couldn't find node ID: {id}")

        n = res[0]["n"]
        edges = list(
            map(lambda r: {"id": r["e_id"], "start": r["e_start"], "end": r["e_end"]}, res)
        )
        src_edges = list(map(lambda e: e["id"], filter(lambda e: e["start"] == id, edges)))
        dst_edges = list(map(lambda e: e["id"], filter(lambda e: e["end"] == id, edges)))
        return Node(
            id=id,
            src_edges=EdgeList(src_edges),
            dst_edges=EdgeList(dst_edges),
            labels=n.labels,
            data=n.properties,
        )

    @cached(cache=LRUCache(settings.node_cache_size), key=lambda id: id, info=True)
    @staticmethod
    def get(id: NodeId) -> Node:
        return Node.load(id)

    @staticmethod
    def save(n: Node) -> Node:
        if n._new:
            return Node.create(n)
        else:
            return Node.update(n)

    @staticmethod
    def update(n: Node) -> Node:
        if n._no_save:
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

        params = {"props": n.model_dump()}

        db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} {set_query} {rm_query}", params=params)

        return n

    @staticmethod
    def create(n: Node) -> Node:
        if n._no_save:
            return n

        db = GraphDB()
        old_id = n.id

        label_str = Node.mklabels(n.labels)
        params = {"props": n.model_dump()}

        res = list(db.raw_fetch(f"CREATE (n{label_str} $props) RETURN id(n) as id", params=params))

        if not len(res) >= 1:
            raise NodeCreationFailed(f"Couldn't find node ID: {id}")

        new_id = res[0]["id"]
        n.id = new_id
        n._new = False
        # update the cache; if being called during __del__ then the cache entry may not exist
        try:
            cache = CacheControl.node_cache_control.cache
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
    def delete(n: Node) -> None:
        # remove edges
        for e in n.src_edges:
            Edge.delete(e)

        for e in n.dst_edges:
            Edge.delete(e)

        # remove from cache
        node_cache = CacheControl.node_cache_control.cache
        if n.id in node_cache:
            del node_cache[n.id]

        if not n._new:
            db = GraphDB()
            db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} DELETE n")

        n._deleted = True
        n._no_save = True

    @staticmethod
    def mklabels(labels: list[str]) -> str:
        "Converts a list of strings into proper Cypher syntax for a graph database query"
        label_str = ":".join([i for i in labels])
        if len(label_str) > 0:
            label_str = ":" + label_str
        return label_str
