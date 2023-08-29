from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, MutableSet
from typing import Any, Callable, Generic, NewType, TypeVar, cast

import mgclient
from cachetools import LRUCache
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from .config import get_setting
from .logger import logger

RecordFn = Callable[[str, Iterator[Any]], None]
CacheType = TypeVar("CacheType")
CacheId = TypeVar("CacheId")
EdgeId = NewType("EdgeId", int)
NodeId = NewType("NodeId", int)
next_new_edge: EdgeId = cast(EdgeId, -1)
next_new_node: NodeId = cast(NodeId, -1)


class ErrorSavingDuringDelWarning(Warning):
    pass


#########
# GRAPHDB
#########
# graph_db_singleton: GraphDB | None = None


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
        self.host = get_setting("db_host", str)
        self.port = get_setting("db_port", int)
        self.encrypted = get_setting("db_conn_encrypted", bool)
        self.username = get_setting("db_username", str)
        self.password = get_setting("db_password", str)
        self.lazy = get_setting("db_lazy", bool)
        self.client_name = "roc-graphdb-client"
        self.db_conn = self.connect()

    def raw_fetch(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        params = params or {}
        logger.trace(f"raw_fetch: '{query}' *** with params: *** '{params}")

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
        logger.trace(f"raw_execute: '{query}' *** with params: *** '{params}'")
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

    # @classmethod
    # def singleton(cls) -> GraphDB:
    #     global graph_db_singleton
    #     if not graph_db_singleton:
    #         graph_db_singleton = GraphDB()

    #     return graph_db_singleton


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


#######
# CACHE
#######
CacheKey = TypeVar("CacheKey")
CacheValue = TypeVar("CacheValue")
CacheDefault = TypeVar("CacheDefault")


class GraphCache(LRUCache[CacheKey, CacheValue], Generic[CacheKey, CacheValue]):
    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.hits = 0
        self.misses = 0

    def get(  # type: ignore [override]
        self,
        key: CacheKey,
        /,
        default: CacheValue | None = None,
    ) -> CacheValue | None:
        v = super().get(key)
        if not v:
            self.misses = self.misses + 1
        else:
            self.hits = self.hits + 1
        return v

    def clear(self) -> None:
        super().clear()
        self.hits = 0
        self.misses = 0


#######
# EDGE
#######
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

    @field_validator("id", mode="before")
    def default_id(cls, id: EdgeId | None) -> EdgeId:
        if isinstance(id, int):
            return id

        return get_next_new_edge_id()

    @property
    def src(self) -> Node:
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        return Node.get(self.dst_id)

    @property
    def new(self) -> bool:
        return self._new

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

        self._new = False
        self._no_save = False
        self._deleted = False

        if self.id < 0:
            self._new = True
            Edge.get_cache()[self.id] = self

    def __del__(self) -> None:
        Edge.save(self)

    # @cached(cache=
    # LRUCache(get_setting("edge_cache_size", int)), key=lambda cls, id: id, info=True)

    @classmethod
    def get_cache(self) -> EdgeCache:
        global edge_cache
        if edge_cache is None:
            edge_cache = EdgeCache(maxsize=get_setting("edge_cache_size", int))

        return edge_cache

    @classmethod
    def get(cls, id: EdgeId) -> Self:
        """Looks up an Edge based on it's ID. If the Edge is cached, the cached edge is returned;
        otherwise the Edge is queried from the graph database based the ID provided and a new
        Edge is returned and cached.

        Args:
            id (EdgeId): the unique identifier for the Edge

        Returns:
            Self: returns the Edge requested by the id
        """
        cache = Edge.get_cache()
        e = cache.get(id)
        if not e:
            e = cls.load(id)
            cache[id] = e

        return cast(Self, e)

    @classmethod
    def load(cls, id: EdgeId) -> Self:
        """Loads an Edge from the graph database without attempting to check if the Edge
        already exists in the cache. Typically this is only called by Edge.get()

        Args:
            id (EdgeId): the unique identifier of the Edge to fetch

        Raises:
            EdgeNotFound: if the specified ID does not exist in the cache or the database

        Returns:
            Self: returns the Edge requested by the id
        """
        db = GraphDB()
        edge_list = list(db.raw_fetch(f"MATCH (n)-[e]-(m) WHERE id(e) = {id} RETURN e LIMIT 1"))
        if not len(edge_list) == 1:
            raise EdgeNotFound(f"Couldn't find edge ID: {id}")

        e = edge_list[0]["e"]
        props = None
        if hasattr(e, "properties"):
            props = e.properties
        return cls(
            e.start_id,
            e.end_id,
            id=id,
            data=props,
            type=e.type,
        )

    @classmethod
    def save(cls, e: Self) -> Self:
        """Saves the edge to the database. Calls Edge.create if the edge is new, or Edge.update if
        edge already exists in the database.

        Args:
            e (Self): The edge to save

        Returns:
            Self: The same edge that was passed in, for convenience. The Edge may be updated with a
            new identifier if it was newly created in the database.
        """
        if e._new:
            return cls.create(e)
        else:
            return cls.update(e)

    @classmethod
    def create(cls, e: Self) -> Self:
        """Creates a new edge in the database. Typically only called by Edge.save

        Args:
            e (Self): The edge to create

        Raises:
            EdgeCreateFailed: Failed to write the edge to the database, for eample
                if the ID is wrong.

        Returns:
            Self: the edge that was created, with an updated identifier and other chagned attributes
        """
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
            cache = Edge.get_cache()
            del cache[old_id]
            cache[e.id] = e
        except KeyError:
            pass
        # update references to edge id
        e.src.src_edges.replace(old_id, e.id)
        e.dst.dst_edges.replace(old_id, e.id)

        return e

    @classmethod
    def update(cls, e: Self) -> Self:
        """Updates the edge in the database. Typically only called by Edge.save

        Args:
            e (Self): The edge to update

        Returns:
            Self: The same edge that was passed in, for convenience
        """
        if e._no_save:
            return e

        db = GraphDB()

        params = {"props": e.model_dump()}

        db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} SET e = $props", params=params)

        return e

    @staticmethod
    def delete(e: Edge) -> None:
        """Deletes the specified edge from the database. If the edge has not already been persisted
        to the database, this marks the edge as deleted and returns.

        Args:
            e (Edge): The edge to delete
        """
        e._deleted = True
        e._no_save = True

        # remove e from src and dst nodes
        e.src.src_edges.discard(e)
        e.dst.dst_edges.discard(e)

        # remove from cache
        edge_cache = Edge.get_cache()
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


EdgeCache = GraphCache[EdgeId, Edge]
edge_cache: EdgeCache | None = None


#######
# EDGE LIST
#######
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

    def __contains__(self, e: Any) -> bool:
        if isinstance(e, Edge) or isinstance(e, int):
            e_id = Edge.to_id(e)  # type: ignore
        else:
            return False

        return e_id in self.__edges

    def add(self, e: Edge | EdgeId) -> None:
        e_id = Edge.to_id(e)

        if e_id in self.__edges:
            return

        self.__edges.append(e_id)

    def discard(self, e: Edge | EdgeId) -> None:
        e_id = Edge.to_id(e)

        self.__edges.remove(e_id)

    def replace(self, old: Edge | EdgeId, new: Edge | EdgeId) -> None:
        old_id = Edge.to_id(old)
        new_id = Edge.to_id(new)
        for i in range(len(self.__edges)):
            if self.__edges[i] == old_id:
                self.__edges[i] = new_id


#######
# NODE
#######
class NodeNotFound(Exception):
    pass


class NodeCreationFailed(Exception):
    pass


def get_next_new_node_id() -> NodeId:
    global next_new_node
    id = next_new_node
    next_new_node = cast(NodeId, next_new_node - 1)
    return id


class Node(BaseModel, extra="allow"):
    """
    An graph database node that automatically handles CRUD for the underlying graph database objects
    """

    id: NodeId = Field(exclude=True)
    labels: set[str] = Field(exclude=True)

    @field_validator("id", mode="before")
    def default_id(cls, id: NodeId | None) -> NodeId:
        if isinstance(id, int):
            return id

        return get_next_new_node_id()

    @field_validator("labels", mode="before")
    def default_labels(cls, labels: list[str] | set[str] | None) -> set[str]:
        if not labels:
            return set()

        if isinstance(labels, list):
            return set(labels)

        return labels

    @property
    def src_edges(self) -> EdgeList:
        return self._src_edges

    @property
    def dst_edges(self) -> EdgeList:
        return self._dst_edges

    @property
    def new(self) -> bool:
        return self._new

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

        self._new = False
        self._no_save = False
        self._deleted = False

        if self.id < 0:
            self._new = True  # TODO: derived?
            Node.get_cache()[self.id] = self

        self._orig_labels = self.labels.copy()
        self._src_edges = src_edges or EdgeList([])
        self._dst_edges = dst_edges or EdgeList([])
        # TODO: ignore fields on save
        # self._ignored_fields = ["new", "no_save", "deleted"]

    def __del__(self) -> None:
        try:
            self.__class__.save(self)
        except Exception as e:
            err_msg = f"error saving during del: {e}"
            logger.warning(err_msg)
            warnings.warn(err_msg, ErrorSavingDuringDelWarning)

    @classmethod
    def load(cls, id: NodeId) -> Self:
        """Loads a node from the database. Use `Node.get` or other methods instead.

        Args:
            id (NodeId): The identifier of the node to fetch

        Raises:
            NodeNotFound: The node specified by the identifier does not exist in the database

        Returns:
            Self: The node from the database
        """

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
        return cls(
            id=id,
            src_edges=EdgeList(src_edges),
            dst_edges=EdgeList(dst_edges),
            labels=n.labels,
            data=n.properties,
        )

    @classmethod
    def get_cache(cls) -> NodeCache:
        global node_cache
        if node_cache is None:
            node_cache = NodeCache(get_setting("node_cache_size", int))

        return node_cache

    @classmethod
    def get(cls, id: NodeId) -> Self:
        """Returns a cached node with the specified id. If no node is cached, it is retrieved from
        the database.


        Args:
            id (NodeId): The unique identifier of the node to fetch

        Returns:
            Self: the cached or newly retrieved node
        """
        cache = Node.get_cache()
        n = cache.get(id)
        if not n:
            n = cls.load(id)
            cache[id] = n

        return cast(Self, n)

    @classmethod
    def save(cls, n: Self) -> Self:
        """Save a node to persistent storage

        Writes the specified node to the GraphDB for persistent storage. If the node does not
        already exist in storage, it is created via the `create` method. If the node does exist, it
        is updated via the `update` method.

        If the _no_save flag is True on the node, the save request will be silently ignored.

        Args:
            n (Self): The Node to be saved

        Returns:
            Self: As a convenience, the node that was stored is returned. This may be useful
            since the the id of the node may change if it was created in the database.
        """
        if n._new:
            return cls.create(n)
        else:
            return cls.update(n)

    @classmethod
    def update(cls, n: Self) -> Self:
        """Update an existing node in the GraphDB.

        Calling `save` is preferred to using this method so that the caller doesn't need to know the
        state of the node.

        Args:
            n (Self): The node to be updated

        Returns:
            Self: The node that was passed in, for convenience
        """
        if n._no_save:
            return n

        db = GraphDB()

        orig_labels = n._orig_labels
        curr_labels = set(n.labels)
        new_labels = curr_labels - orig_labels
        rm_labels = orig_labels - curr_labels
        set_label_str = Node.mklabels(new_labels)
        if set_label_str:
            set_query = f"SET n{set_label_str}, n = $props"
        else:
            set_query = "SET n = $props"
        rm_label_str = Node.mklabels(rm_labels)
        if rm_label_str:
            rm_query = f"REMOVE n{rm_label_str}"
        else:
            rm_query = ""

        params = {"props": n.model_dump()}

        db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} {set_query} {rm_query}", params=params)

        return n

    @classmethod
    def create(cls, n: Self) -> Self:
        """Creates the specified node in the GraphDB.

        Calling `save` is preferred to using this method so that the caller doesn't need to know the
        state of the node.

        Args:
            n (Self): the node to be created

        Raises:
            NodeCreationFailed: if creating the node failed in the database

        Returns:
            Self: the node that was passed in, albeit with a new `id` and potenitally other new
            fields
        """
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
        # update the cache; if being called during c then the cache entry may not exist
        try:
            cache = Node.get_cache()
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

    @classmethod
    def connect(cls, src: NodeId | Self, dst: NodeId | Self, type: str) -> Edge:
        """Connects two nodes (creates an Edge between two nodes)

        Args:
            src (NodeId | Node): _description_
            dst (NodeId | Node): _description_
            type (str): _description_

        Returns:
            Edge: _description_
        """
        if isinstance(src, Node):
            src_id = src.id
        else:
            src_id = src

        if isinstance(dst, Node):
            dst_id = dst.id
        else:
            dst_id = dst

        e = Edge(src_id, dst_id, type)
        src_node = cls.get(src_id)
        dst_node = cls.get(dst_id)
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
        node_cache = Node.get_cache()
        if n.id in node_cache:
            del node_cache[n.id]

        if not n._new:
            db = GraphDB()
            db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} DELETE n")

        n._deleted = True
        n._no_save = True

    @staticmethod
    def mklabels(labels: set[str]) -> str:
        "Converts a list of strings into proper Cypher syntax for a graph database query"
        labels_list = [i for i in labels]
        labels_list.sort()
        label_str = ":".join(labels_list)
        if len(label_str) > 0:
            label_str = ":" + label_str
        return label_str


NodeCache = GraphCache[NodeId, Node]
node_cache: NodeCache | None = None
