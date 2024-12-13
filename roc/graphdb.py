"""This module is a wrapper around a graph database and abstracts away all the
database-specific features as various classes (GraphDB, Node, Edge, etc)
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterator, Mapping, MutableSet
from itertools import islice
from typing import Any, Callable, Generic, Literal, NewType, TypeVar, cast

import mgclient
import networkx as nx
from cachetools import LRUCache
from pydantic import BaseModel, Field, create_model, field_validator
from typing_extensions import Self

from .config import Config
from .logger import logger

RecordFn = Callable[[str, Iterator[Any]], None]
CacheType = TypeVar("CacheType")
CacheId = TypeVar("CacheId")
EdgeId = NewType("EdgeId", int)
NodeId = NewType("NodeId", int)
NodeType = TypeVar("NodeType", bound="Node")
next_new_edge: EdgeId = cast(EdgeId, -1)
next_new_node: NodeId = cast(NodeId, -1)


def true_filter(_: Any) -> bool:
    """Helper function that accepts any value and returns True. Great for
    default filters.
    """
    return True


def no_callback(_: Any) -> None:
    """Helper function that accepts any value and returns None. Great for
    default callback functions.
    """
    pass


class ErrorSavingDuringDelWarning(Warning):
    """An error that occurs while saving a Node during __del__"""

    pass


class GraphDBInternalError(Exception):
    """An generic exception for unexpected errors"""

    pass


#########
# GRAPHDB
#########
graph_db_singleton: GraphDB | None = None

QueryParamType = dict[str, Any]


class GraphDB:
    """A graph database singleton. Settings for the graph database come from the config module."""

    def __init__(self) -> None:
        settings = Config.get()
        self.host = settings.db_host
        self.port = settings.db_port
        self.encrypted = settings.db_conn_encrypted
        self.username = settings.db_username
        self.password = settings.db_password
        self.lazy = settings.db_lazy
        self.client_name = "roc-graphdb-client"
        self.db_conn = self.connect()
        self.closed = False

    def raw_fetch(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Executes a Cypher query and returns the results as an iterator of
        dictionaries. Used for any query that has a 'RETURN' clause.

        Args:
            query (str): The Cypher query to execute
            params (dict[str, Any] | None, optional): Any parameters to pass to
                the query. Defaults to None. See also: https://memgraph.com/docs/querying/expressions#parameters

        Yields:
            Iterator[dict[str, Any]]: An iterator of the results from the database.
        """
        params = params or {}
        logger.trace(f"raw_fetch: '{query}' *** with params: *** '{params}")

        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield {dsc.name: row[index] for index, dsc in enumerate(cursor.description)}

    def raw_execute(self, query: str, *, params: dict[str, Any] | None = None) -> None:
        """Executes a query with no return value. Used for 'SET', 'DELETE' or
        other queries without a 'RETURN' clause.

        Args:
            query (str): The Cypher query to execute
            params (dict[str, Any] | None, optional): Any parameters to pass to
                the query. Defaults to None. See also: https://memgraph.com/docs/querying/expressions#parameters
        """
        params = params or {}
        logger.trace(f"raw_execute: '{query}' *** with params: *** '{params}'")

        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        cursor.fetchall()

    def connected(self) -> bool:
        """Returns True if the database is connected, False otherwise"""
        return self.db_conn is not None and self.db_conn.status == mgclient.CONN_STATUS_READY

    def connect(self) -> mgclient.Connection:
        """Connects to the database and returns a Connection object"""
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

    def close(self) -> None:
        """Closes the connection to the database"""
        self.db_conn.close()
        self.closed = True

    @classmethod
    def singleton(cls) -> GraphDB:
        """This returns a singleton object for the graph database. If the
        singleton isn't created yet, it creates it.
        """
        global graph_db_singleton
        if not graph_db_singleton:
            graph_db_singleton = GraphDB()

        assert graph_db_singleton.closed is False
        return graph_db_singleton

    @staticmethod
    def to_networkx(
        db: GraphDB | None = None,
        node_ids: set[NodeId] | None = None,
        filter: NodeFilterFn | None = None,
    ) -> nx.DiGraph:
        """Converts the entire graph database (and local cache of objects) into
        a NetworkX graph

        Args:
            db (GraphDB | None, optional): The database to convert to NetworkX.
                Defaults to the GraphDB singleton if not specified.
            node_ids (set[NodeId] | None, optional): The NodeIDs to add to the
                NetworkX graph. Defaults to all IDs if not specified.
            filter (NodeFilterFn | None, optional): A Node filter to filter out
                nodes before adding them to the NetworkX graph. Also useful for a
                callback that can be used for progress updates. Defaults to None.

        Returns:
            nx.DiGraph: _description_
        """
        db = db or GraphDB.singleton()
        node_ids = node_ids or Node.all_ids(db=db)
        filter = filter or true_filter
        G = nx.DiGraph()

        def nx_add(n: Node) -> None:
            n_data = Node.to_dict(n, include_labels=True)

            # TODO: this converts labels to a string, but maybe there's a better
            # way to preserve the list so that it can be used for filtering in
            # external programs
            if "labels" in n_data and isinstance(n_data["labels"], set):
                n_data["labels"] = ", ".join(n_data["labels"])

            G.add_node(n.id, **n_data)

            for e in n.src_edges:
                e_data = Edge.to_dict(e, include_type=True)
                G.add_edge(e.src_id, e.dst_id, **e_data)

        # iterate all specified node_ids, adding all of them to the nx graph
        def nx_add_many(nodes: list[Node]) -> None:
            for n in nodes:
                if filter(n):
                    nx_add(n)

        Node.get_many(node_ids, load_edges=True, progress_callback=nx_add_many)

        return G


#######
# CACHE
#######
CacheKey = TypeVar("CacheKey")
CacheValue = TypeVar("CacheValue")
CacheDefault = TypeVar("CacheDefault")


class GraphCache(LRUCache[CacheKey, CacheValue], Generic[CacheKey, CacheValue]):
    """A generic cache that is used for both the Node cache and the Edge cache"""

    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.hits = 0
        self.misses = 0

    def __str__(self) -> str:
        return f"Size: {self.currsize}/{self.maxsize} ({self.currsize/self.maxsize*100:1.2f}%), Hits: {self.hits}, Misses: {self.misses}"

    def get(  # type: ignore [override]
        self,
        key: CacheKey,
        /,
        default: CacheValue | None = None,
    ) -> CacheValue | None:
        """Uses the specified CacheKey to fetch an object from the cache.

        Args:
            key (CacheKey): The key to use to fetch the object
            default (CacheValue | None, optional): If the object isn't found,
                the default value to return. Defaults to None.

        Returns:
            CacheValue | None: The object from the cache, or None if not found.
        """
        v = super().get(key)
        if not v:
            self.misses = self.misses + 1
            if self.currsize == self.maxsize:
                logger.warning(
                    f"Cache miss and cache is full ({self.currsize}/{self.maxsize}). Cache may start thrashing and performance may be impaired."
                )
        else:
            self.hits = self.hits + 1
        return v

    def clear(self) -> None:
        """Clears out all items from the cache and resets the cache
        statistics
        """
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
    """An edge (a.k.a. Relationship or Connection) between two Nodes. An edge obect automatically
    implements all phases of CRUD in the underlying graph database. This is a directional
    relationship with a "source" and "destination". The source and destination properties
    are dynamically loaded through property getters when they are called, and may trigger
    a graph database query if they don't already exist in the edge cache.
    """

    id: EdgeId = Field(exclude=True)
    # XXX: type, src_id, and dst_id used to be pydantic literals, but updating
    # the pydantic version broke them
    type: str = Field(exclude=True)
    src_id: NodeId = Field(exclude=True)
    dst_id: NodeId = Field(exclude=True)
    _no_save = False
    _new = False
    _deleted = False

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

        if self.id < 0:
            self._new = True
            Edge.get_cache()[self.id] = self

    def __del__(self) -> None:
        # print("Edge.__del__:", self)
        Edge.save(self)

    def __repr__(self) -> str:
        return f"Edge({self.id} [{self.src_id}>>{self.dst_id}])"

    @classmethod
    def get_cache(self) -> EdgeCache:
        global edge_cache
        if edge_cache is None:
            settings = Config.get()
            edge_cache = EdgeCache(maxsize=settings.edge_cache_size)

        return edge_cache

    @classmethod
    def get(cls, id: EdgeId, *, db: GraphDB | None = None) -> Self:
        """Looks up an Edge based on it's ID. If the Edge is cached, the cached edge is returned;
        otherwise the Edge is queried from the graph database based the ID provided and a new
        Edge is returned and cached.

        Args:
            id (EdgeId): the unique identifier for the Edge
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: returns the Edge requested by the id
        """
        cache = Edge.get_cache()
        e = cache.get(id)
        if not e:
            e = cls.load(id, db=db)
            cache[id] = e

        return cast(Self, e)

    @classmethod
    def load(cls, id: EdgeId, *, db: GraphDB | None = None) -> Self:
        """Loads an Edge from the graph database without attempting to check if the Edge
        already exists in the cache. Typically this is only called by Edge.get()

        Args:
            id (EdgeId): the unique identifier of the Edge to fetch
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Raises:
            EdgeNotFound: if the specified ID does not exist in the cache or the database

        Returns:
            Self: returns the Edge requested by the id
        """
        db = db or GraphDB.singleton()
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
    def save(cls, e: Self, *, db: GraphDB | None = None) -> Self:
        """Saves the edge to the database. Calls Edge.create if the edge is new, or Edge.update if
        edge already exists in the database.

        Args:
            e (Self): The edge to save
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: The same edge that was passed in, for convenience. The Edge may be updated with a
            new identifier if it was newly created in the database.
        """
        if e._new:
            return cls.create(e, db=db)
        else:
            return cls.update(e, db=db)

    @classmethod
    def create(cls, e: Self, *, db: GraphDB | None = None) -> Self:
        """Creates a new edge in the database. Typically only called by Edge.save

        Args:
            e (Self): The edge to create
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Raises:
            EdgeCreateFailed: Failed to write the edge to the database, for eample
                if the ID is wrong.

        Returns:
            Self: the edge that was created, with an updated identifier and other chagned attributes
        """
        if e._no_save or e.src._no_save or e.dst._no_save:
            return e

        db = db or GraphDB.singleton()
        old_id = e.id

        if e.src._new:
            Node.save(e.src)

        if e.dst._new:
            Node.save(e.dst)

        params = {"props": Edge.to_dict(e)}

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
    def update(cls, e: Self, *, db: GraphDB | None = None) -> Self:
        """Updates the edge in the database. Typically only called by Edge.save

        Args:
            e (Self): The edge to update
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: The same edge that was passed in, for convenience
        """
        if e._no_save:
            return e

        db = db or GraphDB.singleton()

        params = {"props": Edge.to_dict(e)}

        db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} SET e = $props", params=params)

        return e

    @staticmethod
    def delete(e: Edge, *, db: GraphDB | None = None) -> None:
        """Deletes the specified edge from the database. If the edge has not already been persisted
        to the database, this marks the edge as deleted and returns.

        Args:
            e (Edge): The edge to delete
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton
        """
        e._deleted = True
        e._no_save = True
        db = db or GraphDB.singleton()

        # remove e from src and dst nodes
        e.src.src_edges.discard(e)
        e.dst.dst_edges.discard(e)

        # remove from cache
        edge_cache = Edge.get_cache()
        if e.id in edge_cache:
            del edge_cache[e.id]

        # delete from db
        if not e._new:
            db.raw_execute(f"MATCH ()-[e]->() WHERE id(e) = {e.id} DELETE e")

    @staticmethod
    def to_dict(e: Edge, include_type: bool = False) -> dict[str, Any]:
        """Convert a Edge to a Python dictionary"""
        ret = e.model_dump()
        if include_type and hasattr(e, "type"):
            ret["type"] = e.type
        return ret

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
    """The implementation of an iterator for an EdgeList. Only intended to be used internally by
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


EdgeFilter = Callable[[Edge], bool] | str | EdgeId | None


class EdgeList(MutableSet[Edge | EdgeId], Mapping[int, Edge]):
    """A list of Edges that is used by Node for keeping track of the connections it has.
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
        """Adds a new Edge to the list"""
        e_id = Edge.to_id(e)

        if e_id in self.__edges:
            return

        self.__edges.append(e_id)

    def discard(self, e: Edge | EdgeId) -> None:
        """Removes an edge to the list"""
        e_id = Edge.to_id(e)

        self.__edges.remove(e_id)

    def replace(self, old: Edge | EdgeId, new: Edge | EdgeId) -> None:
        """Replaces all instances of an old Edge with a new Edge. Useful for when an Edge is
        persisted to the graph database and its permanent ID is assigned
        """
        old_id = Edge.to_id(old)
        new_id = Edge.to_id(new)
        for i in range(len(self.__edges)):
            if self.__edges[i] == old_id:
                self.__edges[i] = new_id

    def count(self, f: EdgeFilter = None) -> int:
        return len(self.get_edges(f))

    def get_edges(self, f: EdgeFilter = None) -> list[Edge]:
        if not f:
            return list(self.__iter__())

        if isinstance(f, str):
            s = f
            f = lambda e: e.type == s

        if isinstance(f, int):
            n = f
            f = lambda e: e.id == n

        return list(filter(f, self.__iter__()))


#######
# NODE
#######
class NodeNotFound(Exception):
    """An exception raised when trying to retreive a Node that doesn't exist."""

    pass


class NodeCreationFailed(Exception):
    """An exception raised when trying to create a Node in the graph database fails"""

    pass


def get_next_new_node_id() -> NodeId:
    global next_new_node
    id = next_new_node
    next_new_node = cast(NodeId, next_new_node - 1)
    return id


class Node(BaseModel, extra="allow"):
    """An graph database node that automatically handles CRUD for the underlying graph database objects"""

    _id: NodeId
    labels: set[str] = Field(exclude=True, default_factory=set)
    _orig_labels: set[str]
    _src_edges: EdgeList
    _dst_edges: EdgeList
    _db: GraphDB
    _new = False
    _no_save = False
    _deleted = False

    @property
    def id(self) -> NodeId:
        return self._id

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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # set passed-in private values or their defaults
        self._db = kwargs["_db"] if "_db" in kwargs else GraphDB.singleton()
        self._id = kwargs["_id"] if "_id" in kwargs else get_next_new_node_id()
        self._src_edges = kwargs["_src_edges"] if "_src_edges" in kwargs else EdgeList([])
        self._dst_edges = kwargs["_dst_edges"] if "_dst_edges" in kwargs else EdgeList([])

        if self.id < 0:
            self._new = True  # TODO: derived?
            Node.get_cache()[self.id] = self

        self._orig_labels = self.labels.copy()

    def __del__(self) -> None:
        # print("Node.__del__:", self)
        try:
            self.__class__.save(self, db=self._db)
        except Exception as e:
            err_msg = f"error saving during del: {e}"
            # logger.warning(err_msg)
            warnings.warn(err_msg, ErrorSavingDuringDelWarning)

    def __repr__(self) -> str:
        return f"Node({self.id})"

    def __str__(self) -> str:
        return f"Node({self.id}, labels={self.labels})"

    @classmethod
    def load(cls, id: NodeId, *, db: GraphDB | None = None) -> Self:
        """Loads a node from the database. Use `Node.get` or other methods instead.

        Args:
            id (NodeId): The identifier of the node to fetch
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Raises:
            NodeNotFound: The node specified by the identifier does not exist in the database
            GraphDBInternalError: If the requested ID returns multiple nodes

        Returns:
            Self: The node from the database
        """
        res = cls.load_many({id}, db=db)

        # print("RES", res)

        if len(res) < 1:
            raise NodeNotFound(f"Couldn't find node ID: {id}")

        if len(res) > 1:
            raise GraphDBInternalError(
                f"Too many nodes returned while trying to load single node: {id}"
            )

        return res[0]

    @classmethod
    def load_many(
        cls,
        node_set: set[NodeId],
        db: GraphDB | None = None,
        load_edges: bool = False,
    ) -> list[Self]:
        db = db or GraphDB.singleton()
        node_ids = ",".join(map(str, node_set))

        ret = cls.find(
            where=f"id(src) IN [{node_ids}]",  # TODO: use params?
            db=db,
            load_edges=load_edges,
        )

        if len(ret) != len(node_set):
            id_set = {n.id for n in ret}
            missing_ids = node_set - id_set
            raise NodeNotFound(f"Couldn't find node IDs: {', '.join(map(str, missing_ids))}")

        return ret

    @classmethod
    def find(
        cls,
        where: str,
        src_node_name: str = "src",
        src_labels: set[str] = set(),
        edge_name: str = "e",
        edge_type: str = "",
        params: QueryParamType = dict(),
        db: GraphDB | None = None,
        load_edges: bool = False,
        params_to_str: bool = True,
    ) -> list[Self]:
        db = db or GraphDB.singleton()

        if load_edges:
            edge_fmt = f"{edge_name}"
        else:
            edge_fmt = f"{{id: id({edge_name}), start: id(startNode({edge_name})), end: id(endNode({edge_name}))}}"

        if len(src_labels) == 0:
            src_label_str = ""
        else:
            src_label_str = f":{':'.join(src_labels)}"

        if len(edge_type) > 0:
            edge_type = ":" + edge_type

        if params_to_str:
            for k in params.keys():
                params[k] = str(params[k])

        res_iter = db.raw_fetch(
            f"""
                MATCH ({src_node_name}{src_label_str})-[{edge_name}{edge_type}*0..1]-() 
                WITH {src_node_name}, head({edge_name}) AS {edge_name}
                WHERE {where}
                RETURN {src_node_name} AS n, collect({edge_fmt}) AS edges
                """,
            params=params,
        )

        ret_list = list()
        for r in res_iter:
            logger.trace(f"find result: {r}")
            n = r["n"]
            if n is None:
                # NOTE: I can't think of any circumstances where there would be
                # multiple "None" results, so I think this is just an empty list
                continue

            if load_edges:
                # XXX: memgraph converts edges to Relationship objects if you
                # return the whole edge
                src_edges = list()
                dst_edges = list()
                edge_cache = Edge.get_cache()
                for e in r["edges"]:
                    # add edge_id to to the right list for the node creation below
                    if n.id == e.start_id:
                        src_edges.append(e.id)
                    else:
                        dst_edges.append(e.id)

                    # edge already loaded, continue to next one
                    if e.id in edge_cache:
                        continue

                    # create a new edge
                    props = None
                    if hasattr(e, "properties"):
                        props = e.properties
                    new_edge = Edge(
                        e.start_id,
                        e.end_id,
                        id=e.id,
                        data=props,
                        type=e.type,
                    )
                    edge_cache[e.id] = new_edge
            else:
                # edges are just the IDs
                src_edges = [e["id"] for e in r["edges"] if e["start"] == n.id]
                dst_edges = [e["id"] for e in r["edges"] if e["end"] == n.id]

            node_cache = cls.get_cache()
            if n.id in node_cache:
                new_node = cast(Self, node_cache[n.id])
            else:
                mkcls = cls
                cls_lbls = frozenset(n.labels)
                if cls is Node and cls_lbls in node_registry:
                    mkcls = node_registry[cls_lbls]
                new_node = mkcls(
                    _id=n.id,
                    _src_edges=EdgeList(src_edges),
                    _dst_edges=EdgeList(dst_edges),
                    labels=n.labels,
                    **n.properties,
                )
                node_cache[n.id] = new_node
            ret_list.append(new_node)

        return ret_list

    @classmethod
    def find_one(
        cls,
        where: str,
        src_node_name: str = "src",
        src_labels: set[str] = set(),
        edge_name: str = "e",
        edge_type: str = "",
        params: QueryParamType = dict(),
        db: GraphDB | None = None,
        load_edges: bool = False,
        params_to_str: bool = True,
        exactly_one: bool = False,
    ) -> Self | None:
        """Finds a single Node.find results down to a single node. Raises an
        exception of the list contains more than one node.

        Args:
            nodes (Sequence[NodeType]): The list of nodes returned by Node.find

        Raises:
            Exception: Raised if there is more than 1 node in the list

        Returns:
            NodeType | None: Returns None if the list is empty, or the node in the list.
        """
        nodes = cls.find(
            where=where,
            src_node_name=src_node_name,
            src_labels=src_labels,
            edge_name=edge_name,
            edge_type=edge_type,
            params=params,
            db=db,
            load_edges=load_edges,
            params_to_str=params_to_str,
        )

        match len(nodes):
            case 0:
                if exactly_one:
                    raise Exception("expect exactly one node in find_one")
                return None
            case 1:
                return nodes[0]
            case _:
                raise Exception("expected zero or one node in find_one")

    @classmethod
    def get_many(
        cls,
        node_ids: set[NodeId],
        *,
        batch_size: int = 128,
        db: GraphDB | None = None,
        load_edges: bool = False,
        return_nodes: bool = False,
        progress_callback: ProgressFn | None = None,
    ) -> list[Node]:
        db = db or GraphDB.singleton()

        c = Node.get_cache()
        if len(node_ids) > c.maxsize:
            raise GraphDBInternalError(
                f"get_many attempting to load more nodes than cache size ({len(node_ids)} > {c.maxsize})"
            )

        cache_ids = set(c.keys())
        fetch_ids = node_ids - cache_ids

        start = 0
        curr = batch_size
        ret_list = [c[nid] for nid in c]
        if progress_callback:
            progress_callback(ret_list)
        while start < len(fetch_ids):
            id_set = set(islice(fetch_ids, start, curr))

            res = cls.load_many(id_set, db=db, load_edges=load_edges)
            for n in res:
                c[n.id] = n

            if progress_callback:
                progress_callback(res)

            ret_list.extend(res)
            # import pprint
            # pprint.pp(list(res))
            # print(f"got {len(list(res))} nodes")

            start = curr
            curr += batch_size

        assert len(ret_list) == len(node_ids)
        return ret_list

    @classmethod
    def get_cache(cls) -> NodeCache:
        global node_cache
        if node_cache is None:
            settings = Config.get()
            node_cache = NodeCache(settings.node_cache_size)

        return node_cache

    @classmethod
    def get(cls, id: NodeId, *, db: GraphDB | None = None) -> Self:
        """Returns a cached node with the specified id. If no node is cached, it is retrieved from
        the database.


        Args:
            id (NodeId): The unique identifier of the node to fetch
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: the cached or newly retrieved node
        """
        cache = Node.get_cache()
        n = cache.get(id)
        if n is None:
            n = cls.load(id, db=db)

        return cast(Self, n)

    @classmethod
    def save(cls, n: Self, *, db: GraphDB | None = None) -> Self:
        """Save a node to persistent storage

        Writes the specified node to the GraphDB for persistent storage. If the node does not
        already exist in storage, it is created via the `create` method. If the node does exist, it
        is updated via the `update` method.

        If the _no_save flag is True on the node, the save request will be silently ignored.

        Args:
            n (Self): The Node to be saved
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: As a convenience, the node that was stored is returned. This may be useful
            since the the id of the node may change if it was created in the database.
        """
        if n._new:
            return cls.create(n, db=db)
        else:
            return cls.update(n, db=db)

    @classmethod
    def update(cls, n: Self, *, db: GraphDB | None = None) -> Self:
        """Update an existing node in the GraphDB.

        Calling `save` is preferred to using this method so that the caller doesn't need to know the
        state of the node.

        Args:
            n (Self): The node to be updated
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Self: The node that was passed in, for convenience
        """
        if n._no_save:
            return n

        db = db or GraphDB.singleton()

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

        params = {"props": Node.to_dict(n)}

        db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} {set_query} {rm_query}", params=params)

        return n

    @classmethod
    def create(cls, n: Self, *, db: GraphDB | None = None) -> Self:
        """Creates the specified node in the GraphDB.

        Calling `save` is preferred to using this method so that the caller doesn't need to know the
        state of the node.

        Args:
            n (Self): the node to be created
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Raises:
            NodeCreationFailed: if creating the node failed in the database

        Returns:
            Self: the node that was passed in, albeit with a new `id` and potenitally other new
            fields
        """
        if n._no_save:
            return n

        db = db or GraphDB.singleton()
        old_id = n.id

        label_str = Node.mklabels(n.labels)
        params = {"props": Node.to_dict(n)}

        res = list(db.raw_fetch(f"CREATE (n{label_str} $props) RETURN id(n) as id", params=params))

        if not len(res) >= 1:
            raise NodeCreationFailed(f"Couldn't create node ID: {id}")

        new_id = res[0]["id"]
        n._id = new_id
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
    def connect(
        cls,
        src: NodeId | Self,
        dst: NodeId | Self,
        type: str,
        *,
        db: GraphDB | None = None,
    ) -> Edge:
        """Connects two nodes (creates an Edge between two nodes)

        Args:
            src (NodeId | Node): _description_
            dst (NodeId | Node): _description_
            type (str): _description_
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

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
        src_node = cls.get(src_id, db=db)
        dst_node = cls.get(dst_id, db=db)
        src_node.src_edges.add(e)
        dst_node.dst_edges.add(e)
        return e

    @staticmethod
    def delete(n: Node, *, db: GraphDB | None = None) -> None:
        db = db or GraphDB.singleton()

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
            db.raw_execute(f"MATCH (n) WHERE id(n) = {n.id} DELETE n")

        n._deleted = True
        n._no_save = True

    @staticmethod
    def to_dict(n: Node, include_labels: bool = False) -> dict[str, Any]:
        """Convert a Node to a Python dictionary"""
        # XXX: the excluded fields below shouldn't have been included in the
        # first place because Pythonic should exclude fields with underscores
        ret = n.model_dump(exclude={"_id", "_src_edges", "_dst_edges"})

        if include_labels and hasattr(n, "labels"):
            ret["labels"] = n.labels

        return ret

    @staticmethod
    def mklabels(labels: set[str]) -> str:
        """Converts a list of strings into proper Cypher syntax for a graph database query"""
        labels_list = [i for i in labels]
        labels_list.sort()
        label_str = ":".join(labels_list)
        if len(label_str) > 0:
            label_str = ":" + label_str
        return label_str

    @staticmethod
    def all_ids(db: GraphDB | None = None) -> set[NodeId]:
        """Returns an exhaustive Set of all NodeIds that exist in both the graph
        database and the NodeCache
        """
        db = db or GraphDB.singleton()

        # get all NodeIds in the cache
        c = Node.get_cache()
        cached_ids = set(c.keys())

        # get all NodeIds in the database
        db_ids = {n["id"] for n in db.raw_fetch("MATCH (n) RETURN id(n) as id")}

        # return the combination of both
        return db_ids.union(cached_ids)

    @staticmethod
    def walk(
        n: Node,
        *,
        mode: WalkMode = "both",
        edge_filter: EdgeFilterFn | None = None,
        # edge_callback: EdgeCallbackFn | None = None,
        node_filter: NodeFilterFn | None = None,
        node_callback: NodeCallbackFn | None = None,
        _walk_history: set[int] | None = None,
    ) -> None:
        # if we have walked this node before, just return
        _walk_history = _walk_history or set()
        if n.id in _walk_history:
            return
        _walk_history.add(n.id)

        edge_filter = edge_filter or true_filter
        node_filter = node_filter or true_filter
        # edge_callback = edge_callback or no_callback
        node_callback = node_callback or no_callback

        # callback for this node, if not filtered
        if node_filter(n):
            node_callback(n)
        else:
            return

        if mode == "src" or mode == "both":
            for e in n.src_edges:
                if edge_filter(e):
                    Node.walk(
                        e.dst,
                        mode=mode,
                        edge_filter=edge_filter,
                        # edge_callback=edge_callback,
                        node_filter=node_filter,
                        node_callback=node_callback,
                        _walk_history=_walk_history,
                    )

        if mode == "dst" or mode == "both":
            for e in n.dst_edges:
                if edge_filter(e):
                    Node.walk(
                        e.src,
                        mode=mode,
                        edge_filter=edge_filter,
                        # edge_callback=edge_callback,
                        node_filter=node_filter,
                        node_callback=node_callback,
                        _walk_history=_walk_history,
                    )


node_registry: dict[frozenset[str], type] = {}


def register_node(*args: str) -> Callable[[type[NodeType]], type[NodeType]]:
    def node_register_decorator(cls: type[NodeType]) -> type[NodeType]:
        lbls = frozenset(args)
        # lbls = frozenset({c.__name__ for c in MyModel.__mro__ if c.__name__ not in ["BaseModel", "object"]})

        WrapperClass = create_model(
            "WrapperClass",
            __base__=cls,
            labels=(set, Field(exclude=True, default_factory=lambda: set(lbls))),
        )

        # class WrapperClass(cls):  # type: ignore
        #     labels: set[str] = Field(default=set(lbls))

        functools.update_wrapper(WrapperClass, cls, updated=())
        node_registry[lbls] = WrapperClass

        return WrapperClass

    return node_register_decorator


WalkMode = Literal["src", "dst", "both"]
NodeFilterFn = Callable[[Node], bool]
EdgeFilterFn = Callable[[Edge], bool]
ProgressFn = Callable[[list[Node]], None]
NodeCallbackFn = Callable[[Node], None]
EdgeCallbackFn = Callable[[Edge], None]

NodeCache = GraphCache[NodeId, Node]
node_cache: NodeCache | None = None
