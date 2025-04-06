"""This module is a wrapper around a graph database and abstracts away all the
database-specific features as various classes (GraphDB, Node, Edge, etc)
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import time
import warnings
from collections.abc import Collection, Iterable, Iterator, MutableSet, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import islice
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Literal,
    NewType,
    TypeGuard,
    TypeVar,
    _SpecialForm,
    cast,
    overload,
)

import mgclient
import networkx as nx
import scipy as sp
from cachetools import LRUCache
from networkx.drawing.nx_pydot import write_dot
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from tqdm import tqdm
from typing_extensions import Self

from .config import Config
from .logger import logger
from .reporting.observability import Observability

RecordFn = Callable[[str, Iterator[Any]], None]
CacheType = TypeVar("CacheType")
CacheId = TypeVar("CacheId")
EdgeId = NewType("EdgeId", int)
NodeId = NewType("NodeId", int)
EdgeType = TypeVar("EdgeType", bound="Edge")
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


class ErrorSavingDuringDelWarning(Warning):
    """An error that occurs while saving a Node during __del__"""


class GraphDBInternalError(Exception):
    """An generic exception for unexpected errors"""


class StrictSchemaWarning(Warning):
    """A warning that strict schema mode is enabled, but there was a violation"""


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
        self.strict_schema = settings.db_strict_schema
        self.strict_schema_warns = settings.db_strict_schema_warns
        self.client_name = "roc-graphdb-client"
        self.db_conn = self.connect()
        self.closed = False
        self.query_counter = Observability.meter.create_counter(
            "roc.graphdb.query",
            unit="query",
            description="the total number of queries database",
        )
        self.node_counter = Observability.meter.create_counter(
            "roc.graphdb.nodes",
            unit="nodes",
            description="the total number of nodes",
        )
        self.edge_counter = Observability.meter.create_counter(
            "roc.graphdb.edges",
            unit="edges",
            description="the total number of edges",
        )

        if self.strict_schema:
            Schema.validate()

    @Observability.tracer.start_as_current_span("graphdb.fetch")
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
        self.query_counter.add(1, attributes={"type": "fetch"})
        params = params or {}
        logger.trace(f"raw_fetch: '{query}' *** with params: *** '{params}")

        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield {dsc.name: row[index] for index, dsc in enumerate(cursor.description)}

    @Observability.tracer.start_as_current_span("graphdb.execute")
    def raw_execute(self, query: str, *, params: dict[str, Any] | None = None) -> None:
        """Executes a query with no return value. Used for 'SET', 'DELETE' or
        other queries without a 'RETURN' clause.

        Args:
            query (str): The Cypher query to execute
            params (dict[str, Any] | None, optional): Any parameters to pass to
                the query. Defaults to None. See also: https://memgraph.com/docs/querying/expressions#parameters
        """
        self.query_counter.add(1, attributes={"type": "execute"})
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

    @staticmethod
    def flush(db: GraphDB | None = None) -> None:
        """Saves any cached nodes and edges back to the graph database"""
        Node.get_cache().flush()
        Edge.get_cache().flush()

    @staticmethod
    def export(
        format: str = "gml",
        filename: str = "graph",
        timestamp: bool = True,
        db: GraphDB | None = None,
    ) -> None:
        """Saves the graph database to a file in the selected format

        Args:
            format (str, optional): The graph format to use which is passed
                along to networkx. Options include "gexf", "gml", "dot",
                "graphml", "json-node-link", "json-adj", "cytoscape", "pajek",
                "matrix-market", "adj-list", "multi-adj-list", "edge-list".
                Defaults to "gml".
            filename (str, optional): The filename to write the graph to. Defaults to "graph".
            timestamp (bool, optional): Whether or not to append a timestamp to
                the end of the file name. Defaults to True.
            db (GraphDB, optional): The GraphDB to export from.
        """
        db = db or GraphDB.singleton()
        ids = Node.all_ids()
        logger.info(f"Saving {len(ids)} nodes...")
        start_time = time.time()

        # tqdm options: https://github.com/tqdm/tqdm?tab=readme-ov-file#parameters
        with tqdm(total=len(ids), desc="Nodes", unit="node", ncols=80, colour="blue") as pbar:

            def progress_update(n: Node) -> bool:
                pbar.update(1)
                return True

            G = GraphDB.to_networkx(node_ids=ids, filter=progress_update, db=db)

        # format timestamp
        if timestamp:
            # time format: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
            timestr = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
            filename = f"{filename}-{timestr}"

        logger.info(f"Writing graph to '{filename}'...")
        match format:
            case "gexf":
                nx.write_gexf(G, f"{filename}.gexf")
            case "gml":
                nx.write_gml(G, f"{filename}.gml")
            case "dot":
                # XXX: pydot uses the 'name' attribute internally, so rename ours if it exists
                for n in G.nodes(data=True):
                    if "name" in n[1]:
                        n[1]["nme"] = n[1]["name"]
                        del n[1]["name"]
                write_dot(G, f"{filename}.dot")
            case "graphml":
                nx.write_graphml(G, f"{filename}.graphml")
            case "json-node-link":
                with open(f"{filename}.node-link.json", "w", encoding="utf8") as f:
                    json.dump(nx.node_link_data(G), f)
            case "json-adj":
                with open(f"{filename}.adj.json", "w", encoding="utf8") as f:
                    json.dump(nx.adjacency_data(G), f)
            case "cytoscape":
                with open(f"{filename}.cytoscape.json", "w", encoding="utf8") as f:
                    json.dump(nx.cytoscape_data(G), f)
            case "pajek":
                nx.write_pajek(G, f"{filename}.pajek")
            case "matrix-market":
                np_graph = nx.to_numpy_array(G)
                sp.io.mmwrite(f"{filename}.mm", np_graph)
            case "adj-list":
                nx.write_adjlist(G, f"{filename}.adjlist")
            case "multi-adj-list":
                nx.write_multiline_adjlist(G, f"{filename}.madjlist")
            case "edge-list":
                nx.write_edgelist(G, f"{filename}.edges")

        end_time = time.time()

        nc = Node.get_cache()
        ec = Edge.get_cache()
        assert len(nc) == len(ids)
        logger.info(
            f"Saved {len(ids)} nodes and {len(ec)} edges. Elapsed time: {timedelta(seconds=(end_time - start_time))}"
        )

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


class GraphCache(LRUCache[CacheKey, CacheValue], Generic[CacheKey, CacheValue]):
    """A generic cache that is used for both the Node cache and the Edge cache"""

    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.hits = 0
        self.misses = 0

    def __str__(self) -> str:
        return f"Size: {self.currsize}/{self.maxsize} ({self.currsize / self.maxsize * 100:1.2f}%), Hits: {self.hits}, Misses: {self.misses}"

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

    def flush(self) -> None:
        """Flushes the cache by saving every Node and Edge"""
        node_ids = {n_id for n_id in self}
        while len(node_ids) > 0:
            cache_id = node_ids.pop()
            cache_item = self[cache_id]
            if isinstance(cache_item, Node):
                cache_item.__class__.save(cache_item)
            elif isinstance(cache_item, Edge):
                cache_item.__class__.save(cache_item)


#######
# EDGE
#######
class EdgeNotFound(Exception):
    """Error raised when attempting to look up a specific Edge and no result
    is returned
    """


class EdgeCreateFailed(Exception):
    """Error raised when creating a new Edge fails"""


def _get_next_new_edge_id() -> EdgeId:
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

    _id: EdgeId
    type: str = Field(exclude=True)
    src_id: NodeId = Field(exclude=True)
    dst_id: NodeId = Field(exclude=True)
    allowed_connections: EdgeConnectionsList | None = Field(exclude=True, default=None)
    _no_save = False
    _new = False
    _deleted = False

    @property
    def id(self) -> EdgeId:
        """The unique identfier for the Edge, as defined by the underlying graph database."""
        return self._id

    @property
    def src(self) -> Node:
        """The ID for the Node at the source side of the Edge."""
        return Node.get(self.src_id)

    @property
    def dst(self) -> Node:
        """The ID for the Node at the destination side of the Edge."""
        return Node.get(self.dst_id)

    @property
    def new(self) -> bool:
        """Whether this Edge is new, or has been previously saved."""
        return self._new

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # set passed-in values or their defaults
        self._id = kwargs["_id"] if "_id" in kwargs else _get_next_new_edge_id()
        self._db = kwargs["_db"] if "_db" in kwargs else GraphDB.singleton()

        if self._id < 0:
            self._new = True
            Edge.get_cache()[self.id] = self

        self._db.edge_counter.add(1, attributes={"new": self._new, "type": self.type})

    def __del__(self) -> None:
        # logger.trace(f"Edge.__del__: {self}")
        Edge.save(self)

    def __repr__(self) -> str:
        return f"Edge({self.id} [{self.src_id}>>{self.dst_id}])"

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        if not hasattr(cls, "type"):
            cls.type = Field(exclude=True, default_factory=lambda: cls.__name__)
            edgetype = cls.__name__
        else:
            # XXX: not sure why this makes mypy angry here but not in Node.__init_subclass__
            if isinstance(cls.type, FieldInfo):  # type: ignore
                edgetype = cls.type.get_default(call_default_factory=True)  # type: ignore
            else:
                edgetype = cls.type

        if edgetype in edge_registry:
            raise Exception(
                f"edge_register can't register type '{edgetype}' because it has already been registered"
            )

        edge_registry[edgetype] = cls

    def _repr_dot_(self) -> str:
        return f"node{self.src_id} -> node{self.dst_id}"

    @classmethod
    def get_cache(self) -> EdgeCache:
        """Gets the edge cache

        Returns:
            EdgeCache: the global edge cache
        """
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
        props = {}
        if hasattr(e, "properties"):
            props = e.properties
        return cls(
            src_id=e.start_id,
            dst_id=e.end_id,
            _id=id,
            type=e.type,
            **props,
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

        e._id = ret[0]["e_id"]
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

    @classmethod
    def connect(
        cls,
        src: Node | NodeId,
        dst: Node | NodeId,
        edgetype: str | None = None,
        db: GraphDB | None = None,
        **kwargs: Any,
    ) -> Self:
        """Connects two nodes using this Edge type

        Args:
            src (Node | NodeId): The Node to use as the starting point of the Edge
            dst (Node | NodeId): The Node to use as the ending point of the Edge
            edgetype (str | None, optional): The type of the edge. If this is
                being called from the base Edge class, the edgetype is looked up to
                determine what class to use to create the edge.
            db (GraphDB | None, optional): The GraphDB where the Edge will be
                created. Defaults to None, which indicates using the singleton
                GraphDB.
            **kwargs (Any): Any data to be stored on the Edge. Will be validated
                by the specific Pydantic model of the Edge subclass.

        Raises:
            Exception: raised if it can't determine what type of Edge to use

        Returns:
            Self: the newly created Edge
        """
        db = db or GraphDB.singleton()
        src_id = Node.to_id(src)
        dst_id = Node.to_id(dst)
        src_node = Node.get(src_id, db=db)
        dst_node = Node.get(dst_id, db=db)

        clstype: str | None = None
        # lookup class in based on specified type
        if cls is Edge and edgetype in edge_registry:
            cls = edge_registry[edgetype]  # type: ignore

        # get type from class model
        if cls is not Edge:
            clstype = _pydantic_get_default(cls, "type")

        # no class found, use edge type instead
        if clstype is None and edgetype is not None:
            clstype = edgetype

        # couldn't find any type
        if clstype is None:
            raise Exception("no Edge type provided")

        # check allowed_connections
        _check_schema(cls, clstype, src_node, dst_node, db)

        e = cls(src_id=src_id, dst_id=dst_id, type=clstype, **kwargs)
        src_node.src_edges.add(e)
        dst_node.dst_edges.add(e)

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
        # XXX: the _id field below shouldn't have been included in the
        # first place because Pythonic should exclude fields with underscores
        ret = e.model_dump(exclude={"_id", "allowed_connections"})

        if include_type and hasattr(e, "type"):
            ret["type"] = e.type
        return ret

    @staticmethod
    def to_id(e: Edge | EdgeId) -> EdgeId:
        """Convenience method to convert an Edge or EdgeID to an EdgeId

        Args:
            e (Edge | EdgeId): The Edge or EdgeId to be converted

        Returns:
            EdgeId: The resulting EdgeId
        """
        if isinstance(e, Edge):
            return e.id
        else:
            return e


EdgeCache = GraphCache[EdgeId, Edge]
edge_cache: EdgeCache | None = None
EdgeConnectionsList = Iterable[tuple[str, str]]
edge_registry: dict[str, type[Edge]] = {}


def _check_schema(
    edge_cls: type[Edge],
    clstype: str,
    src: Node,
    dst: Node,
    db: GraphDB,
) -> None:
    allowed_connections = _pydantic_get_default(edge_cls, "allowed_connections")
    src_name = src.__class__.__name__
    src_names = _get_node_parent_names(src.__class__)
    src_names.add(src_name)

    dst_name = dst.__class__.__name__
    dst_names = _get_node_parent_names(dst.__class__)
    dst_names.add(dst_name)

    # check if the src (or it's parents) are allowed to connect to dst (or it's parents)
    if allowed_connections is not None:
        found = False
        for conn in allowed_connections:
            if conn[0] in src_names and conn[1] in dst_names:
                found = True
                break

        if not found:
            raise Exception(
                f"attempting to connect edge '{clstype}' from '{src_name}' to '{dst_name}' not in allowed connections list"
            )
    # no allowed_connections set, which is a no-no for strict mode
    elif db.strict_schema:
        err_msg = f"allowed_connections missing in '{edge_cls.__name__}' and strict_schema is set"
        if db.strict_schema_warns:
            warnings.warn(err_msg, StrictSchemaWarning)
        else:
            raise Exception(err_msg)


#######
# EDGE LIST
#######
class EdgeFetchIterator(Iterable[Edge]):
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


class EdgeList(MutableSet[Edge | EdgeId], Sequence[Edge]):
    """A list of Edges that is used by Node for keeping track of the connections it has.
    Implements interfaces for both a MutableSet (i.e. set()) and a Sequence (i.e. read-only list())
    """

    def __init__(self, ids: Iterable[EdgeId]):
        self._edges: list[EdgeId] = list(ids)

    def __iter__(self) -> EdgeFetchIterator:
        return EdgeFetchIterator(self._edges)

    @overload
    def __getitem__(self, idx: int, /) -> Edge: ...
    @overload
    def __getitem__(self, idx: slice[Any, Any, Any], /) -> EdgeList: ...

    def __getitem__(self, idx: slice[Any, Any, Any] | int, /) -> Edge | EdgeList:
        if isinstance(idx, slice):
            return EdgeList(self._edges[idx])
        else:
            return Edge.get(self._edges[idx])

    def __len__(self) -> int:
        return len(self._edges)

    def __contains__(self, e: Any) -> bool:
        if isinstance(e, Edge) or isinstance(e, int):
            e_id = Edge.to_id(e)  # type: ignore
        else:
            return False

        return e_id in self._edges

    def __add__(self, l2: EdgeList) -> EdgeList:
        return EdgeList(self._edges + l2._edges)

    def __str__(self) -> str:
        ret = f"EdgeList({id(self)}):\n"

        for e in self:
            ret += f"\t{e}\n"

        return ret

    @property
    def ids(self) -> set[EdgeId]:
        return set(self._edges)

    def add(self, e: Edge | EdgeId) -> None:
        """Adds a new Edge to the list"""
        e_id = Edge.to_id(e)

        if e_id in self._edges:
            return

        self._edges.append(e_id)

    def discard(self, e: Edge | EdgeId) -> None:
        """Removes an edge from the list"""
        e_id = Edge.to_id(e)

        self._edges.remove(e_id)

    def replace(self, old: Edge | EdgeId, new: Edge | EdgeId) -> None:
        """Replaces all instances of an old Edge with a new Edge. Useful for when an Edge is
        persisted to the graph database and its permanent ID is assigned
        """
        old_id = Edge.to_id(old)
        new_id = Edge.to_id(new)
        for i in range(len(self._edges)):
            if self._edges[i] == old_id:
                self._edges[i] = new_id

    def select(
        self,
        *,
        filter_fn: EdgeFilterFn | None = None,
        type: str | None = None,
        id: EdgeId | None = None,
        db: GraphDB | None = None,
    ) -> EdgeList:
        """Returns a list of Edges that meet the specified criteria. If multiple
        criteria are specified, all of them are applied.

        Args:
            filter_fn (EdgeFilterFn | None, optional): A function applied to
                each Edge. If it returns True, the result is included in the return
                results. Defaults to None.
            type (str | None, optional): If the Edge type matches this type, it
                is included in the return results. Defaults to None.
            id (EdgeId | None, optional): If the EdgeId matches this id it is
                included in the results. Defaults to None.

        Returns:
            EdgeList: An EdgeList of the matching Edges.
        """
        self.db = db or GraphDB.singleton()
        edge_ids = self._edges
        if filter_fn is not None:
            # TODO: Edge.get_many() would be more efficient here if / when it
            # gets implemented
            edge_ids = [e for e in edge_ids if filter_fn(Edge.get(e))]

        if type is not None:
            if self.db.strict_schema and type not in edge_registry:
                raise Exception(f"Edge type '{type}' not a known Edge type")
            edge_ids = [e for e in edge_ids if Edge.get(e).type == type]

        if id is not None:
            edge_ids = [e for e in edge_ids if e == id]

        return EdgeList(edge_ids)


#######
# NODE
#######
class NodeNotFound(Exception):
    """An exception raised when trying to retreive a Node that doesn't exist."""


class NodeCreationFailed(Exception):
    """An exception raised when trying to create a Node in the graph database fails"""


def _get_next_new_node_id() -> NodeId:
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
        """The unique ID of the node"""
        return self._id

    @property
    def src_edges(self) -> EdgeList:
        """All Edges that originate at this Node"""
        return self._src_edges

    @property
    def dst_edges(self) -> EdgeList:
        """All Edges that terminate at this Node"""
        return self._dst_edges

    @property
    def edges(self) -> EdgeList:
        """All Edges attached to this Node, regardless of direction"""
        return self._src_edges + self._dst_edges

    @property
    def predecessors(self) -> NodeList:
        """All Nodes connected with an directed Edge that ends with this node.
        Also referred to as an 'in-neighbor'.
        """
        return NodeList([e.src.id for e in self.dst_edges])

    @property
    def successors(self) -> NodeList:
        """All Nodes connected with an directed Edge that starts with this node.
        Also referred to as an 'out-neighbor'.
        """
        return NodeList([e.dst.id for e in self.src_edges])

    @property
    def src_nodes(self) -> NodeList:
        """An alias for Node.predecessors"""
        return self.predecessors

    @property
    def dst_nodes(self) -> NodeList:
        """An alias for Node.successors"""
        return self.successors

    @property
    def neighbors(self) -> NodeList:
        """All adjacent nodes, regardless of edge direction"""
        return self.successors + self.predecessors

    @property
    def new(self) -> bool:
        """Whether or not this Node is new (not saved to the database yet)"""
        return self._new

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # set passed-in private values or their defaults
        self._db = kwargs["_db"] if "_db" in kwargs else GraphDB.singleton()
        self._id = kwargs["_id"] if "_id" in kwargs else _get_next_new_node_id()
        self._src_edges = kwargs["_src_edges"] if "_src_edges" in kwargs else EdgeList([])
        self._dst_edges = kwargs["_dst_edges"] if "_dst_edges" in kwargs else EdgeList([])

        if self.id < 0:
            self._new = True  # TODO: derived?
            Node.get_cache()[self.id] = self

        self._orig_labels = self.labels.copy()
        self._db.node_counter.add(1, attributes={"new": self._new, "labels": ":".join(self.labels)})

    def __del__(self) -> None:
        # logger.trace(f"Node.__del__: {self}")
        try:
            self.__class__.save(self, db=self._db)
        except Exception as e:
            err_msg = f"error saving during del: {e}"
            warnings.warn(err_msg, ErrorSavingDuringDelWarning)

    def __repr__(self) -> str:
        return f"Node({self.id})"

    def __str__(self) -> str:
        return f"Node({self.id}, labels={self.labels})"

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)
        clsname = cls.__name__

        if not hasattr(cls, "labels"):
            new_lbls = {c.__name__ for c in cls.__mro__ if issubclass(c, Node) and not c is Node}

            def default_subclass_fields() -> set[str]:
                return new_lbls

            cls.labels = Field(default_factory=default_subclass_fields, exclude=True)
            labels_key = frozenset(new_lbls)
        else:
            if isinstance(cls.labels, FieldInfo):
                labels_key = frozenset(cls.labels.get_default(call_default_factory=True))
            else:
                labels_key = frozenset(cls.labels)

        if clsname in node_registry:
            raise Exception(
                f"""node_register can't register '{clsname}' because that name has already been registered"""
            )

        if labels_key in node_label_registry:
            labels = ", ".join(sorted(list(labels_key)))
            raise Exception(
                f"""node_register can't register labels '{labels}' because they have already been registered"""
            )

        node_registry[clsname] = cls
        node_label_registry[labels_key] = cls

    def _repr_dot_(self, extra_style: str = "") -> str:
        # name
        name = f"<b>{self.__class__.__name__}({self.id})</b>"

        # props
        props: list[str] = []
        for f in _pydantic_get_fields(self.__class__):
            fi = _pydantic_get_field(self.__class__, f)
            prop_str = f"{f}: {_clean_annotation(fi.annotation)} = {str(getattr(self, f))}"

            # escape prop str
            # TODO: this could just be a static string
            pattern = r"[" + re.escape('<>{}|\\"') + r"]"
            prop_str = re.sub(pattern, r"\\\g<0>", prop_str)

            props.append(f'{prop_str}<br align="left"/>')
        props.sort()

        # create string
        extra_space = " " if len(extra_style) else ""
        return f"node{self.id} [label=<{{{name} | {''.join(props)}}}>{extra_space}{extra_style}]"

    def neighborhood(self, depth: int = 1) -> NodeList:
        if depth < 0:
            raise Exception("neighborhood depth must be greater than zero")
        elif depth == 0:
            return NodeList([self.id])
        elif depth == 1:
            neighborhood = self.neighbors
            neighborhood.add(self.id)
            return neighborhood
        else:
            ret: set[NodeId] = set()
            for node_id in self.neighbors._nodes:
                n = Node.get(node_id)
                node_ids = n.neighborhood(depth - 1).ids
                ret.update(node_ids)

            return NodeList(ret)

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

        logger.trace(f"Node load result: {res}")

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
        """Returns all the specified Nodes from the GraphDB. This allows for
        optimizing database queries and enhances performance.

        Args:
            node_set (set[NodeId]): The Set of NodeIDs to be returned.
            db (GraphDB | None, optional): The database to load the Nodes from.
                Defaults to None, meaning the singleton database will be used.
            load_edges (bool, optional): If True, the Edges related to each node
                (source or destination) will also be loaded into the EdgeCache.
                Defaults to False.

        Raises:
            NodeNotFound: If any Node in the node_set is not found, this
                exception will be raised.

        Returns:
            list[Self]: A list of all the Nodes that were retreived.
        """
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
                    props = {}
                    if hasattr(e, "properties"):
                        props = e.properties
                    new_edge = Edge(
                        src_id=e.start_id,
                        dst_id=e.end_id,
                        _id=e.id,
                        type=e.type,
                        **props,
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
                if cls is Node and cls_lbls in node_label_registry:
                    mkcls = cast(type[Self], node_label_registry[cls_lbls])
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
        """Calls Node.find and expects to return exactly one Node. Raises an
        exception of the list contains more than one Node or no Nodes. All
        arguments are the same as Node.find.

        Raises:
            Exception: Raised if there is more than one node in the results
            Exception: Raised if there are no nodes in the results

        Returns:
            Self | None: Returns None if the list is empty, or the node in the list.
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
        node_ids: Collection[NodeId],
        *,
        batch_size: int = 128,
        db: GraphDB | None = None,
        load_edges: bool = False,
        return_nodes: bool = False,
        progress_callback: ProgressFn | None = None,
    ) -> list[Node]:
        db = db or GraphDB.singleton()

        if not isinstance(node_ids, set):
            node_ids = set(node_ids)

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

            start = curr
            curr += batch_size

        assert len(ret_list) == len(node_ids)
        return ret_list

    @classmethod
    def get_cache(cls) -> NodeCache:
        """Returns the NodeCache

        Returns:
            NodeCache: The cache of all currently loaded Nodes.
        """
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
        type: str | None = None,
        *,
        db: GraphDB | None = None,
    ) -> Edge:
        """Connects two nodes (creates an Edge between two nodes)

        Args:
            src (NodeId | Node): The Node to use at the start of the connection
            dst (NodeId | Node): The Node to use at the end of the connection
            type (str): The type of the edge to use for the connection
            db (GraphDB | None): the graph database to use, or None to use the GraphDB singleton

        Returns:
            Edge: The Edge that was created
        """
        return Edge.connect(src, dst, type, db=db)

    @staticmethod
    def delete(n: Node, *, db: GraphDB | None = None) -> None:
        """Deletes the specified Node and its Edges from the underlying database
        and the NodeCache

        Args:
            n (Node): The Node to be deleted
            db (GraphDB | None, optional): The GraphDb to delete from. Defaults
                to None, which means the singleton database will be used.
        """
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
    def to_id(n: Node | NodeId) -> NodeId:
        """Convenience method to convert an Node or NodeId to an NodeId

        Args:
            n (Node | NodeId): The Node or NodeId to be converted

        Returns:
            NodeId: The resulting NodeId
        """
        if isinstance(n, Node):
            return n.id
        else:
            return n

    @staticmethod
    def walk(
        n: Node,
        *,
        mode: WalkMode = "both",
        edge_filter: EdgeFilterFn | None = None,
        node_filter: NodeFilterFn | None = None,
        node_callback: NodeCallbackFn | None = None,
        _walk_history: set[int] | None = None,
    ) -> None:
        """Performs a depth-first search (DFS) of the graph starting from the
        specified node.

        Args:
            n (Node): The Node to start the search from
            mode (WalkMode, optional): The type of walk to perform. Options
                include .... Defaults to "both".
            edge_filter (EdgeFilterFn | None, optional): A function to be called
                on each Edge. If it turns true, the Edge will be included in the
                walk, otherwise the Edge and attached Node will be skipped. Defaults
                to None which results in all Edges being included.
            node_filter (NodeFilterFn | None, optional): A function to be called
                on each node. If it turns true, the Node will be included in the
                walk, otherwise the Node and attached Edges will be skipped. Defaults
                to None which results in all Nodes being included.
            node_callback (NodeCallbackFn | None, optional): Called on each
                included Node. Defaults to None.
            _walk_history (set[int] | None, optional): For internal recurision
                use only.
        """
        # if we have walked this node before, just return
        _walk_history = _walk_history or set()
        if n.id in _walk_history:
            return
        _walk_history.add(n.id)

        edge_filter = edge_filter or cast(EdgeFilterFn, true_filter)
        node_filter = node_filter or true_filter
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
                        node_filter=node_filter,
                        node_callback=node_callback,
                        _walk_history=_walk_history,
                    )


#######
# Node List
#######
class NodeFetchIterator(Iterable[Node]):
    """The implementation of an iterator for an NodeList. Only intended to be used internally by
    NodeList.
    """

    def __init__(self, node_list: list[NodeId]):
        self._node_list = node_list
        self.cur = 0

    def __iter__(self) -> NodeFetchIterator:
        return self

    def __next__(self) -> Node:
        if self.cur >= len(self._node_list):
            raise StopIteration

        id = self._node_list[self.cur]
        self.cur = self.cur + 1
        return Node.get(id)


class NodeList(MutableSet[Node | NodeId], Sequence[Node]):
    """A list of Nodes. Implements interfaces for both a MutableSet (i.e. set())
    and a Sequence (i.e. read-only dict())
    """

    def __init__(self, ids: Iterable[NodeId]):
        self._nodes: list[NodeId] = list(ids)

    def __iter__(self) -> NodeFetchIterator:
        return NodeFetchIterator(self._nodes)

    @overload
    def __getitem__(self, idx: int, /) -> Node: ...
    @overload
    def __getitem__(self, idx: slice[Any, Any, Any], /) -> NodeList: ...

    def __getitem__(self, idx: slice[Any, Any, Any] | int, /) -> Node | NodeList:
        if isinstance(idx, slice):
            return NodeList(self._nodes[idx])
        else:
            return Node.get(self._nodes[idx])

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, n: Any) -> bool:
        if isinstance(n, Node) or isinstance(n, int):
            n_id = Node.to_id(n)  # type: ignore
        else:
            return False

        return n_id in self._nodes

    def __add__(self, l2: NodeList) -> NodeList:
        return NodeList(self._nodes + l2._nodes)

    def __str__(self) -> str:
        ret = f"NodeList({id(self)}):\n"

        for n in self:
            ret += f"\t{n}\n"

        return ret

    @property
    def ids(self) -> set[NodeId]:
        return set(self._nodes)

    @property
    def connections(self) -> EdgeList:
        edge_set: set[EdgeId] = set()
        for node_id in self._nodes:
            n = Node.get(node_id)
            for e in n.edges:
                if e.src_id in self._nodes and e.dst_id in self._nodes:
                    edge_set.add(e.id)

        return EdgeList(edge_set)

    def add(self, n: Node | NodeId) -> None:
        """Adds a new Node to the list"""
        n_id = Node.to_id(n)

        if n_id in self._nodes:
            return

        self._nodes.append(n_id)

    def discard(self, n: Node | NodeId) -> None:
        """Removes an Node from the list"""
        n_id = Node.to_id(n)

        self._nodes.remove(n_id)

    def select(
        self,
        *,
        filter_fn: NodeFilterFn | None = None,
        labels: set[str] | str | None = None,
        partial_labels: set[str] | None = None,
    ) -> NodeList:
        """Returns a list of Nodes that meet the specified criteria. If multiple
        criteria are specified, all of them are applied.

        Args:
            filter_fn (NodeFilterFn | None, optional): A function applied to
                each Node. If it returns True, the result is included in the return
                results. Defaults to None.
            labels (set[str] | str | None, optional): If the Node labels exactly matches
                this set, it is included in the return results. Defaults to None.
            partial_labels (set[str] | None): If the Node labels contain all the
                labels in partial_labels (and potentially additional lebels) it is
                included in the results. Defaults to None
            id (NodeId | None, optional): If the NodeId matches this id it is
                included in the results. Defaults to None.

        Returns:
            NodeList: An NodeList of the matching Nodes.
        """
        node_ids = self._nodes
        if filter_fn is not None:
            Node.get_many(node_ids)
            node_ids = [n for n in node_ids if filter_fn(Node.get(n))]

        if labels is not None:
            labels = set(labels) if isinstance(labels, str) else labels
            node_ids = [n for n in node_ids if Node.get(n).labels == labels]

        if partial_labels is not None:
            node_ids = [n for n in node_ids if Node.get(n).labels.issuperset(partial_labels)]

        return NodeList(node_ids)

    def to_dot(self, extra_styles: dict[int, str] = dict()) -> str:
        """Converts the NodeList and all the Edges between Nodes in the list
        into a string representing Graphviz DOT diagram.
        """
        ret = dot_graph_header

        nodes = Node.get_many(self._nodes, load_edges=True)
        for n in nodes:
            style = extra_styles[n.id] if n.id in extra_styles else ""
            ret += f"\n    // Node {n.id}\n"
            ret += f"    {n._repr_dot_(style)}\n"

        edges = self.connections
        for e in edges:
            ret += f"\n    // Edge {e.id}\n"
            ret += f"    {e._repr_dot_()}\n"

        ret += "}"

        return ret


node_registry: dict[str, type[Node]] = {}
node_label_registry: dict[frozenset[str], type[Node]] = {}


WalkMode = Literal["src", "dst", "both"]
NodeFilterFn = Callable[[Node], bool]
EdgeFilterFn = Callable[[Edge], TypeGuard[Edge]]
ProgressFn = Callable[[list[Node]], None]
NodeCallbackFn = Callable[[Node], None]
EdgeCallbackFn = Callable[[Edge], None]

NodeCache = GraphCache[NodeId, Node]
node_cache: NodeCache | None = None


class SchemaValidationError(Exception):
    """An error raised when the GraphDB schema isn't valid"""

    def __init__(self, errors: list[str]) -> None:
        err_str = ""
        self.errors = errors

        for errno in range(len(errors)):
            err = errors[errno]
            err_str += f"\t{errno}: {err}\n"

        super().__init__(f"Error validating schema:\n{err_str}")


dot_graph_header = """digraph {
    graph [
        fontname="Arial"
        labelloc="t"
    ]

    node [
        fontname="Arial"
        shape=record
        style=filled
        fillcolor=gray95
    ]

    edge [
        fontname="Arial"
        style=""
    ]
    """


class Schema:
    """The automatically generated GraphDB schema."""

    def __init__(self, skip_validation: bool = False) -> None:
        if not skip_validation:
            self.validate()

        # edges
        self.edge_names = set(edge_registry.keys())
        self.edges = [_EdgeDescription(edge_cls) for edge_cls in edge_registry.values()]
        self.edges.sort(key=lambda e: e.name)

        # nodes
        self.node_names = set(node_registry.keys())
        self.nodes = [_NodeDescription(node_cls) for node_cls in node_registry.values()]
        self.nodes.sort(key=lambda n: n.name)

    @classmethod
    def validate(cls) -> None:
        """Ensures that the GraphDB schema is valid by checking that all the
        Edge relationships refer to Nodes types that exist.

        Raises:
            SchemaValidationError: If validation fails this error will be raised.
        """
        errors: list[str] = []
        for edge_name, edge_cls in edge_registry.items():
            allowed_connections = _pydantic_get_default(edge_cls, "allowed_connections")

            if allowed_connections is None:
                continue

            for src, dst in allowed_connections:
                if src not in node_registry:
                    errors.append(
                        f"Edge '{edge_name}' requires src Node '{src}', which is not registered"
                    )

                if dst not in node_registry:
                    errors.append(
                        f"Edge '{edge_name}' requires dst Node '{dst}', which is not registered"
                    )

        if len(errors) > 0:
            raise SchemaValidationError(errors)

    def to_mermaid(self) -> str:
        """Converts a schema to a Mermaid diagram

        Returns:
            str: The text of the Mermaid diagram. Can be passed to a Mermaid
            interpreter to convert to a digram.
        """
        ret = "classDiagram\n"

        # nodes
        for n in self.nodes:
            ret += n.to_mermaid()

        # edges
        for e in self.edges:
            ret += e.to_mermaid()

        return ret

    def to_dot(self) -> str:
        """Converts a schema to a Graphviz DOT diagram

        Returns:
            str: The text of the Graphviz DOT diagram. Can be passed to
            the Graphviz program to convert to an image.
        """
        ret = dot_graph_header

        # nodes
        for n in self.nodes:
            ret += n.to_dot()

        # edges
        for e in self.edges:
            ret += e.to_dot()

        ret += "}"
        return ret

    @classmethod
    def _repr_markdown_(cls) -> str:
        return f"``` mermaid\n{Schema().to_mermaid()}\n```\n"


def _pydantic_get_fields(m: type[BaseModel]) -> set[str]:
    return set(m.model_fields.keys())


def _pydantic_get_field(m: type[BaseModel], f: str) -> FieldInfo:
    return m.model_fields[f]


def _pydantic_get_default(m: type[BaseModel], f: str) -> Any:
    return m.model_fields[f].get_default(call_default_factory=True)


def _is_local(c: type[object], attr: str) -> bool:
    local_to_parent = [_is_local(p, attr) for p in c.__mro__ if p is not c]
    if any(local_to_parent):
        return False

    if attr in c.__dict__:
        return True

    if hasattr(c, "__fields__") and attr in c.__fields__:
        return True

    if hasattr(c, "__wrapped__"):
        return _is_local(c.__wrapped__, attr)

    return False


def _get_node_parent_names(model: type[BaseModel]) -> set[str]:
    ret = {c.__name__ for c in model.__mro__ if Node in c.__mro__}
    if model.__name__ in ret:
        ret.remove(model.__name__)
    if "Node" in ret:
        ret.remove("Node")

    return ret


def _clean_annotation(annotation: Any) -> str:
    import typing

    if isinstance(annotation, str):
        return annotation
    elif annotation is None:
        return "None"
    elif isinstance(annotation, typing._GenericAlias):  # type: ignore
        # Handle generics like List, Dict, etc.
        origin = annotation.__origin__
        args = [_clean_annotation(arg) for arg in annotation.__args__]
        return f"{origin.__name__}[{', '.join(args)}]"
    elif isinstance(annotation, _SpecialForm):
        # Handle special forms like Any, Union, etc.
        return annotation._name  # type: ignore
    else:
        return annotation.__name__  # type: ignore


@functools.cache
def _get_methods(c: type[object]) -> set[str]:
    return {name for name, member in inspect.getmembers(c) if inspect.isfunction(member)}


class _FieldDescription:
    def __init__(self, model: type[BaseModel], fieldname: str) -> None:
        self.model = model
        self.field_info = _pydantic_get_field(model, fieldname)
        self.name = fieldname
        self.default_val = self.field_info.get_default(call_default_factory=True)
        self.type = _clean_annotation(self.field_info.annotation)
        self.exclude = self.field_info.exclude

    def __str__(self) -> str:
        return f"Field({self.name}: {self.type} = {self.default_val})"

    @property
    def default_val_str(self) -> str:
        """Control over a reliable and reproducable default value for printing
        the schema.
        """
        if isinstance(self.default_val, set):
            return str(sorted(self.default_val))

        return str(self.default_val)


@dataclass
class ParamData:
    type: str
    name: str
    default: Any

    @property
    def formatted_default(self) -> str:
        return f" = {self.default}" if self.default is not inspect._empty else ""


class _MethodDescription:
    def __init__(self, model: type[BaseModel], name: str) -> None:
        self.model = model
        self.name = name
        self.signature = inspect.signature(getattr(model, name))
        self.return_type = _clean_annotation(self.signature.return_annotation)
        self.raw_params = self.signature.parameters

    @property
    def params(self) -> list[ParamData]:
        ret: list[ParamData] = []

        for param_name, param_type in self.raw_params.items():
            if param_name == "self":
                continue

            t = (
                _clean_annotation(param_type.annotation)
                if param_type.annotation is not inspect._empty
                else ""
            )
            ret.append(ParamData(type=t, name=param_name, default=param_type.default))

        return ret


class _ModelDescription:
    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model

        # fields
        self.fields = [
            _FieldDescription(model, fieldname) for fieldname in _pydantic_get_fields(model)
        ]
        self.fields.sort(key=lambda f: f.name)

        # parents
        self.parent_class_names = _get_node_parent_names(model)
        self.parents = [
            _NodeDescription(node_registry[node_name]) for node_name in self.parent_class_names
        ]
        self.parents.sort(key=lambda p: p.name)

        # methods
        self.method_names = (
            _get_methods(model)
            - _get_methods(object)
            - _get_methods(BaseModel)
            - _get_methods(Node)
        )
        self.methods = [_MethodDescription(model, name) for name in self.method_names]
        self.methods.sort(key=lambda m: m.name)


class _NodeDescription(_ModelDescription):
    def __init__(self, node_cls: type[Node]) -> None:
        super().__init__(node_cls)

        self.name = node_cls.__name__

    def __str__(self) -> str:
        return f"NodeDesc({self.name})"

    def to_mermaid(self, indent: int = 4) -> str:
        ret = f"""\n{" ":>{indent}}%% Node: {self.name}\n"""

        # add fields
        for field in self.fields:
            sym = "+" if _is_local(self.model, field.name) else "^"
            default_val = (
                f" = {field.default_val_str}" if field.default_val is not PydanticUndefined else ""
            )
            ret += f"""{" ":>{indent}}{self.name}: {sym}{field.type} {field.name}{default_val}\n"""

        # add methods
        for method in self.methods:
            sym = "+" if _is_local(self.model, method.name) else "^"
            param_list = [f"{p.type} {p.name}{p.formatted_default}" for p in method.params]
            params = ", ".join(param_list)
            ret += f"""{" ":>{indent}}{self.name}: {sym}{method.name}({params}) {method.return_type}\n"""

        # add links to inherited nodes
        for parent in self.parent_class_names:
            ret += f"""{" ":>{indent}}{self.name} ..|> {parent}: inherits\n"""

        return ret

    def to_dot(self, indent: int = 4) -> str:
        ret = f"""\n{" ":>{indent}}// Node: {self.name}\n"""

        # create fields
        fields_str = ""
        for field in self.fields:
            sym = "+" if _is_local(self.model, field.name) else "^"
            default_val = (
                f" = {field.default_val_str}" if field.default_val is not PydanticUndefined else ""
            )
            fields_str += f"""{sym}{field.name}: {field.type}{default_val}<br align="left"/>"""

        # create methods
        methods_str = ""
        for method in self.methods:
            sym = "+" if _is_local(self.model, method.name) else "^"
            param_list = [f"{p.name}: {p.type}{p.formatted_default}" for p in method.params]
            params = ", ".join(param_list)
            methods_str += (
                f"""{sym}{method.name}({params}): {method.return_type}<br align="left"/>"""
            )

        # add node
        ret += f"""{" ":>{indent}}{self.name} [label=<{{ <b>{self.name}</b> | {fields_str} | {methods_str} }}>]\n"""

        # add links to inherited nodes
        for parent in self.parent_class_names:
            ret += (
                f"""{" ":>{indent}}{self.name} -> {parent} [label="inherits" arrowhead=empty]\n"""
            )

        return ret


class _EdgeDescription(_ModelDescription):
    def __init__(self, edge_cls: type[Edge]) -> None:
        super().__init__(edge_cls)

        self.edge_cls = edge_cls
        self.name = edge_cls.__name__
        self.edgetype = _pydantic_get_default(edge_cls, "type")

        # allowed connections
        self.allowed_connections = cast(
            EdgeConnectionsList, _pydantic_get_default(edge_cls, "allowed_connections")
        )
        assert self.allowed_connections is not None

        # related nodes
        self.related_nodes: set[str] = set()
        for conn in self.allowed_connections:
            self.related_nodes.add(conn[0])
            self.related_nodes.add(conn[1])

    def __str__(self) -> str:
        return f"EdgeDesc({self.name})"

    @property
    def resolved_name(self) -> str:
        if self.edgetype == self.name:
            return self.name

        return f"{self.edgetype} ({self.name})"

    def to_mermaid(self, indent: int = 4) -> str:
        ret = f"""\n{" ":>{indent}}%% Edge: {self.resolved_name}\n"""

        # add connections
        for conn in self.allowed_connections:
            ret += f"""{" ":>{indent}}{conn[0]} --> {conn[1]}: {self.resolved_name}\n"""

        return ret

    def to_dot(self, indent: int = 4) -> str:
        ret = f"""\n{" ":>{indent}}// Edge: {self.resolved_name}\n"""

        # add connections
        for conn in self.allowed_connections:
            ret += f"""{" ":>{indent}}{conn[0]} -> {conn[1]} [label="{self.resolved_name}" arrowhead=vee]\n"""

        return ret
