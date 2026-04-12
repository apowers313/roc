"""FastAPI router for graph visualization API endpoints.

Serves graph data from on-disk ``graph.json`` archives, falling back to the
in-memory ``GraphCache`` for runs whose archive has not yet been written
(active games and recently-stopped games where the cache is still warm).
All endpoints are mounted under ``/api/runs/{run_name}/graph/``.
"""

from __future__ import annotations

from typing import Annotated, Any

import networkx as nx
from fastapi import APIRouter, HTTPException, Query

from roc.reporting.graph_service import GraphService

graph_router = APIRouter(prefix="/api/runs/{run_name}/graph", tags=["graph"])


def _get_graph_service() -> GraphService:
    """Get a GraphService bound to the registry's data directory.

    Raises:
        HTTPException: 503 if the dashboard registry is not initialized.
    """
    from roc.reporting.api_server import _run_registry

    if _run_registry is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return GraphService(_run_registry.data_dir)


def _format_response(G: nx.DiGraph, root_id: int | None, fmt: str) -> dict[str, Any]:
    """Convert a subgraph to the requested output format."""
    if fmt == "node-link":
        result: dict[str, Any] = nx.node_link_data(G, edges="links")
        return result
    return GraphService.to_cytoscape(G, root_id=root_id)


@graph_router.get("/frame/{tick}")
def get_frame_graph(
    run_name: str,
    tick: int,
    depth: Annotated[int, Query(ge=1, le=5)] = 2,
    game: Annotated[int | None, Query()] = None,
    format: Annotated[str, Query()] = "cytoscape",
) -> dict[str, Any]:
    """Get the subgraph around a specific frame tick.

    Tries the on-disk ``graph.json`` first; falls back to the live
    ``GraphCache`` if the archive does not yet exist (active or
    recently-stopped runs where the writer has not flushed).
    """
    svc = _get_graph_service()
    try:
        sub = svc.subgraph_from_frame(run_name, tick, depth=depth)
    except FileNotFoundError:
        try:
            sub = GraphService.subgraph_from_frame_live(tick, depth=depth)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    root_id = GraphService._find_frame(sub, tick) if format != "node-link" else None
    return _format_response(sub, root_id, format)


@graph_router.get("/node/{node_id}")
def get_node_graph(
    run_name: str,
    node_id: int,
    depth: Annotated[int, Query(ge=1, le=5)] = 2,
    format: Annotated[str, Query()] = "cytoscape",
) -> dict[str, Any]:
    """Get the subgraph around a specific node.

    Tries the on-disk ``graph.json`` first; falls back to the live
    ``GraphCache`` if the archive does not yet exist.
    """
    svc = _get_graph_service()
    try:
        sub = svc.subgraph_from_node(run_name, node_id, depth=depth)
    except FileNotFoundError:
        try:
            sub = GraphService.subgraph_from_node_live(node_id, depth=depth)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _format_response(sub, node_id, format)


@graph_router.get("/object/{uuid}")
def get_object_history(
    run_name: str,
    uuid: str,
    format: Annotated[str, Query()] = "cytoscape",
) -> dict[str, Any]:
    """Get the full history graph for an object by UUID.

    The uuid path param is a string because Object UUIDs are 63-bit ints
    that overflow JS Number precision -- a numeric path param would force
    JS clients to lose precision before the request even goes out.

    Tries the on-disk ``graph.json`` first; falls back to the live
    ``GraphCache`` if the archive does not yet exist.
    """
    try:
        uuid_int = int(uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid uuid: {uuid!r}")
    svc = _get_graph_service()
    try:
        sub = svc.object_history(run_name, uuid_int)
    except FileNotFoundError:
        try:
            sub = GraphService.object_history_live(uuid_int)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _format_response(sub, None, format)
