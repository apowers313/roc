"""FastAPI router for graph visualization API endpoints.

Serves graph data from either the live in-memory GraphCache (for active
game runs) or historical graph.json archives (for past runs).
All endpoints are mounted under ``/api/runs/{run_name}/graph/``.
"""

from __future__ import annotations

from typing import Annotated, Any

import networkx as nx
from fastapi import APIRouter, HTTPException, Query

from roc.reporting.graph_service import GraphService

graph_router = APIRouter(prefix="/api/runs/{run_name}/graph", tags=["graph"])


def _get_graph_service() -> GraphService:
    """Get the GraphService from the DataStore singleton.

    Raises:
        HTTPException: 503 if dashboard is not initialized.
    """
    from roc.reporting.api_server import _data_store

    if _data_store is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return _data_store.get_graph_service()


def _is_run_live(run_name: str) -> bool:
    """Check if the run is the currently active live session."""
    from roc.reporting.api_server import _data_store

    if _data_store is None:
        return False
    return _data_store.is_live(run_name)


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
    """Get the subgraph around a specific frame tick."""
    try:
        if _is_run_live(run_name):
            sub = GraphService.subgraph_from_frame_live(tick, depth=depth)
        else:
            svc = _get_graph_service()
            sub = svc.subgraph_from_frame(run_name, tick, depth=depth)
    except FileNotFoundError as e:
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
    """Get the subgraph around a specific node."""
    try:
        if _is_run_live(run_name):
            sub = GraphService.subgraph_from_node_live(node_id, depth=depth)
        else:
            svc = _get_graph_service()
            sub = svc.subgraph_from_node(run_name, node_id, depth=depth)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _format_response(sub, node_id, format)


@graph_router.get("/object/{uuid}")
def get_object_history(
    run_name: str,
    uuid: int,
    format: Annotated[str, Query()] = "cytoscape",
) -> dict[str, Any]:
    """Get the full history graph for an object by UUID."""
    try:
        if _is_run_live(run_name):
            sub = GraphService.object_history_live(uuid)
        else:
            svc = _get_graph_service()
            sub = svc.object_history(run_name, uuid)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _format_response(sub, None, format)
