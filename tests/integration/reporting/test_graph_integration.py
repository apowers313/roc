# mypy: disable-error-code="no-untyped-def"

"""Integration tests for the graph visualization pipeline.

Tests the full round-trip: graph archive creation, loading via GraphService,
API endpoint serving, and live snapshot routing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app
from roc.reporting.graph_service import GraphService


def _build_large_graph(node_count: int = 1200) -> nx.DiGraph:
    """Build a large graph for performance testing.

    Creates a chain of Frame nodes, each with a few children
    (TakeAction, ObjectInstance, Object), producing a graph
    with approximately ``node_count`` nodes.
    """
    G = nx.DiGraph()
    node_id = 1
    frames: list[int] = []

    while node_id < node_count:
        frame_id = node_id
        G.add_node(frame_id, labels="Frame", tick=len(frames) + 1)
        node_id += 1

        # Link to previous frame
        if frames:
            G.add_edge(frames[-1], frame_id, type="NextFrame")

        frames.append(frame_id)

        # Add TakeAction child
        if node_id < node_count:
            action_id = node_id
            G.add_node(action_id, labels="TakeAction", action_id=19)
            G.add_edge(frame_id, action_id, type="FrameAttribute")
            node_id += 1

        # Add ObjectInstance + Object children
        if node_id + 1 < node_count:
            oi_id = node_id
            obj_id = node_id + 1
            G.add_node(oi_id, labels="ObjectInstance", x=1, y=1)
            G.add_node(obj_id, labels="Object", uuid=obj_id, resolve_count=1)
            G.add_edge(frame_id, oi_id, type="SituatedObjectInstance")
            G.add_edge(oi_id, obj_id, type="ObservedAs")
            node_id += 2

    return G


def _write_graph_archive(run_dir: Path, G: nx.DiGraph) -> Path:
    """Write a graph.json archive file from a NetworkX DiGraph."""
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_path = run_dir / "graph.json"
    data = nx.node_link_data(G, edges="links")
    with open(graph_path, "w") as f:
        json.dump(data, f)
    return graph_path


class TestGraphArchiveRoundTrip:
    """Test the full export-load-extract cycle."""

    def test_graph_archive_round_trip(self, tmp_path: Path):
        """Export graph, load via GraphService, extract subgraph, verify structure."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=1)
        G.add_node(2, labels="TakeAction", action_id=19)
        G.add_node(3, labels="ObjectInstance", x=5, y=10)
        G.add_node(4, labels="Object", uuid=42, resolve_count=1)
        G.add_edge(1, 2, type="FrameAttribute")
        G.add_edge(1, 3, type="SituatedObjectInstance")
        G.add_edge(3, 4, type="ObservedAs")

        _write_graph_archive(tmp_path / "test-run", G)

        svc = GraphService(tmp_path)
        loaded = svc.get_graph("test-run")
        assert len(loaded.nodes) == 4
        assert len(loaded.edges) == 3

        sub = svc.subgraph_from_frame("test-run", tick=1, depth=2)
        assert 1 in sub.nodes  # Frame
        assert 2 in sub.nodes  # TakeAction
        assert 3 in sub.nodes  # ObjectInstance
        assert 4 in sub.nodes  # Object (depth 2 via ObjInst)

        cyto = GraphService.to_cytoscape(sub, root_id=1)
        assert cyto["meta"]["node_count"] == 4
        assert cyto["meta"]["edge_count"] == 3
        assert all(isinstance(n["data"]["id"], str) for n in cyto["elements"]["nodes"])


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_store(tmp_path: Path) -> Generator[None, None, None]:
    """Point the API at a RunRegistry backed by a temp directory."""
    import roc.reporting.api_server as mod

    orig_registry = mod._run_registry
    orig_reader = mod._run_reader
    mod.init_data_dir(tmp_path)
    mod._run_reader = None
    yield
    mod._run_registry = orig_registry
    mod._run_reader = orig_reader


class TestGraphApiServesArchivedData:
    """Full stack: write graph.json, hit API endpoint, verify Cytoscape JSON."""

    def test_frame_endpoint_returns_cytoscape_json(self, client: TestClient, tmp_path: Path):
        """API serves Cytoscape JSON from a graph.json archive."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=1)
        G.add_node(2, labels="TakeAction", action_id=19)
        G.add_edge(1, 2, type="FrameAttribute")

        _write_graph_archive(tmp_path / "test-run", G)

        resp = client.get("/api/runs/test-run/graph/frame/1?depth=2")
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert "meta" in data
        assert data["meta"]["node_count"] == 2

    def test_node_endpoint_returns_subgraph(self, client: TestClient, tmp_path: Path):
        """API serves node neighborhood from a graph.json archive."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=1)
        G.add_node(2, labels="TakeAction", action_id=19)
        G.add_edge(1, 2, type="FrameAttribute")

        _write_graph_archive(tmp_path / "test-run", G)

        resp = client.get("/api/runs/test-run/graph/node/1?depth=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["node_count"] == 2

    def test_404_for_missing_archive(self, client: TestClient):
        """API returns 404 when graph.json does not exist."""
        resp = client.get("/api/runs/nonexistent/graph/frame/1")
        assert resp.status_code == 404


class TestLargeGraphSubgraphExtraction:
    """Performance: subgraph extraction on large graphs completes quickly."""

    def test_large_graph_extraction_under_100ms(self, tmp_path: Path):
        """Graph with 1000+ nodes: subgraph extraction completes in <100ms."""
        G = _build_large_graph(1200)
        _write_graph_archive(tmp_path / "large-run", G)

        svc = GraphService(tmp_path)

        start = time.perf_counter()
        sub = svc.subgraph_from_frame("large-run", tick=1, depth=2)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Subgraph extraction took {elapsed_ms:.1f}ms"
        assert len(sub.nodes) > 0

    def test_max_nodes_limits_large_graph(self, tmp_path: Path):
        """max_nodes parameter prevents unbounded subgraph growth."""
        G = _build_large_graph(1200)
        _write_graph_archive(tmp_path / "large-run", G)

        svc = GraphService(tmp_path)
        sub = svc.subgraph_from_frame("large-run", tick=1, depth=5, max_nodes=50)
        assert len(sub.nodes) <= 50
