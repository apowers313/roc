# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/graph_api.py -- Graph Data API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app
from roc.reporting.data_store import DataStore


def _make_subgraph(node_count: int = 3) -> nx.DiGraph:
    """Build a small DiGraph with the given number of nodes."""
    G = nx.DiGraph()
    for i in range(1, node_count + 1):
        G.add_node(i, labels="Frame" if i == 1 else "Node", tick=1 if i == 1 else None)
    for i in range(2, node_count + 1):
        G.add_edge(1, i, type="Connected")
    return G


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_store(tmp_path: Path) -> Generator[None, None, None]:
    """Point the API at a DataStore backed by a temp directory."""
    import roc.reporting.api_server as mod

    orig = mod._data_store
    mod._data_store = DataStore(tmp_path)
    yield
    mod._data_store = orig


@pytest.fixture()
def mock_graph_service():
    """Create a mock GraphService and patch it into the graph_api module."""
    mock_svc = MagicMock()
    with patch("roc.reporting.graph_api._get_graph_service", return_value=mock_svc):
        yield mock_svc


class TestGetFrameGraph:
    def test_returns_cytoscape_json(self, client: TestClient, mock_graph_service):
        """GET /graph/frame/1 returns Cytoscape JSON with elements and meta."""
        sub = _make_subgraph(3)
        mock_graph_service.subgraph_from_frame.return_value = sub

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get("/api/runs/test-run/graph/frame/1?depth=2")
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert "nodes" in data["elements"]
        assert "edges" in data["elements"]
        assert "meta" in data

    def test_respects_depth_param(self, client: TestClient, mock_graph_service):
        """depth=1 vs depth=3 passes different depth to GraphService."""
        small = _make_subgraph(2)
        large = _make_subgraph(5)

        call_depths = []

        def track_depth(run_name, tick, depth=2):
            call_depths.append(depth)
            return small if depth == 1 else large

        mock_graph_service.subgraph_from_frame.side_effect = track_depth

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        client.get("/api/runs/test-run/graph/frame/1?depth=1")
        client.get("/api/runs/test-run/graph/frame/1?depth=3")
        assert call_depths == [1, 3]

    def test_node_link_format(self, client: TestClient, mock_graph_service):
        """format=node-link returns node-link JSON instead of Cytoscape."""
        sub = _make_subgraph(3)
        mock_graph_service.subgraph_from_frame.return_value = sub

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get("/api/runs/test-run/graph/frame/1?format=node-link")
        assert resp.status_code == 200
        data = resp.json()
        assert "directed" in data
        assert "nodes" in data
        assert "links" in data

    def test_404_missing_tick(self, client: TestClient, mock_graph_service):
        """404 when requested tick has no Frame node."""
        mock_graph_service.subgraph_from_frame.side_effect = ValueError(
            "No Frame node with tick=99"
        )
        resp = client.get("/api/runs/test-run/graph/frame/99")
        assert resp.status_code == 404

    def test_404_no_archive(self, client: TestClient, mock_graph_service):
        """404 when graph.json does not exist for the run."""
        mock_graph_service.subgraph_from_frame.side_effect = FileNotFoundError(
            "No graph archive found"
        )
        resp = client.get("/api/runs/test-run/graph/frame/1")
        assert resp.status_code == 404

    def test_depth_validation_rejects_zero(self, client: TestClient, mock_graph_service):
        """depth=0 returns 422."""
        resp = client.get("/api/runs/test-run/graph/frame/1?depth=0")
        assert resp.status_code == 422

    def test_depth_validation_rejects_too_large(self, client: TestClient, mock_graph_service):
        """depth=6 returns 422."""
        resp = client.get("/api/runs/test-run/graph/frame/1?depth=6")
        assert resp.status_code == 422


class TestGetNodeGraph:
    def test_returns_subgraph(self, client: TestClient, mock_graph_service):
        """GET /graph/node/123 returns neighborhood of that node."""
        sub = _make_subgraph(3)
        mock_graph_service.subgraph_from_node.return_value = sub

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get("/api/runs/test-run/graph/node/123?depth=2")
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert "meta" in data

    def test_404_missing_node(self, client: TestClient, mock_graph_service):
        """404 when node ID does not exist."""
        mock_graph_service.subgraph_from_node.side_effect = KeyError("node not found")
        resp = client.get("/api/runs/test-run/graph/node/999")
        assert resp.status_code == 404


class TestGetObjectHistory:
    def test_returns_all_frames(self, client: TestClient, mock_graph_service):
        """GET /graph/object/456 includes frames, instances, transforms."""
        G = nx.DiGraph()
        G.add_node(5, labels="Object", uuid=456)
        G.add_node(4, labels="ObjectInstance", x=5, y=10)
        G.add_node(1, labels="Frame", tick=1)
        G.add_edge(4, 5, type="ObservedAs")
        G.add_edge(1, 4, type="SituatedObjectInstance")

        mock_graph_service.object_history.return_value = G

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get("/api/runs/test-run/graph/object/456")
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert data["meta"]["node_count"] == 3

    def test_404_missing_object(self, client: TestClient, mock_graph_service):
        """404 when object UUID does not exist."""
        mock_graph_service.object_history.side_effect = ValueError("No Object node with uuid=999")
        resp = client.get("/api/runs/test-run/graph/object/999")
        assert resp.status_code == 404


class TestDepthValidation:
    def test_depth_min_is_1(self, client: TestClient, mock_graph_service):
        """Depth must be >= 1."""
        resp = client.get("/api/runs/test-run/graph/frame/1?depth=0")
        assert resp.status_code == 422

    def test_depth_max_is_5(self, client: TestClient, mock_graph_service):
        """Depth must be <= 5."""
        resp = client.get("/api/runs/test-run/graph/frame/1?depth=6")
        assert resp.status_code == 422

    def test_depth_default_is_2(self, client: TestClient, mock_graph_service):
        """Default depth is 2 when not specified."""
        sub = _make_subgraph(3)
        mock_graph_service.subgraph_from_frame.return_value = sub

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        client.get("/api/runs/test-run/graph/frame/1")
        mock_graph_service.subgraph_from_frame.assert_called_once_with("test-run", 1, depth=2)


# ---------------------------------------------------------------------------
# Live routing tests -- endpoints route to live cache for active runs
# ---------------------------------------------------------------------------


def _wire_edge(src: Any, dst: Any, etype: str) -> Any:
    """Create an Edge and wire it into the source and destination node edge lists."""
    from roc.db.graphdb import Edge

    e = Edge(type=etype, src_id=src.id, dst_id=dst.id)
    e._no_save = True
    src._src_edges.add(e)
    dst._dst_edges.add(e)
    return e


def _build_live_api_graph() -> dict[str, Any]:
    """Build minimal live graph for API routing tests."""
    from roc.db.graphdb import Node

    frame = Node(labels={"Frame"}, tick=1)
    action = Node(labels={"TakeAction"}, action_id=19)
    obj_inst = Node(labels={"ObjectInstance"}, x=5, y=10)
    obj = Node(labels={"Object"}, uuid=42, resolve_count=1)

    _wire_edge(frame, action, "FrameAttribute")
    _wire_edge(frame, obj_inst, "SituatedObjectInstance")
    _wire_edge(obj_inst, obj, "ObservedAs")

    return {"frame": frame, "action": action, "obj_inst": obj_inst, "obj": obj}


@pytest.fixture()
def live_data_store(tmp_path: Path) -> Generator[DataStore, None, None]:
    """Set up a DataStore with a live session named 'live-run'."""
    import roc.reporting.api_server as mod
    from roc.reporting.step_buffer import StepBuffer

    orig = mod._data_store
    ds = DataStore(tmp_path)
    buf = StepBuffer(capacity=100)
    ds.set_live_session("live-run", buf)
    mod._data_store = ds
    yield ds
    ds.clear_live_session()
    mod._data_store = orig


class TestLiveRouting:
    def test_frame_graph_routes_live(self, client: TestClient, live_data_store):
        """Live run uses live cache instead of graph.json archive."""
        g = _build_live_api_graph()
        resp = client.get("/api/runs/live-run/graph/frame/1?depth=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert data["meta"]["node_count"] >= 3  # frame + action + obj_inst

    def test_frame_graph_routes_historical(self, client: TestClient, mock_graph_service):
        """Non-live run falls back to graph.json archive via GraphService."""
        sub = _make_subgraph(3)
        mock_graph_service.subgraph_from_frame.return_value = sub

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get("/api/runs/test-run/graph/frame/1?depth=2")
        assert resp.status_code == 200
        mock_graph_service.subgraph_from_frame.assert_called_once()

    def test_node_graph_routes_live(self, client: TestClient, live_data_store):
        """Live run node endpoint uses live cache."""
        g = _build_live_api_graph()
        resp = client.get(f"/api/runs/live-run/graph/node/{g['frame'].id}?depth=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["node_count"] >= 2

    def test_object_history_routes_live(self, client: TestClient, live_data_store):
        """Live run object history uses live cache."""
        _build_live_api_graph()
        resp = client.get("/api/runs/live-run/graph/object/42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["node_count"] >= 3  # obj + obj_inst + frame

    def test_live_frame_404_missing_tick(self, client: TestClient, live_data_store):
        """Live run returns 404 for missing tick."""
        resp = client.get("/api/runs/live-run/graph/frame/999")
        assert resp.status_code == 404

    def test_live_node_404_missing(self, client: TestClient, live_data_store):
        """Live run returns 404 for missing node."""
        resp = client.get("/api/runs/live-run/graph/node/999999")
        assert resp.status_code == 404

    def test_live_object_404_missing(self, client: TestClient, live_data_store):
        """Live run returns 404 for missing object UUID."""
        resp = client.get("/api/runs/live-run/graph/object/999999")
        assert resp.status_code == 404
