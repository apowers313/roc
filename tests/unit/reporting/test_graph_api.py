# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/graph_api.py -- Graph Data API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app


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
    """Point the API at a RunRegistry backed by a temp directory."""
    import roc.reporting.api_server as mod

    orig_registry = mod._run_registry
    orig_reader = mod._run_reader
    mod.init_data_dir(tmp_path)
    mod._run_reader = None
    yield
    mod._run_registry = orig_registry
    mod._run_reader = orig_reader


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


class TestObjectUuidPrecision:
    """The /graph/object/{uuid} endpoint must accept and round-trip a 63-bit
    Object UUID without precision loss. ROC's UUIDs exceed JS Number's safe
    integer range (2^53 - 1), so the path param is a string and the
    response stringifies the uuid in the response body. This pins the
    failure mode that hit production: 6853809301722933453 -> 6853809301722933000.
    """

    BIG_UUID = 6853809301722933453

    def test_path_param_accepts_63bit_uuid_string(self, client: TestClient, mock_graph_service):
        """The handler parses the string path param into the int the
        graph service expects, and forwards the *exact* value -- no JS
        Number precision loss because the int never goes through JS."""
        G = nx.DiGraph()
        G.add_node(-1, labels="Object", uuid=self.BIG_UUID, resolve_count=1)
        mock_graph_service.object_history.return_value = G

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get(f"/api/runs/test-run/graph/object/{self.BIG_UUID}")
        assert resp.status_code == 200
        # The graph service was called with the exact int (lossless).
        mock_graph_service.object_history.assert_called_once_with("test-run", self.BIG_UUID)

    def test_invalid_uuid_returns_400(self, client: TestClient, mock_graph_service):
        """Non-numeric path param is rejected with a 400 Bad Request."""
        resp = client.get("/api/runs/test-run/graph/object/not-a-number")
        assert resp.status_code == 400

    def test_response_uuid_is_string(self, client: TestClient, mock_graph_service):
        """The uuid field in the response body is a JSON string, not a
        JSON number. A JS client parses strings as `string`, never as
        Number, so the precision is preserved end-to-end."""
        G = nx.DiGraph()
        G.add_node(-1, labels="Object", uuid=self.BIG_UUID, resolve_count=189)
        mock_graph_service.object_history.return_value = G

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get(f"/api/runs/test-run/graph/object/{self.BIG_UUID}")
        assert resp.status_code == 200
        body = resp.json()
        nodes = body["elements"]["nodes"]
        obj = next(n for n in nodes if n["data"].get("labels") == "Object")
        assert isinstance(obj["data"]["uuid"], str)
        assert obj["data"]["uuid"] == str(self.BIG_UUID)
        # And the raw JSON text contains the uuid as a quoted string.
        # `"6853809301722933453"`, not `6853809301722933453`.
        assert f'"{self.BIG_UUID}"' in resp.text

    def test_response_includes_human_name(self, client: TestClient, mock_graph_service):
        """Object nodes carry a derived human_name (FlexiHumanHash) so the
        UI has something readable to display alongside the 19-digit uuid."""
        G = nx.DiGraph()
        G.add_node(-1, labels="Object", uuid=self.BIG_UUID, resolve_count=1)
        mock_graph_service.object_history.return_value = G

        from roc.reporting.graph_service import GraphService

        mock_graph_service.to_cytoscape = GraphService.to_cytoscape

        resp = client.get(f"/api/runs/test-run/graph/object/{self.BIG_UUID}")
        body = resp.json()
        obj = next(n for n in body["elements"]["nodes"] if n["data"].get("labels") == "Object")
        assert "human_name" in obj["data"]
        assert isinstance(obj["data"]["human_name"], str)
        assert len(obj["data"]["human_name"]) > 0


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
