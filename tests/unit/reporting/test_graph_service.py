# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/graph_service.py -- graph traversal and format conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from roc.reporting.graph_service import GraphService


def _build_test_graph() -> nx.DiGraph:
    """Build a small graph matching the ROC schema shape.

    Structure:
        Frame(tick=1) --FrameAttribute--> TakeAction(action_id=19)
        Frame(tick=1) --FrameAttribute--> IntrinsicNode(name="hp")
        Frame(tick=1) --SituatedObjectInstance--> ObjectInstance
        ObjectInstance --ObservedAs--> Object(uuid=42)
        ObjectInstance --Features--> FeatureGroup
        FeatureGroup --Detail--> FeatureNode(name="color", value="red")

        Frame(tick=2) --NextFrame-- (from tick=1)
        Frame(tick=2) --FrameAttribute--> TakeAction(action_id=5)
        Frame(tick=2) --SituatedObjectInstance--> ObjectInstance2
        ObjectInstance2 --ObservedAs--> Object(uuid=42)  (same object)
    """
    G = nx.DiGraph()

    # Frame 1 and its children
    G.add_node(1, labels="Frame", tick=1)
    G.add_node(2, labels="TakeAction", action_id=19)
    G.add_node(3, labels="IntrinsicNode", name="hp", raw_value=10, normalized_value=0.5)
    G.add_node(4, labels="ObjectInstance", x=5, y=10)
    G.add_node(5, labels="Object", uuid=42, resolve_count=2)
    G.add_node(6, labels="FeatureGroup")
    G.add_node(7, labels="FeatureNode", name="color", value="red", kind="PHYSICAL")

    G.add_edge(1, 2, type="FrameAttribute")
    G.add_edge(1, 3, type="FrameAttribute")
    G.add_edge(1, 4, type="SituatedObjectInstance")
    G.add_edge(4, 5, type="ObservedAs")
    G.add_edge(4, 6, type="Features")
    G.add_edge(6, 7, type="Detail")

    # Frame 2 and its children
    G.add_node(10, labels="Frame", tick=2)
    G.add_node(11, labels="TakeAction", action_id=5)
    G.add_node(12, labels="ObjectInstance", x=6, y=10)

    G.add_edge(1, 10, type="NextFrame")
    G.add_edge(10, 11, type="FrameAttribute")
    G.add_edge(10, 12, type="SituatedObjectInstance")
    G.add_edge(12, 5, type="ObservedAs")  # same Object

    return G


class TestFindFrame:
    def test_find_frame_by_tick(self):
        """Finds the correct Frame node among multiple frames."""
        G = _build_test_graph()
        node_id = GraphService._find_frame(G, tick=1)
        assert node_id == 1

    def test_find_frame_tick_2(self):
        """Finds frame with tick=2."""
        G = _build_test_graph()
        node_id = GraphService._find_frame(G, tick=2)
        assert node_id == 10

    def test_find_frame_missing_tick(self):
        """Raises ValueError when tick does not exist."""
        G = _build_test_graph()
        with pytest.raises(ValueError, match="No Frame node.*tick=99"):
            GraphService._find_frame(G, tick=99)


class TestBfsSubgraph:
    def test_subgraph_depth_1(self):
        """Depth 1 returns only Frame's direct neighbors."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        # Root (1) + direct neighbors: 2, 3, 4, 10
        assert 1 in sub.nodes
        assert 2 in sub.nodes  # TakeAction
        assert 3 in sub.nodes  # IntrinsicNode
        assert 4 in sub.nodes  # ObjectInstance
        assert 10 in sub.nodes  # Frame 2 (via NextFrame)
        # Should NOT include depth-2 nodes
        assert 5 not in sub.nodes  # Object (2 hops from frame 1)
        assert 6 not in sub.nodes  # FeatureGroup
        assert 7 not in sub.nodes  # FeatureNode

    def test_subgraph_depth_2(self):
        """Depth 2 includes Objects, FeatureGroups, etc."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=2)
        assert 1 in sub.nodes
        assert 5 in sub.nodes  # Object (via ObjectInstance)
        assert 6 in sub.nodes  # FeatureGroup (via ObjectInstance)
        # FeatureNode is 3 hops from Frame: Frame->ObjInst->FeatGroup->FeatNode
        assert 7 not in sub.nodes

    def test_subgraph_depth_3(self):
        """Depth 3 includes individual FeatureNodes."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=3)
        assert 7 in sub.nodes  # FeatureNode is now reachable

    def test_bfs_follows_both_directions(self):
        """BFS follows both incoming and outgoing edges."""
        G = nx.DiGraph()
        G.add_node(1, labels="A")
        G.add_node(2, labels="B")
        G.add_node(3, labels="C")
        # 2 -> 1 (incoming to root=1), 1 -> 3 (outgoing from root=1)
        G.add_edge(2, 1, type="points_to")
        G.add_edge(1, 3, type="goes_to")

        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        assert 1 in sub.nodes
        assert 2 in sub.nodes  # reached via incoming edge
        assert 3 in sub.nodes  # reached via outgoing edge

    def test_subgraph_preserves_edges(self):
        """Subgraph includes edges between included nodes."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        assert sub.has_edge(1, 2)
        assert sub.has_edge(1, 4)

    def test_subgraph_depth_0(self):
        """Depth 0 returns only the root node."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=0)
        assert list(sub.nodes) == [1]
        assert len(sub.edges) == 0

    def test_max_nodes_limits_subgraph_size(self):
        """BFS stops early when max_nodes is reached."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=5, max_nodes=3)
        assert len(sub.nodes) <= 3
        assert 1 in sub.nodes  # Root is always included

    def test_max_nodes_includes_root(self):
        """Root node is always included even with max_nodes=1."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=5, max_nodes=1)
        assert len(sub.nodes) == 1
        assert 1 in sub.nodes


class TestToCytoscape:
    def test_to_cytoscape_structure(self):
        """Cytoscape output has elements.nodes, elements.edges, and meta."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        result = GraphService.to_cytoscape(sub, root_id=1)

        assert "elements" in result
        assert "nodes" in result["elements"]
        assert "edges" in result["elements"]
        assert "meta" in result
        assert result["meta"]["root_id"] == 1
        assert result["meta"]["node_count"] == len(sub.nodes)
        assert result["meta"]["edge_count"] == len(sub.edges)

    def test_to_cytoscape_node_ids_are_strings(self):
        """Cytoscape node IDs are strings (Cytoscape.js requirement)."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        result = GraphService.to_cytoscape(sub, root_id=1)

        for node in result["elements"]["nodes"]:
            assert isinstance(node["data"]["id"], str)

    def test_to_cytoscape_edge_source_target_strings(self):
        """Cytoscape edge source/target are strings."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        result = GraphService.to_cytoscape(sub, root_id=1)

        for edge in result["elements"]["edges"]:
            assert isinstance(edge["data"]["source"], str)
            assert isinstance(edge["data"]["target"], str)

    def test_to_cytoscape_node_has_label(self):
        """Cytoscape nodes include the labels attribute."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        result = GraphService.to_cytoscape(sub, root_id=1)

        ids_to_labels = {
            n["data"]["id"]: n["data"].get("labels") for n in result["elements"]["nodes"]
        }
        assert ids_to_labels["1"] == "Frame"

    def test_to_cytoscape_no_root(self):
        """Cytoscape output works with root_id=None."""
        G = _build_test_graph()
        sub = GraphService._bfs_subgraph(G, root=1, depth=1)
        result = GraphService.to_cytoscape(sub, root_id=None)
        assert result["meta"]["root_id"] is None


class TestGraphServiceIO:
    def test_get_graph_loads_from_file(self, tmp_path: Path):
        """get_graph loads a graph.json file into a NetworkX DiGraph."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        # Write a minimal node-link JSON
        G = _build_test_graph()
        data = nx.node_link_data(G, edges="links")
        with open(run_dir / "graph.json", "w") as f:
            json.dump(data, f)

        svc = GraphService(tmp_path)
        loaded = svc.get_graph("test-run")
        assert isinstance(loaded, nx.DiGraph)
        assert len(loaded.nodes) == len(G.nodes)
        assert len(loaded.edges) == len(G.edges)

    def test_graph_cache_reuses_loaded_graph(self, tmp_path: Path):
        """Second call to get_graph() returns cached instance, not re-read."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        G = _build_test_graph()
        data = nx.node_link_data(G, edges="links")
        with open(run_dir / "graph.json", "w") as f:
            json.dump(data, f)

        svc = GraphService(tmp_path)
        g1 = svc.get_graph("test-run")
        g2 = svc.get_graph("test-run")
        assert g1 is g2  # Same object, not re-loaded

    def test_load_graph_file_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError for missing graph.json."""
        svc = GraphService(tmp_path)
        with pytest.raises(FileNotFoundError):
            svc.get_graph("nonexistent-run")

    def test_subgraph_from_frame(self, tmp_path: Path):
        """subgraph_from_frame returns a subgraph rooted at the frame."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        G = _build_test_graph()
        data = nx.node_link_data(G, edges="links")
        with open(run_dir / "graph.json", "w") as f:
            json.dump(data, f)

        svc = GraphService(tmp_path)
        sub = svc.subgraph_from_frame("test-run", tick=1, depth=1)
        assert isinstance(sub, nx.DiGraph)
        assert 1 in sub.nodes

    def test_subgraph_from_node(self, tmp_path: Path):
        """subgraph_from_node returns neighborhood of a specific node."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        G = _build_test_graph()
        data = nx.node_link_data(G, edges="links")
        with open(run_dir / "graph.json", "w") as f:
            json.dump(data, f)

        svc = GraphService(tmp_path)
        sub = svc.subgraph_from_node("test-run", node_id=4, depth=1)
        assert 4 in sub.nodes  # ObjectInstance
        assert 1 in sub.nodes  # Frame (incoming edge)
        assert 5 in sub.nodes  # Object (outgoing edge)

    def test_object_history(self, tmp_path: Path):
        """object_history collects all FeatureGroups/Frames for a given object UUID."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        G = _build_test_graph()
        data = nx.node_link_data(G, edges="links")
        with open(run_dir / "graph.json", "w") as f:
            json.dump(data, f)

        svc = GraphService(tmp_path)
        sub = svc.object_history("test-run", uuid=42)
        assert isinstance(sub, nx.DiGraph)
        # The Object node (uuid=42) should be included
        assert 5 in sub.nodes


# ---------------------------------------------------------------------------
# Live graph tests -- query in-memory GraphCache instead of graph.json
# ---------------------------------------------------------------------------


def _wire_edge(src: Any, dst: Any, etype: str) -> Any:
    """Create an Edge and wire it into the source and destination node edge lists."""
    from roc.db.graphdb import Edge

    e = Edge(type=etype, src_id=src.id, dst_id=dst.id)
    e._no_save = True
    src._src_edges.add(e)
    dst._dst_edges.add(e)
    return e


def _build_live_test_graph() -> dict[str, Any]:
    """Build test nodes and edges in the live GraphCache.

    Structure mirrors ``_build_test_graph()`` but uses real Node/Edge objects
    in the cache rather than a plain NetworkX DiGraph.
    """
    from roc.db.graphdb import Node

    frame1 = Node(labels={"Frame"}, tick=1)
    action1 = Node(labels={"TakeAction"}, action_id=19)
    intrinsic1 = Node(labels={"IntrinsicNode"}, name="hp", raw_value=10, normalized_value=0.5)
    obj_inst1 = Node(labels={"ObjectInstance"}, x=5, y=10)
    obj = Node(labels={"Object"}, uuid=42, resolve_count=2)
    fg1 = Node(labels={"FeatureGroup"})
    fn1 = Node(labels={"FeatureNode"}, name="color", value="red", kind="PHYSICAL")

    frame2 = Node(labels={"Frame"}, tick=2)
    action2 = Node(labels={"TakeAction"}, action_id=5)
    obj_inst2 = Node(labels={"ObjectInstance"}, x=6, y=10)

    _wire_edge(frame1, action1, "FrameAttribute")
    _wire_edge(frame1, intrinsic1, "FrameAttribute")
    _wire_edge(frame1, obj_inst1, "SituatedObjectInstance")
    _wire_edge(obj_inst1, obj, "ObservedAs")
    _wire_edge(obj_inst1, fg1, "Features")
    _wire_edge(fg1, fn1, "Detail")

    _wire_edge(frame1, frame2, "NextFrame")
    _wire_edge(frame2, action2, "FrameAttribute")
    _wire_edge(frame2, obj_inst2, "SituatedObjectInstance")
    _wire_edge(obj_inst2, obj, "ObservedAs")

    return {
        "frame1": frame1,
        "frame2": frame2,
        "action1": action1,
        "action2": action2,
        "intrinsic1": intrinsic1,
        "obj_inst1": obj_inst1,
        "obj_inst2": obj_inst2,
        "obj": obj,
        "fg1": fg1,
        "fn1": fn1,
    }


class TestFindFrameLive:
    def test_find_frame_by_tick(self):
        """Finds the correct Frame node by tick from live cache."""
        g = _build_live_test_graph()
        node = GraphService._find_frame_live(1)
        assert node is g["frame1"]

    def test_find_frame_tick_2(self):
        """Finds frame with tick=2 from live cache."""
        g = _build_live_test_graph()
        node = GraphService._find_frame_live(2)
        assert node is g["frame2"]

    def test_find_frame_missing_tick(self):
        """Raises ValueError when tick does not exist in cache."""
        _build_live_test_graph()
        with pytest.raises(ValueError, match="No Frame node.*tick=99"):
            GraphService._find_frame_live(99)


class TestBfsSubgraphLive:
    def test_depth_0_returns_root_only(self):
        """Depth 0 returns only the root node."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=0)
        assert len(sub.nodes) == 1
        assert g["frame1"].id in sub.nodes

    def test_depth_1_returns_direct_neighbors(self):
        """Depth 1 returns Frame's direct neighbors."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=1)
        assert g["frame1"].id in sub.nodes
        assert g["action1"].id in sub.nodes
        assert g["intrinsic1"].id in sub.nodes
        assert g["obj_inst1"].id in sub.nodes
        assert g["frame2"].id in sub.nodes
        # Depth-2 nodes should NOT be included
        assert g["obj"].id not in sub.nodes
        assert g["fg1"].id not in sub.nodes

    def test_depth_2_includes_object(self):
        """Depth 2 includes Object and FeatureGroup via ObjectInstance."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=2)
        assert g["obj"].id in sub.nodes
        assert g["fg1"].id in sub.nodes
        # FeatureNode is 3 hops away (frame->objinst->fg->fn)
        assert g["fn1"].id not in sub.nodes

    def test_depth_3_includes_feature_node(self):
        """Depth 3 includes individual FeatureNodes."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=3)
        assert g["fn1"].id in sub.nodes

    def test_follows_both_directions(self):
        """BFS follows both incoming and outgoing edges."""
        from roc.db.graphdb import Node

        n1 = Node(labels={"A"})
        n2 = Node(labels={"B"})
        n3 = Node(labels={"C"})
        _wire_edge(n2, n1, "points_to")  # incoming to n1
        _wire_edge(n1, n3, "goes_to")  # outgoing from n1

        sub = GraphService._bfs_subgraph_live(n1, depth=1)
        assert n1.id in sub.nodes
        assert n2.id in sub.nodes  # via incoming edge
        assert n3.id in sub.nodes  # via outgoing edge

    def test_preserves_edges_between_visited_nodes(self):
        """Subgraph includes edges between included nodes."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=1)
        assert sub.has_edge(g["frame1"].id, g["action1"].id)
        assert sub.has_edge(g["frame1"].id, g["obj_inst1"].id)

    def test_node_has_labels_string(self):
        """NetworkX node data includes labels as a string, not a set."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=0)
        data = sub.nodes[g["frame1"].id]
        assert isinstance(data["labels"], str)
        assert "Frame" in data["labels"]

    def test_edge_has_type(self):
        """NetworkX edge data includes the type attribute."""
        g = _build_live_test_graph()
        sub = GraphService._bfs_subgraph_live(g["frame1"], depth=1)
        edge_data = sub.edges[g["frame1"].id, g["action1"].id]
        assert edge_data["type"] == "FrameAttribute"


class TestSubgraphFromFrameLive:
    def test_returns_digraph(self):
        """subgraph_from_frame_live returns a NetworkX DiGraph."""
        _build_live_test_graph()
        sub = GraphService.subgraph_from_frame_live(tick=1, depth=1)
        assert isinstance(sub, nx.DiGraph)

    def test_missing_tick_raises(self):
        """ValueError when tick not found in cache."""
        _build_live_test_graph()
        with pytest.raises(ValueError, match="No Frame node.*tick=99"):
            GraphService.subgraph_from_frame_live(tick=99)


class TestSubgraphFromNodeLive:
    def test_returns_neighborhood(self):
        """subgraph_from_node_live returns neighborhood of a specific node."""
        g = _build_live_test_graph()
        sub = GraphService.subgraph_from_node_live(g["obj_inst1"].id, depth=1)
        assert g["obj_inst1"].id in sub.nodes
        assert g["frame1"].id in sub.nodes  # incoming edge
        assert g["obj"].id in sub.nodes  # outgoing edge

    def test_missing_node_raises(self):
        """ValueError when node not found in cache."""
        _build_live_test_graph()
        with pytest.raises(ValueError, match="not found in live cache"):
            GraphService.subgraph_from_node_live(999999, depth=1)


class TestObjectHistoryLive:
    def test_collects_object_and_instances(self):
        """object_history_live collects Object + ObjectInstances + Frames."""
        g = _build_live_test_graph()
        sub = GraphService.object_history_live(uuid=42)
        assert isinstance(sub, nx.DiGraph)
        # Object node
        assert g["obj"].id in sub.nodes
        # ObjectInstances (predecessors of Object via ObservedAs)
        assert g["obj_inst1"].id in sub.nodes
        assert g["obj_inst2"].id in sub.nodes
        # Frames (predecessors of ObjectInstances via SituatedObjectInstance)
        assert g["frame1"].id in sub.nodes
        assert g["frame2"].id in sub.nodes

    def test_missing_uuid_raises(self):
        """ValueError when UUID not found in cache."""
        _build_live_test_graph()
        with pytest.raises(ValueError, match="No Object node with uuid=999"):
            GraphService.object_history_live(uuid=999)
