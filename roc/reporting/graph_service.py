"""Graph archive loading, BFS subgraph extraction, and format conversion.

Loads node-link JSON archives (written at game end) into NetworkX digraphs,
provides BFS-based subgraph extraction around frames or arbitrary nodes,
and converts subgraphs to Cytoscape.js-compatible JSON.

Live graph methods query the in-memory GraphCache directly for real-time
graph exploration while a game is running or recently stopped.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from roc.db.graphdb import Node


class GraphService:
    """Loads graph archives and provides subgraph extraction and format conversion.

    Args:
        data_dir: Root data directory containing per-run subdirectories.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: dict[str, nx.DiGraph] = {}

    def get_graph(self, run_name: str) -> nx.DiGraph:
        """Load the graph archive for a run, returning a cached instance if available.

        Args:
            run_name: Name of the run directory.

        Returns:
            The loaded NetworkX DiGraph.

        Raises:
            FileNotFoundError: If graph.json does not exist for the run.
        """
        if run_name in self._cache:
            return self._cache[run_name]

        graph_path = self._data_dir / run_name / "graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"No graph archive found at {graph_path}")

        with open(graph_path) as f:
            data = json.load(f)

        G = nx.node_link_graph(data, edges="links")
        self._cache[run_name] = G
        return G

    def subgraph_from_frame(
        self, run_name: str, tick: int, depth: int = 2, max_nodes: int = 500
    ) -> nx.DiGraph:
        """Extract a BFS subgraph rooted at the Frame with the given tick.

        Args:
            run_name: Name of the run directory.
            tick: The frame tick number.
            depth: BFS traversal depth.
            max_nodes: Maximum number of nodes to include. BFS stops early
                when this limit is reached.

        Returns:
            A subgraph containing nodes within ``depth`` hops of the frame.
        """
        G = self.get_graph(run_name)
        root = self._find_frame(G, tick)
        return self._bfs_subgraph(G, root, depth, max_nodes=max_nodes)

    def subgraph_from_node(self, run_name: str, node_id: int, depth: int = 2) -> nx.DiGraph:
        """Extract a BFS subgraph rooted at an arbitrary node.

        Args:
            run_name: Name of the run directory.
            node_id: The node ID to center the subgraph on.
            depth: BFS traversal depth.

        Returns:
            A subgraph containing nodes within ``depth`` hops of the node.
        """
        G = self.get_graph(run_name)
        return self._bfs_subgraph(G, node_id, depth)

    def object_history(self, run_name: str, uuid: int) -> nx.DiGraph:
        """Collect all graph structure related to an Object by UUID.

        Finds the Object node with the given UUID, then collects all
        ObjectInstances that ObservedAs this Object, and all Frames those
        instances belong to, plus their edges.

        Args:
            run_name: Name of the run directory.
            uuid: The object UUID.

        Returns:
            A subgraph containing the object's full history.
        """
        G = self.get_graph(run_name)

        # Find the Object node with this UUID
        object_node = None
        for node_id, data in G.nodes(data=True):
            labels = data.get("labels", "")
            node_uuid = data.get("uuid")
            if "Object" in str(labels) and node_uuid == uuid:
                object_node = node_id
                break

        if object_node is None:
            raise ValueError(f"No Object node with uuid={uuid}")

        # Collect related nodes: Object + all predecessors (ObjectInstances via
        # ObservedAs) + their predecessors (Frames via SituatedObjectInstance)
        collected: set[int] = {object_node}

        # Find ObjectInstances that point to this Object
        for pred in G.predecessors(object_node):
            collected.add(pred)
            # Find Frames that point to those ObjectInstances
            for frame_pred in G.predecessors(pred):
                collected.add(frame_pred)

        return G.subgraph(collected).copy()

    @staticmethod
    def to_cytoscape(G: nx.DiGraph, root_id: int | None = None) -> dict[str, Any]:
        """Convert a NetworkX DiGraph to Cytoscape.js JSON format.

        Args:
            G: The graph to convert.
            root_id: Optional root node ID for metadata.

        Returns:
            Dict with 'elements' (nodes/edges) and 'meta'.
        """
        nodes = []
        for node_id, data in G.nodes(data=True):
            node_data = {"id": str(node_id)}
            node_data.update({k: v for k, v in data.items()})
            nodes.append({"data": node_data})

        edges = []
        for src, dst, data in G.edges(data=True):
            edge_data = {
                "id": f"{src}-{dst}",
                "source": str(src),
                "target": str(dst),
            }
            edge_data.update({k: v for k, v in data.items()})
            edges.append({"data": edge_data})

        return {
            "elements": {
                "nodes": nodes,
                "edges": edges,
            },
            "meta": {
                "root_id": root_id,
                "node_count": len(G.nodes),
                "edge_count": len(G.edges),
            },
        }

    @staticmethod
    def _find_frame(G: nx.DiGraph, tick: int) -> int:
        """Find the Frame node with the given tick.

        Args:
            G: The graph to search.
            tick: The tick value to find.

        Returns:
            The node ID of the matching Frame.

        Raises:
            ValueError: If no Frame with the given tick exists.
        """
        for node_id, data in G.nodes(data=True):
            labels = data.get("labels", "")
            if "Frame" in str(labels) and data.get("tick") == tick:
                return int(node_id)
        raise ValueError(f"No Frame node with tick={tick}")

    @staticmethod
    def _bfs_subgraph(G: nx.DiGraph, root: int, depth: int, max_nodes: int = 500) -> nx.DiGraph:
        """Extract a subgraph via BFS from root, following edges in both directions.

        Args:
            G: The full graph.
            root: Starting node ID.
            depth: Maximum traversal depth.
            max_nodes: Maximum number of nodes to include. BFS stops early
                when this limit is reached.

        Returns:
            A new DiGraph containing all nodes within ``depth`` hops and edges
            between them.
        """
        visited: set[int] = {root}
        queue: deque[tuple[int, int]] = deque([(root, 0)])

        while queue:
            if len(visited) >= max_nodes:
                break
            node, d = queue.popleft()
            if d >= depth:
                continue
            # Follow outgoing edges
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
            # Follow incoming edges
            for neighbor in G.predecessors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
                    if len(visited) >= max_nodes:
                        break

        return G.subgraph(visited).copy()

    # ------------------------------------------------------------------
    # Live graph queries -- read from in-memory GraphCache
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_nx_data(node: Node) -> dict[str, Any]:
        """Convert a live Node to a NetworkX-compatible attribute dict.

        Labels are converted from a set to a sorted comma-separated string
        to match the format produced by historical graph.json archives.
        """
        from roc.db.graphdb import Node as NodeCls

        n_data = NodeCls.to_dict(node, include_labels=True)
        if "labels" in n_data and isinstance(n_data["labels"], set):
            n_data["labels"] = ", ".join(sorted(n_data["labels"]))
        return n_data

    @staticmethod
    def _find_frame_live(tick: int) -> Node:
        """Find a Frame node with the given tick from the live GraphCache.

        Args:
            tick: The frame tick number.

        Returns:
            The matching Node object.

        Raises:
            ValueError: If no Frame with the given tick exists in cache.
        """
        from roc.db.graphdb import Node as NodeCls

        cache = NodeCls.get_cache()
        for node_id in cache:
            node = cache[node_id]
            if "Frame" in node.labels and getattr(node, "tick", None) == tick:
                return node
        raise ValueError(f"No Frame node with tick={tick}")

    @staticmethod
    def _bfs_subgraph_live(root: Node, depth: int) -> nx.DiGraph:
        """BFS subgraph from a live Node, building a NetworkX DiGraph.

        Traverses edges in both directions (outgoing via ``src_edges``,
        incoming via ``dst_edges``).  Only edges whose both endpoints are
        within the visited set are included in the result.

        Args:
            root: The starting Node object from the live cache.
            depth: Maximum BFS traversal depth.

        Returns:
            A new DiGraph containing all nodes within *depth* hops and
            the edges between them.
        """
        from roc.db.graphdb import Edge as EdgeCls

        visited: dict[int, Any] = {root.id: root}
        frontier: set[int] = {root.id}

        for _ in range(depth):
            next_frontier: set[int] = set()
            for nid in frontier:
                node = visited[nid]
                for edge in node.src_edges:
                    dst = edge.dst
                    if dst.id not in visited:
                        visited[dst.id] = dst
                        next_frontier.add(dst.id)
                for edge in node.dst_edges:
                    src = edge.src
                    if src.id not in visited:
                        visited[src.id] = src
                        next_frontier.add(src.id)
            frontier = next_frontier

        # Build NetworkX graph with only visited nodes and internal edges
        G = nx.DiGraph()
        visited_ids = set(visited.keys())

        for nid, node in visited.items():
            G.add_node(nid, **GraphService._node_to_nx_data(node))

        for nid, node in visited.items():
            for edge in node.src_edges:
                if edge.dst_id in visited_ids:
                    e_data = EdgeCls.to_dict(edge, include_type=True)
                    G.add_edge(edge.src_id, edge.dst_id, **e_data)

        return G

    @staticmethod
    def subgraph_from_frame_live(tick: int, depth: int = 2) -> nx.DiGraph:
        """Extract a BFS subgraph rooted at the Frame with the given tick.

        Queries the live in-memory GraphCache instead of a graph.json archive.

        Args:
            tick: The frame tick number.
            depth: BFS traversal depth.

        Returns:
            A subgraph containing nodes within *depth* hops of the frame.
        """
        root = GraphService._find_frame_live(tick)
        return GraphService._bfs_subgraph_live(root, depth)

    @staticmethod
    def subgraph_from_node_live(node_id: int, depth: int = 2) -> nx.DiGraph:
        """Extract a BFS subgraph rooted at an arbitrary node from live cache.

        Args:
            node_id: The node ID to center the subgraph on.
            depth: BFS traversal depth.

        Returns:
            A subgraph containing nodes within *depth* hops of the node.

        Raises:
            ValueError: If the node is not found in the live cache.
        """
        from roc.db.graphdb import Node as NodeCls, NodeId

        cache = NodeCls.get_cache()
        nid = NodeId(node_id)
        if nid not in cache:
            raise ValueError(f"Node {node_id} not found in live cache")
        root = cache[nid]
        return GraphService._bfs_subgraph_live(root, depth)

    @staticmethod
    def object_history_live(uuid: int) -> nx.DiGraph:
        """Collect all graph structure related to an Object by UUID from live cache.

        Finds the Object node, then collects all ObjectInstances that point
        to it (via incoming edges), and all Frames that point to those
        instances.

        Args:
            uuid: The object UUID.

        Returns:
            A subgraph containing the object's full history.

        Raises:
            ValueError: If no Object with the given UUID exists in cache.
        """
        from roc.db.graphdb import Edge as EdgeCls
        from roc.db.graphdb import Node as NodeCls

        cache = NodeCls.get_cache()

        # Find the Object node with this UUID
        object_node = None
        for node_id in cache:
            node = cache[node_id]
            if "Object" in node.labels and getattr(node, "uuid", None) == uuid:
                object_node = node
                break

        if object_node is None:
            raise ValueError(f"No Object node with uuid={uuid}")

        # Collect: Object + predecessors (ObjectInstances) + their predecessors (Frames)
        collected: dict[int, Any] = {object_node.id: object_node}

        for edge in object_node.dst_edges:
            pred = edge.src
            collected[pred.id] = pred
            for inner_edge in pred.dst_edges:
                frame_pred = inner_edge.src
                collected[frame_pred.id] = frame_pred

        # Build NetworkX graph
        G = nx.DiGraph()
        collected_ids = set(collected.keys())

        for nid, node in collected.items():
            G.add_node(nid, **GraphService._node_to_nx_data(node))

        for nid, node in collected.items():
            for edge in node.src_edges:
                if edge.dst_id in collected_ids:
                    e_data = EdgeCls.to_dict(edge, include_type=True)
                    G.add_edge(edge.src_id, edge.dst_id, **e_data)

        return G
