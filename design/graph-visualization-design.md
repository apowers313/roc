# Graph Visualization Design

## Problem

ROC builds a rich directed graph at each game step -- Frames, Objects, ObjectInstances, FeatureGroups, Transforms, IntrinsicNodes, and their connecting edges. This graph is the core data structure of the system, but there is currently no way to visualize it. The only graph-related data exposed to the dashboard is `graph_summary` (cache utilization counts: node_count, node_max, edge_count, edge_max), which tells you nothing about the graph's structure.

Understanding what the graph looks like at each step is essential for debugging object resolution, transform computation, prediction logic, and the overall pipeline. We need:

1. A way to **archive the full graph** so historical runs have graph data available
2. A **graph data API** to query subgraphs by frame, node, or object
3. A **frontend visualization** using Cytoscape.js in the dashboard

## Design Decisions

### Archive format: Node-Link JSON (not Cytoscape JSON)

The archive format and the wire format (what the API sends to the frontend) are separate concerns.

**Node-Link JSON** (`nx.node_link_data()`) is NetworkX's canonical JSON serialization. It is format-agnostic and does not assume anything about the consumer:

```json
{
  "directed": true,
  "multigraph": false,
  "nodes": [
    {"id": 123, "labels": "Frame,Node", "tick": 42}
  ],
  "links": [
    {"source": 123, "target": 456, "type": "SituatedObjectInstance"}
  ]
}
```

**Cytoscape JSON** (`nx.cytoscape_data()`) nests everything under `.data` and uses `elements.nodes` / `elements.edges` -- a structure that exists solely to satisfy Cytoscape.js's API. Baking a frontend library's format into the archive would create unnecessary coupling.

The graph API converts to Cytoscape JSON (or any other format) at serving time. If the frontend library changes, only the API conversion changes -- not the archived data.

### Data storage: two complementary strategies

The system needs graph data in two contexts with different requirements:

1. **Historical runs** -- comprehensive, queryable, supports arbitrary subgraph extraction
2. **Live games** -- immediate, per-step, supports real-time visualization in the dashboard

These use different strategies because the game subprocess and dashboard server are separate processes with no shared memory (see "Cross-process graph access" below).

#### Historical: Full graph archive at game end

Export the complete graph as node-link JSON to the run directory at game end. The graph API loads this file (cached in memory), builds a NetworkX DiGraph, and derives per-step subgraphs at query time by walking from Frame(tick=N).

- One file per game, written once at game end
- The graph API handles all subgraph extraction, filtering, and format conversion
- Full cross-step connections preserved (NextFrame chains, Object histories, Transform chains)
- Works naturally with the existing DataStore routing pattern

Cache eviction is not a concern: the default cache sizes are `node_cache_size=2**30` and `edge_cache_size=2**30` (over 1 billion entries each). The cache holds every node and edge created during the entire game, so the archive at game end is always complete.

#### Live: Direct GraphCache queries via static methods

During a live game, the graph API queries the in-memory GraphCache directly via static methods on `GraphService` (`subgraph_from_frame_live`, `subgraph_from_node_live`, etc.). The GraphCache holds every node and edge created during the game, so the same BFS subgraph extraction used for historical archives works on live data.

- Zero per-step overhead -- no serialization unless the dashboard actually requests graph data
- Arbitrary depth at query time (not fixed at collection time)
- No config flag needed -- live queries work automatically when a game is running
- Uses the same DataStore `is_live()` routing as other live/historical endpoints
- Live methods traverse `Node.src_edges` / `Node.dst_edges` and build a NetworkX DiGraph on the fly

This replaced the originally planned StepData snapshot approach, which would have serialized a depth-2 subgraph at every step and included it in the HTTP callback. The direct cache query approach is strictly better: no wasted work when the graph panel is closed, and full depth control at query time.

**Rejected alternatives:**

- **Per-step StepData snapshots** -- adds ~1-5ms serialization overhead to every step even when nobody is viewing the graph panel. Depth is fixed at collection time. Requires a config flag to gate. The direct cache query approach avoids all three problems.
- **Incremental diffs / event sourcing** -- random access to step 500 requires replaying 500 diffs; complex reconstruction; fragile if events are missed.
- **Shared memory between processes** -- Python's `multiprocessing.shared_memory` shares raw byte buffers, not Python objects. The graph is made of Pydantic BaseModel instances with inheritance, cross-references, and arbitrary fields. To share it, you'd still need to serialize/deserialize.
- **Reverse query endpoint in game subprocess** -- starting an HTTP server inside the subprocess violates the "one UI server, one API server" architectural invariant and adds threading complexity.
- **Shared Memgraph** -- requires `graphdb_flush=True` (per-step overhead), Memgraph running, and a second GraphDB connection from the dashboard. Too much coupling for the initial implementation.

### Frontend library: Cytoscape.js

**Chosen: Cytoscape.js** (`cytoscape` + `react-cytoscapejs`)

Best fit for this use case:
- Purpose-built for graph/network visualization
- Built-in pan, zoom, click, hover, selection
- Native Cytoscape JSON format (the API converts to this from node-link)
- ~500KB bundle size, acceptable for a dev dashboard
- Note: Cytoscape's built-in layout algorithms (dagre, cose) were tried during prototyping but are **not used**. Dagre's global optimization fights with pinned frame nodes, producing unpredictable placement. We use manual hierarchical positioning with `layout: { name: "preset" }` instead. Cytoscape is still the right choice for its rendering, interaction, and styling capabilities.

**Rejected: D3.js** -- no graph-specific abstractions, much more code to write for comparable features.

**Rejected: Mermaid** -- already installed and has a component, but not interactive (no pan/zoom/click), does not scale past ~30 nodes, static rendering only. Good for schema diagrams, not for per-step graph exploration.

**Rejected: External tools** (Cytoscape Desktop, Gephi) -- breaks the workflow of scrubbing through steps in the dashboard.

### Graph data API: dedicated router (not ad-hoc endpoints)

Graph data will be a growing concern. A dedicated `graph_api.py` module with a FastAPI `APIRouter` mounted at `/api/runs/{run_name}/graph/` keeps the endpoints organized and the traversal logic testable.

The existing `schema` and `graph-history` endpoints should eventually migrate under this router, but that is not required for the initial implementation.

## Graph Schema

### Node Types

| Node Type | File | Key Fields | Notes |
|-----------|------|------------|-------|
| `Frame` | `roc/pipeline/temporal/sequencer.py:68` | `tick: int` | Central hub; auto-incrementing tick |
| `Object` | `roc/pipeline/object/object.py:59` | `uuid: ObjectId`, `annotations: list[str]`, `resolve_count: int`, `last_x`, `last_y`, `last_tick` | Persistent entity across frames |
| `ObjectInstance` | `roc/pipeline/object/object_instance.py:83` | `object_uuid: ObjectId`, `x: XLoc`, `y: YLoc`, `tick: int`, physical features (`glyph_type`, `color_type`, `shape_type`, `flood_size`, `line_size`), relational features (`delta_old`, `delta_new`, `motion_direction`, `distance`) | Per-observation record; implements Transformable |
| `FeatureGroup` | `roc/pipeline/object/object.py:130` | (none beyond base) | Collection of physical feature nodes |
| `RelationshipGroup` | `roc/pipeline/object/object_instance.py:190` | (none beyond base) | Collection of relational feature nodes |
| `FeatureNode` | `roc/perception/base.py:141` | `kind: FeatureKind` (PHYSICAL or RELATIONAL) | Abstract base; 9 concrete subclasses below |
| `TakeAction` | `roc/pipeline/action.py:18` | `action: Any` | Selected action for a step |
| `IntrinsicNode` | `roc/pipeline/intrinsic.py:127` | `name: str`, `raw_value: Any`, `normalized_value: float` | Agent internal state; implements Transformable |
| `Transform` | `roc/pipeline/temporal/transformable.py:39` | (none beyond base) | Abstract base for diffs between frames |
| `ObjectTransform` | `roc/pipeline/object/object_transform.py:76` | `object_uuid: ObjectId`, `num_discrete_changes: int`, `num_continuous_changes: int` | Extends Transform |
| `IntrinsicTransform` | `roc/pipeline/intrinsic.py:162` | (none beyond base) | Extends Transform |
| `PropertyTransformNode` | `roc/pipeline/object/object_transform.py:119` | `property_name: str`, `change_type: str` ("continuous" or "discrete"), `old_value`, `new_value`, `delta: float` | Individual property change within a transform |

**FeatureNode subclasses** (all in `roc/perception/feature_extractors/`):

| Subclass | File | Kind |
|----------|------|------|
| `SingleNode` | `single.py` | PHYSICAL |
| `ColorNode` | `color.py` | PHYSICAL |
| `ShapeNode` | `shape.py` | PHYSICAL |
| `FloodNode` | `flood.py` | PHYSICAL |
| `LineNode` | `line.py` | PHYSICAL |
| `DeltaNode` | `delta.py` | RELATIONAL |
| `MotionNode` | `motion.py` | RELATIONAL |
| `DistanceNode` | `distance.py` | RELATIONAL |
| `PhonemeNode` | `phoneme.py` | PHYSICAL |

### Edge Types

| Edge Type | File | Allowed Connections | Purpose |
|-----------|------|---------------------|---------|
| `FrameAttribute` | `sequencer.py:122` | (Frame, TakeAction), (TakeAction, Frame), (Frame, IntrinsicNode) | Attaches actions and intrinsics to frames |
| `NextFrame` | `sequencer.py:132` | (Frame, Frame) | Temporal ordering of consecutive frames |
| `SituatedObjectInstance` | `object_instance.py:178` | (Frame, ObjectInstance) | Places an instance within a frame |
| `FrameFeatures` | `object_instance.py:184` | (Frame, FeatureGroup) | Links frame to physical feature groups |
| `ObservedAs` | `object_instance.py:172` | (ObjectInstance, Object) | Links instance to persistent object |
| `Features` | `object.py:50` | (Object, FeatureGroup), (ObjectInstance, FeatureGroup) | Object/instance to its feature groups |
| `Relationships` | `object_instance.py:209` | (ObjectInstance, RelationshipGroup) | Instance to relational features |
| `Detail` | `perception/base.py:121` | (FeatureGroup, FeatureNode), (RelationshipGroup, FeatureNode) | Group contains feature nodes |
| `Change` | `transformer.py:23` | (Frame, Transform), (Transform, Frame), (Transform, Transform) | Transform connections between frames |
| `TransformDetail` | `object_transform.py:129` | (ObjectTransform, PropertyTransformNode) | Transform contains property changes |
| `ObjectHistory` | `object_transform.py:135` | (Object, ObjectTransform) | Object accumulates transforms over time |

### Graph Structure at Step N

```
Frame(tick=N)
 |
 |-- NextFrame --> Frame(tick=N-1)
 |
 |-- SituatedObjectInstance --> ObjectInstance(A, x=5, y=10, tick=N)
 |   |-- ObservedAs --> Object(A, uuid=123)
 |   |   |-- Features --> FeatureGroup -- Detail --> [ColorNode, ShapeNode, ...]
 |   |   |-- ObjectHistory --> ObjectTransform
 |   |       |-- TransformDetail --> [PropertyTransformNode, ...]
 |   |-- Features --> FeatureGroup(physical) -- Detail --> [FeatureNodes]
 |   |-- Relationships --> RelationshipGroup -- Detail --> [DeltaNode, MotionNode, ...]
 |
 |-- SituatedObjectInstance --> ObjectInstance(B, x=20, y=3, tick=N)
 |   |-- ObservedAs --> Object(B, uuid=456)
 |   |-- Features --> FeatureGroup(physical)
 |   |-- Relationships --> RelationshipGroup
 |
 |-- FrameFeatures --> FeatureGroup -- Detail --> [FeatureNodes]
 |
 |-- FrameAttribute --> TakeAction(action=19)
 |-- FrameAttribute --> IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
 |-- FrameAttribute --> IntrinsicNode(name="energy", raw_value=30, normalized_value=0.3)
 |
 |-- Change --> Transform --> Change --> Frame(tick=N+1)
```

**Typical node counts per step:**
- Depth 1 (Frame + direct connections): ~5-15 nodes
- Depth 2 (+ FeatureGroups, Objects, Transforms): ~20-50 nodes
- Depth 3 (+ individual FeatureNodes, PropertyTransformNodes): ~50-200 nodes

## Architecture

### Cross-Process Graph Access

The game runs as a subprocess (`subprocess.Popen` in `GameManager`) separate from the dashboard server. Communication is one-directional: the game subprocess POSTs `StepData` to the dashboard server via HTTP callback (`/api/internal/step`). However, the GraphCache (all nodes and edges) lives in the game subprocess's memory and is accessible to code running in that process.

```
Dashboard Server (process A)          Game Subprocess (process B)
  FastAPI + Socket.io                   uv run play
  Serves the frontend                   Runs the pipeline
  Has: DataStore, StepBuffer            Has: GraphCache (all nodes/edges)
  Has: GraphService (live methods)      Does NOT have: API server
       |                                     |
       |  <--- POST /api/internal/step ---   |
       |        (StepData JSON)              |
       |                                     |
       |  GraphService static methods        |
       |  query GraphCache directly          |
```

This drives the two-strategy data storage design:
- **Live:** GraphService static methods query the in-memory GraphCache directly (same process)
- **Historical:** full graph archive is written to disk at game end, loaded by the dashboard server on demand

### Data Flow

```
Game subprocess                    Dashboard server                    Browser
     |                                  |                                |
     |  [each step]                     |                                |
     |  StepData (no graph)             |                                |
     |  --- POST /api/internal/step --> |                                |
     |                                  | store in StepBuffer            |
     |                                  | broadcast via Socket.io ------>|
     |                                  |                                |
     |  (GraphCache holds all           |                                |
     |   nodes/edges in memory)         |                                |
     |                                  |                                |
     |                           [live API request: /graph/frame/N]      |
     |                            GraphService.subgraph_from_frame_live()|
     |                            query GraphCache directly              |
     |                            BFS walk from Frame(tick=N)            |
     |                            build NetworkX DiGraph on the fly      |
     |                            convert to Cytoscape JSON              |
     |                            return response ---------------------->|
     |                                  |                    Cytoscape.js|
     |                                  |                    renders     |
     |                                  |                                |
     |  [game end]                      |                                |
     |  export full graph               |                                |
     |  to run_dir/graph.json           |                                |
     |                                  |                                |
     |                           [historical API request]                |
     |                            load graph.json                        |
     |                            build NetworkX DiGraph (cached)        |
     |                            walk from Frame(tick=N)                |
     |                            convert to Cytoscape JSON              |
     |                            return response ---------------------->|
     |                                  |                    Cytoscape.js|
     |                                  |                    renders     |
```

## Backend Implementation

### 1. Graph Archive Export

**When:** At game end, in `_handle_game_over()` (`roc/game/gymnasium.py`).

**How:** Follow the pattern used by schema export in `State.print_startup_info()` (`roc/reporting/state.py:231-241`).

```python
# In _handle_game_over() or a new function called from there:
def _export_graph_archive() -> None:
    """Export the full graph as node-link JSON to the run directory."""
    from roc.db.graphdb import GraphDB
    store = Observability.get_ducklake_store()
    if store is None:
        return

    G = GraphDB.to_networkx()
    archive_path = store.run_dir / "graph.json"
    data = nx.node_link_data(G)
    archive_path.write_text(json.dumps(data, indent=2, default=str))
```

**File location:** `{data_dir}/{instance_id}/graph.json` alongside `schema.json` and the `ducklake/` directory.

**Considerations:**
- `GraphDB.to_networkx()` already handles all node/edge serialization via `Node.to_dict(include_labels=True)` and `Edge.to_dict(include_type=True)`
- `to_networkx()` accepts `node_ids` and `node_filter` params for selective export if needed
- The export uses `Node.all_ids()` to get all cached node IDs
- For large graphs, `to_networkx()` supports progress callbacks via `node_filter`
- Use `default=str` in `json.dumps` to handle any non-serializable types (numpy types, sets, etc.)

### 2. Graph Data API

**New file:** `roc/reporting/graph_api.py`

**Mounted in:** `roc/reporting/api_server.py` via `app.include_router(graph_router)`

```python
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Annotated

graph_router = APIRouter(prefix="/api/runs/{run_name}/graph", tags=["graph"])
```

#### Endpoints

**GET /api/runs/{run_name}/graph/frame/{tick}**

Returns the subgraph rooted at Frame(tick=N), walked to a configurable depth.

```python
@graph_router.get("/frame/{tick}")
def get_frame_graph(
    run_name: str,
    tick: int,
    depth: Annotated[int, Query(ge=1, le=5)] = 2,
    game: Annotated[int | None, Query()] = None,
    format: Annotated[str, Query()] = "cytoscape",
) -> JSONResponse:
    """Get the subgraph rooted at a specific frame."""
```

Query params:
- `depth` (1-5, default 2): How many edge hops from Frame to include
- `game` (optional): Filter by game number if multiple games in the run
- `format` ("cytoscape" or "node-link", default "cytoscape"): Response format

Response (Cytoscape format):
```json
{
  "elements": {
    "nodes": [
      {"data": {"id": "123", "label": "Frame", "labels": "Frame,Node", "tick": 42}}
    ],
    "edges": [
      {"data": {"id": "e456", "source": "123", "target": "789", "type": "SituatedObjectInstance"}}
    ]
  },
  "meta": {
    "root_id": 123,
    "depth": 2,
    "tick": 42,
    "node_count": 34,
    "edge_count": 41
  }
}
```

**GET /api/runs/{run_name}/graph/node/{node_id}**

Returns the subgraph rooted at any node by ID.

```python
@graph_router.get("/node/{node_id}")
def get_node_graph(
    run_name: str,
    node_id: int,
    depth: Annotated[int, Query(ge=1, le=5)] = 2,
    format: Annotated[str, Query()] = "cytoscape",
) -> JSONResponse:
    """Get the subgraph rooted at a specific node."""
```

**GET /api/runs/{run_name}/graph/object/{uuid}**

Returns all frames where an Object was observed, with its ObjectInstances and Transforms.

```python
@graph_router.get("/object/{uuid}")
def get_object_history(
    run_name: str,
    uuid: int,
    format: Annotated[str, Query()] = "cytoscape",
) -> JSONResponse:
    """Get the full history of an Object across all frames."""
```

Response includes:
- The Object node
- All ObjectInstances linked via ObservedAs edges
- All Frames those instances appear in (via SituatedObjectInstance)
- All ObjectTransforms linked via ObjectHistory edges

**GET /api/runs/{run_name}/graph/stats** (migrated from existing)

Replaces the current `/api/runs/{run_name}/graph-history` endpoint. Returns graph cache statistics over time.

**GET /api/runs/{run_name}/graph/schema** (migrated from existing)

Replaces the current `/api/runs/{run_name}/schema` endpoint.

### 3. Graph Service Layer

**New file:** `roc/reporting/graph_service.py`

Separates graph traversal logic from the API layer so it is testable independently.

```python
class GraphService:
    """Loads and queries graph archives."""

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store
        self._graph_cache: dict[str, nx.DiGraph] = {}  # run_name -> cached graph

    def get_graph(self, run_name: str) -> nx.DiGraph:
        """Load the graph for a run, with caching."""
        if run_name not in self._graph_cache:
            self._graph_cache[run_name] = self._load_graph(run_name)
        return self._graph_cache[run_name]

    def _load_graph(self, run_name: str) -> nx.DiGraph:
        """Load graph from archive file."""
        run_dir = self._data_store.get_run_dir(run_name)
        graph_path = run_dir / "graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"No graph archive for run {run_name}")
        data = json.loads(graph_path.read_text())
        return nx.node_link_graph(data)

    def subgraph_from_frame(
        self, run_name: str, tick: int, depth: int = 2
    ) -> nx.DiGraph:
        """Extract the subgraph rooted at Frame(tick=N) to the given depth."""
        G = self.get_graph(run_name)
        frame_id = self._find_frame(G, tick)
        return self._bfs_subgraph(G, frame_id, depth)

    def subgraph_from_node(
        self, run_name: str, node_id: int, depth: int = 2
    ) -> nx.DiGraph:
        """Extract the subgraph rooted at any node to the given depth."""
        G = self.get_graph(run_name)
        return self._bfs_subgraph(G, node_id, depth)

    def object_history(self, run_name: str, uuid: int) -> nx.DiGraph:
        """Get all nodes/edges related to an Object across all frames."""
        G = self.get_graph(run_name)
        # Find the Object node by uuid, then collect connected
        # ObjectInstances, Frames, ObjectTransforms
        ...

    @staticmethod
    def _find_frame(G: nx.DiGraph, tick: int) -> int:
        """Find the node ID of the Frame with the given tick."""
        for node_id, attrs in G.nodes(data=True):
            if "Frame" in attrs.get("labels", "") and attrs.get("tick") == tick:
                return node_id
        raise ValueError(f"No Frame found with tick={tick}")

    @staticmethod
    def _bfs_subgraph(G: nx.DiGraph, root: int, depth: int) -> nx.DiGraph:
        """BFS from root to the given depth, return induced subgraph."""
        visited = {root}
        frontier = {root}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                # Follow both incoming and outgoing edges
                next_frontier.update(G.successors(node))
                next_frontier.update(G.predecessors(node))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        return G.subgraph(visited).copy()

    @staticmethod
    def to_cytoscape(G: nx.DiGraph, root_id: int | None = None) -> dict:
        """Convert a NetworkX subgraph to Cytoscape JSON with metadata."""
        cyto = nx.cytoscape_data(G)
        cyto["meta"] = {
            "root_id": root_id,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        }
        return cyto
```

### 4. Live Game Support: Direct GraphCache Queries

During a live game, the graph API queries the in-memory GraphCache directly via static methods on `GraphService`. No per-step serialization or config flags needed.

#### GraphService Live Methods

**File:** `roc/reporting/graph_service.py`

Static methods that query the live GraphCache and return NetworkX DiGraphs:

```python
@staticmethod
def _find_frame_live(tick: int) -> Node:
    """Find Frame node by tick in the live GraphCache."""

@staticmethod
def _bfs_subgraph_live(root: Node, depth: int) -> nx.DiGraph:
    """BFS from a live Node, building a NetworkX DiGraph."""

@staticmethod
def subgraph_from_frame_live(tick: int, depth: int = 2) -> nx.DiGraph: ...

@staticmethod
def subgraph_from_node_live(node_id: int, depth: int = 2) -> nx.DiGraph: ...

@staticmethod
def object_history_live(uuid: int) -> nx.DiGraph: ...
```

The `_bfs_subgraph_live` helper traverses `node.src_edges` (outgoing) and `node.dst_edges` (incoming) to discover neighbors, mirrors the historical `_bfs_subgraph` but operates on live `Node` objects. Only edges between visited nodes are included in the result graph.

#### API Routing

The graph API routes between live and historical data using `DataStore.is_live()`:

```python
# In graph_api.py, for each endpoint:
if _is_run_live(run_name):
    sub = GraphService.subgraph_from_frame_live(tick, depth=depth)
else:
    svc = _get_graph_service()
    sub = svc.subgraph_from_frame(run_name, tick, depth=depth)
```

Both live and historical paths support arbitrary depth at query time. The frontend fetches graph data via REST API calls, not Socket.io push.

### 5. DataStore Integration

Add graph-related methods to `DataStore` (`roc/reporting/data_store.py`) following the existing routing pattern:

```python
def get_graph_service(self) -> GraphService:
    """Get or create the GraphService instance."""
    if self._graph_service is None:
        self._graph_service = GraphService(self)
    return self._graph_service
```

## Prototype

A working prototype lives in `dashboard-ui/src/prototype/` with a standalone entry point at `dashboard-ui/graph-prototype.html`. It uses mock data extracted from a real game run (`20260329072844-clerically-cornelia-orabel`, game 1, steps 98-102) and can be served via `VITE_DEV_PORT=9055 VITE_HOST=dev.ato.ms npx vite` from the `dashboard-ui/` directory.

The data extraction scripts are in `tmp/extract_real_data.py` and `tmp/generate_mock_data.py`.

### Design validated by prototype

The following design decisions were tested and validated through the prototype:

1. **Frames pinned in a horizontal timeline** -- Frame nodes are always in a fixed row at the top, connected by NextFrame edges. Frame spacing increases dynamically as children are expanded to avoid overlap.

2. **Click-to-expand interaction model** -- clicking a node expands its children below; clicking again collapses. This controls graph complexity incrementally:
   - Click Frame -> shows Action (red triangle) + ObjectInstances (lime circles)
   - Click ObjectInstance -> shows Object (green circle) + FeatureGroup (yellow diamond) + RelationshipGroup (orange diamond)
   - Click FeatureGroup or RelationshipGroup -> shows individual FeatureNode children (gray circles)

3. **Literal node type names in labels** -- labels use the real class names with the glyph character or action name appended: "ObjectInstance @", "Object @", "Action (West)", "FeatureGroup", "FeatureNode ShapeNode".

4. **Distinct shapes and colors per node type** -- each type has a unique shape/color combination. Labels appear below nodes.

5. **IntrinsicNodes in the detail panel, not as graph nodes** -- clicking a Frame shows intrinsics with progress bars and raw values in the side panel. This reduces graph clutter significantly.

6. **Feature separation** -- FeatureGroup contains PHYSICAL features (ColorNode, ShapeNode, SingleNode), RelationshipGroup contains RELATIONAL features (DistanceNode, DeltaNode, MotionNode). This matches the architectural invariant.

7. **Detail panel** -- right side panel shows properties for the clicked node:
   - Frame: action taken (badge), intrinsic bars, objects in frame (with colored glyphs)
   - ObjectInstance: colored glyph in header, features table, relationships table, Object UUID, resolve count
   - Object: colored glyph, UUID, resolve count, first/last seen
   - FeatureGroup/RelationshipGroup: features or relationships table with colored glyph context
   - FeatureNode: name and value

8. **Edge labels on all connections** -- every edge shows its type name (NextFrame, Situated, ObservedAs, Features, Relationships, FrameAttribute, Detail). Edge colors match their target node type.

9. **Manual hierarchical positioning** -- nodes are positioned programmatically in a consistent hierarchy below their parent frame, rather than using a global layout algorithm. This gives predictable, repeatable layouts.

10. **Real game data** -- the prototype uses data extracted from an actual game run via DuckLake queries, including real NetHack glyphs (@, d, ., -, <), real actions (N, SE, W, EAT), and real intrinsic values.

### Node styling (validated)

| Node Type | Shape | Color | Label Example |
|-----------|-------|-------|---------------|
| Frame | round-rectangle | #4C6EF5 (blue) | "Frame 100" |
| ObjectInstance | ellipse | #82C91E (lime) | "ObjectInstance @" |
| Object | ellipse | #40C057 (green) | "Object @" |
| FeatureGroup | diamond | #FAB005 (yellow) | "FeatureGroup" |
| RelationshipGroup | diamond | #FD7E14 (orange) | "RelationshipGroup" |
| TakeAction | triangle | #F03E3E (red) | "Action (West)" |
| FeatureNode | ellipse | #ADB5BD (gray) | "FeatureNode ShapeNode" |
| Transform | round-rectangle | #15AABF (cyan) | (not yet rendered as node) |

### Edge styling (validated)

| Edge Type | Color | Style |
|-----------|-------|-------|
| NextFrame | #4C6EF5 (blue) | solid, straight |
| Situated | #82C91E (lime) | solid, bezier |
| ObservedAs | #40C057 (green) | dashed, straight |
| Features | #FAB005 (yellow) | solid, bezier |
| Relationships | #FD7E14 (orange) | solid, bezier |
| FrameAttribute | #F03E3E (red) | solid, bezier |
| Detail | #868E96 (gray) | solid, bezier |
| Transform | #15AABF (cyan) | dashed, curved |

### Gaps between prototype and production implementation (all resolved)

All prototype gaps have been resolved in the production implementation:

1. **Transform nodes** -- RESOLVED. Transform/ObjectTransform rendered as cyan rounded-rectangle nodes, connected to frames via Change edges. Styled in STYLESHEET with type `transform`.

2. **Data source** -- RESOLVED. Full backend pipeline: graph archive at game end, GraphService for historical, direct GraphCache queries for live, TanStack Query hooks fetching from REST API.

3. **Step navigation** -- RESOLVED. `useFrameGraph` re-fetches on step change (debounced at 300ms). Graph resets expand state on step change.

4. **Action name mapping** -- RESOLVED. `useActionMap` hook fetches from `/api/runs/{run}/action-map`. `lookupActionName()` resolves action IDs to names in labels and detail panels.

5. **Glyph rendering** -- RESOLVED. Glyph characters from graph node data displayed in labels (e.g., "ObjInst @") and detail panel headers.

6. **Dashboard integration** -- RESOLVED. Integrated as accordion Section in App.tsx with Network icon, wired to run/step/game context.

7. **Cytoscape stylesheet ordering** -- RESOLVED. Base node style is first entry in STYLESHEET array, with type-specific styles following.

8. **cytoscape-dagre dependency** -- RESOLVED. Removed from package.json. Prototype files deleted.

## Frontend Implementation

### Component Structure

**New file:** `dashboard-ui/src/components/panels/GraphVisualization.tsx`

Based on the prototype, the component uses:
- `CytoscapeComponent` from `react-cytoscapejs` with `layout={{ name: "preset" }}` (manual positioning)
- State: `expandedNodes: Set<string>` tracking which nodes are expanded
- `buildElements()` function that computes Cytoscape elements with explicit positions based on expand state
- Detail panel as a conditional sibling `<Paper>` rendered beside the graph

### TanStack Query Hook

**Add to:** `dashboard-ui/src/api/queries.ts`

```tsx
export function useFrameGraph(run: string, tick: number, game?: number) {
  return useQuery({
    queryKey: ["frame-graph", run, tick, game],
    queryFn: () => fetchFrameGraph(run, tick, game),
    staleTime: Infinity,  // graph data for a given step is immutable
    enabled: !!run && tick >= 0,
  });
}
```

### Dashboard Integration

Add as a Section in the Accordion or as a PopoutPanel. The graph visualization needs more vertical space than typical chart panels:

```tsx
<Section value="graph-visualization" title="Graph Visualization" icon={Network} color="cyan">
  <GraphVisualization run={run} game={game || undefined} currentStep={step} />
</Section>
```

### Layout Strategy

**Manual hierarchical positioning** (not dagre, not force-directed). Each frame's children are positioned in a consistent tree below it:

- Level 0 (Y=50): Frame nodes, spaced dynamically based on child count
- Level 1 (Y=140): Action + ObjectInstances, centered below parent frame
- Level 2 (Y=230): Object + FeatureGroup + RelationshipGroup, centered below parent instance
- Level 3 (Y=320): FeatureNodes, centered below parent group

Frame spacing is computed dynamically: `measureFrameWidth()` counts leaf-level columns for each frame's subtree, then frames are spaced to avoid overlap.

### Interaction

- **Click node**: expand/collapse children AND show detail panel
- **Click edge**: show edge detail in panel (transform changes shown as from/to table)
- **Click background**: close detail panel
- **Pan/zoom**: built into Cytoscape.js (wheelSensitivity=0.3 for iPad)

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `roc/reporting/graph_api.py` | FastAPI router with graph endpoints |
| `roc/reporting/graph_service.py` | Graph loading, caching, traversal, format conversion |
| `dashboard-ui/src/components/panels/GraphVisualization.tsx` | Cytoscape.js visualization panel |

### Modified Files

| File | Changes |
|------|---------|
| `roc/game/gymnasium.py` | Add `_export_graph_archive()` in `_handle_game_over()` |
| `roc/reporting/api_server.py` | Mount `graph_router` from `graph_api.py` |
| `roc/reporting/data_store.py` | Add `get_graph_service()` method, expose `get_run_dir()` |
| `dashboard-ui/src/api/client.ts` | Add `fetchFrameGraph()`, `fetchNodeGraph()`, `fetchObjectHistoryGraph()` |
| `dashboard-ui/src/api/queries.ts` | Add `useFrameGraph()`, `useObjectHistoryGraph()` query hooks |
| `dashboard-ui/src/types/step-data.ts` | Add `CytoscapeData` interface |
| `dashboard-ui/src/App.tsx` | Add GraphVisualization section to the accordion |
| `dashboard-ui/package.json` | Add `cytoscape`, `react-cytoscapejs` dependencies |

### New Dependencies

**Python:** None (NetworkX already installed)

**JavaScript:**
- `cytoscape` -- core graph visualization library (ships its own types, no `@types/cytoscape` needed)
- `react-cytoscapejs` -- React wrapper for Cytoscape.js (custom `.d.ts` provides type definitions)

## Testing

### Backend Tests

**Unit tests** (`tests/unit/reporting/`):

- `test_graph_service.py`:
  - `test_load_graph_from_archive` -- loads a node-link JSON file, verifies NetworkX DiGraph
  - `test_subgraph_from_frame` -- walks from Frame(tick=N), verifies correct nodes at each depth
  - `test_subgraph_depth_1` -- only direct connections to Frame
  - `test_subgraph_depth_2` -- includes FeatureGroups, Objects
  - `test_subgraph_depth_3` -- includes individual FeatureNodes
  - `test_find_frame_by_tick` -- finds correct Frame node
  - `test_find_frame_missing_tick` -- raises ValueError for nonexistent tick
  - `test_object_history` -- collects all frames/instances for an Object
  - `test_to_cytoscape` -- conversion includes meta block
  - `test_graph_cache` -- second load returns cached graph

- `test_graph_api.py`:
  - `test_get_frame_graph_endpoint` -- returns Cytoscape JSON with correct structure
  - `test_get_frame_graph_depth_param` -- respects depth query param
  - `test_get_frame_graph_not_found` -- 404 for missing run or tick
  - `test_get_frame_graph_live_uses_snapshot` -- live run returns data from StepBuffer graph_snapshot
  - `test_get_frame_graph_live_no_snapshot` -- 404 when emit_state_graph is disabled
  - `test_get_frame_graph_historical_uses_archive` -- historical run walks the archived graph
  - `test_get_node_graph_endpoint` -- subgraph from arbitrary node
  - `test_get_object_history_endpoint` -- object history response
  - `test_graph_archive_missing` -- 404 when graph.json does not exist

**Unit tests** (`tests/unit/game/`):

- `test_graph_snapshot.py`:
  - `test_collect_graph_snapshot_disabled` -- returns None when `emit_state_graph=False`
  - `test_collect_graph_snapshot_enabled` -- returns node-link JSON dict when enabled
  - `test_collect_graph_snapshot_depth` -- respects depth parameter, correct node count at each depth
  - `test_collect_graph_snapshot_format` -- output has `nodes` and `links` keys (node-link format)
  - `test_bfs_collect_visits_both_directions` -- follows both src and dst edges

**Integration tests** (`tests/integration/`):

- `test_graph_archive.py`:
  - `test_graph_export_at_game_end` -- verify graph.json is written to run_dir
  - `test_graph_archive_format` -- verify node-link JSON structure, all nodes/edges present
  - `test_graph_archive_round_trip` -- export, load into NetworkX, verify same structure

## Implementation Order

1. **Graph archive export** -- add `_export_graph_archive()` to gymnasium.py and `emit_state_graph` config flag. This is the data foundation; everything else depends on it.
2. **Live graph snapshot** -- add `_collect_graph_snapshot()` to `_collect_step_data()`, add `graph_snapshot` field to StepData. This gives live games graph data through the existing callback.
3. **GraphService** -- implement graph_service.py with loading, caching, and BFS subgraph extraction for historical runs.
4. **Graph API** -- implement graph_api.py with the frame and node endpoints, routing between live (StepBuffer) and historical (GraphService) sources.
5. **Frontend basics** -- install dependencies, create GraphVisualization panel with Cytoscape.js, wire up to the frame endpoint.
6. **Frontend polish** -- add depth control, layout toggle, node click inspection, styling.
7. **Object history endpoint** -- implement after the basics are working.

## Future Considerations

- **Graph search:** "Find all Objects that were observed at position (x, y)" or "Find all Frames where Object X changed color." Requires index structures or full-graph scan.
- **Time-lapse animation:** Auto-play through steps showing the graph evolving. The frontend scrubs through steps and re-fetches the subgraph at each.
- **Cypher passthrough:** For power users, expose a raw Cypher query endpoint when Memgraph is available.
- **DuckDB graph tables:** If SQL-style analytics across graph data becomes important (e.g., "count Objects by resolve_count across all runs"), add `graph_nodes` and `graph_edges` tables to DuckLake.
- **Demand-driven collection:** Use a shared flag (shared memory boolean, file, or environment variable) so the game subprocess only serializes graph snapshots when the dashboard has the graph panel open. Avoids per-step serialization overhead when nobody is watching. Not needed initially since `emit_state_graph` defaults to False.
