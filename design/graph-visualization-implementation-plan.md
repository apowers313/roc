# Implementation Plan for Graph Visualization

**Status: COMPLETE** -- All 7 phases implemented and verified. See architectural notes in Phase 3 for the live query approach change from the original design.

## Overview

Add graph visualization to the ROC dashboard: archive full graphs at game end, expose graph data via a dedicated API, deliver live graph queries, and render interactive Cytoscape.js visualizations in the existing dashboard. Based on the design in `design/graph-visualization-design.md` and validated through prototyping (prototype files removed in Phase 7).

## Phase Breakdown

### Phase 1: Graph Archive Export and GraphService

**What this phase accomplishes**: The data foundation. At game end, the full in-memory graph is exported to `graph.json` in the run directory. A new `GraphService` class loads these archives into NetworkX digraphs and provides BFS subgraph extraction and format conversion. No API endpoints yet -- this phase is backend-only, verified by running a game and inspecting the output file, plus unit tests against synthetic graphs.

**Duration**: 2 days

**Tests to Write First**:

- `tests/unit/reporting/test_graph_service.py`: Core graph traversal logic against synthetic NetworkX graphs (no Memgraph required)
  ```python
  # Build a small graph matching the ROC schema shape:
  # Frame(tick=1) --SituatedObjectInstance--> ObjectInstance --ObservedAs--> Object
  #                --FrameAttribute--> TakeAction
  #                --FrameAttribute--> IntrinsicNode
  # ObjectInstance --Features--> FeatureGroup --Detail--> FeatureNode

  def test_find_frame_by_tick():
      """Finds the correct Frame node among multiple frames."""

  def test_find_frame_missing_tick():
      """Raises ValueError when tick does not exist."""

  def test_subgraph_depth_1():
      """Depth 1 returns only Frame's direct neighbors."""

  def test_subgraph_depth_2():
      """Depth 2 includes FeatureGroups, Objects, etc."""

  def test_subgraph_depth_3():
      """Depth 3 includes individual FeatureNodes."""

  def test_bfs_follows_both_directions():
      """BFS follows both incoming and outgoing edges."""

  def test_to_cytoscape_structure():
      """Cytoscape output has elements.nodes, elements.edges, and meta."""

  def test_to_cytoscape_node_ids_are_strings():
      """Cytoscape node IDs are strings (Cytoscape.js requirement)."""

  def test_graph_cache_reuses_loaded_graph():
      """Second call to get_graph() returns cached instance, not re-read."""

  def test_load_graph_file_not_found():
      """Raises FileNotFoundError for missing graph.json."""
  ```

- `tests/unit/game/test_graph_export.py`: Export function in isolation (mock GraphDB/Node)
  ```python
  def test_export_graph_archive_writes_json(tmp_path):
      """Verify graph.json is written to the run directory."""

  def test_export_graph_archive_node_link_format(tmp_path):
      """Output has 'directed', 'nodes', and 'links' keys."""

  def test_export_graph_archive_skips_when_no_store():
      """Returns without error when Observability store is None."""

  def test_export_graph_archive_handles_non_serializable_types(tmp_path):
      """numpy types, sets, etc. are converted via default=str."""
  ```

**Implementation**:

- `roc/reporting/graph_service.py` (new file): `GraphService` class
  ```python
  class GraphService:
      def __init__(self, data_dir: Path) -> None: ...
      def get_graph(self, run_name: str) -> nx.DiGraph: ...
      def subgraph_from_frame(self, run_name: str, tick: int, depth: int = 2) -> nx.DiGraph: ...
      def subgraph_from_node(self, run_name: str, node_id: int, depth: int = 2) -> nx.DiGraph: ...
      def object_history(self, run_name: str, uuid: int) -> nx.DiGraph: ...
      @staticmethod
      def to_cytoscape(G: nx.DiGraph, root_id: int | None = None) -> dict: ...
      @staticmethod
      def _find_frame(G: nx.DiGraph, tick: int) -> int: ...
      @staticmethod
      def _bfs_subgraph(G: nx.DiGraph, root: int, depth: int) -> nx.DiGraph: ...
  ```

- `roc/game/gymnasium.py` (modify): Add `_export_graph_archive()` function, call it from `_handle_game_over()`
  ```python
  def _export_graph_archive() -> None:
      """Export the full graph as node-link JSON to the run directory."""
      # Uses GraphDB.to_networkx(), writes nx.node_link_data() to run_dir/graph.json
  ```

**Dependencies**:
- External: None (NetworkX already installed)
- Internal: `roc/db/graphdb.py` (GraphDB.to_networkx), `roc/reporting/ducklake_store.py` (run_dir path)

**Verification**:
1. Run: `make test` -- all new unit tests pass, existing tests unbroken
2. Run a short game: `roc_num_games=1 roc_nethack_max_turns=20 uv run play`
3. Check: `ls data/*/graph.json` -- file exists in the latest run directory
4. Check: `python -c "import json; d = json.load(open('data/<run>/graph.json')); print(f'nodes: {len(d[\"nodes\"])}, links: {len(d[\"links\"])}')"` -- nonzero counts

---

### Phase 2: Graph Data API (Historical)

**What this phase accomplishes**: A dedicated FastAPI router at `/api/runs/{run_name}/graph/` with endpoints for frame subgraphs, node subgraphs, and object history. All endpoints serve historical data from graph.json archives via GraphService. Live game support comes in Phase 3. The API is testable via `curl` or the browser against any completed run that has a graph.json.

**Duration**: 2 days

**Tests to Write First**:

- `tests/unit/reporting/test_graph_api.py`: FastAPI TestClient tests with mock GraphService
  ```python
  # Use FastAPI's TestClient with a test app that includes graph_router.
  # Mock GraphService to return pre-built NetworkX subgraphs.

  def test_get_frame_graph_returns_cytoscape_json():
      """GET /graph/frame/42 returns Cytoscape JSON with elements and meta."""

  def test_get_frame_graph_respects_depth_param():
      """depth=1 returns fewer nodes than depth=3."""

  def test_get_frame_graph_node_link_format():
      """format=node-link returns node-link JSON instead of Cytoscape."""

  def test_get_frame_graph_404_missing_run():
      """404 when run directory does not exist."""

  def test_get_frame_graph_404_missing_tick():
      """404 when requested tick has no Frame node."""

  def test_get_frame_graph_404_no_archive():
      """404 when graph.json does not exist for the run."""

  def test_get_node_graph_returns_subgraph():
      """GET /graph/node/123 returns neighborhood of that node."""

  def test_get_object_history_returns_all_frames():
      """GET /graph/object/456 includes frames, instances, transforms."""

  def test_depth_validation_rejects_out_of_range():
      """depth=0 or depth=6 returns 422."""
  ```

**Implementation**:

- `roc/reporting/graph_api.py` (new file): FastAPI APIRouter
  ```python
  graph_router = APIRouter(prefix="/api/runs/{run_name}/graph", tags=["graph"])

  @graph_router.get("/frame/{tick}")
  def get_frame_graph(run_name, tick, depth=2, game=None, format="cytoscape"): ...

  @graph_router.get("/node/{node_id}")
  def get_node_graph(run_name, node_id, depth=2, format="cytoscape"): ...

  @graph_router.get("/object/{uuid}")
  def get_object_history(run_name, uuid, format="cytoscape"): ...
  ```

- `roc/reporting/api_server.py` (modify): Mount graph_router
  ```python
  from roc.reporting.graph_api import graph_router
  app.include_router(graph_router)
  ```

- `roc/reporting/data_store.py` (modify): Add `get_graph_service()` method and `get_run_dir()` accessor
  ```python
  def get_graph_service(self) -> GraphService: ...
  def get_run_dir(self, run_name: str) -> Path: ...
  ```

**Dependencies**:
- External: None
- Internal: Phase 1 (GraphService, graph.json archive)

**Verification**:
1. Run: `make test` -- all new and existing tests pass
2. Start servers: `make run`
3. Use an existing run that has graph.json (from Phase 1 verification):
   ```bash
   curl https://dev.ato.ms:<port>/api/runs/<run_name>/graph/frame/1?depth=2 | python -m json.tool
   ```
4. Verify response has `elements.nodes`, `elements.edges`, and `meta` fields
5. Verify 404 for nonexistent tick: `curl -w "%{http_code}" .../graph/frame/99999`

---

### Phase 3: Live Game Graph Queries

**What this phase accomplishes**: During a live game, the graph API queries the in-memory GraphCache directly via static methods on GraphService, then routes between live cache queries and historical graph.json archives. This enables the frontend to show graph data during a running game with zero per-step overhead.

**Duration**: 2 days

**Architecture note**: The original plan called for per-step StepData snapshots (serializing a subgraph at every step and including it in the HTTP callback). During implementation, this was replaced with direct GraphCache queries, which are strictly better: no serialization overhead when the graph panel is closed, arbitrary depth at query time, and no config flag needed.

**Tests to Write First**:

- `tests/unit/reporting/test_graph_service.py` (extend): Live graph query tests
  ```python
  class TestFindFrameLive:
      def test_find_frame_by_tick(): ...
      def test_find_frame_missing_tick(): ...

  class TestBfsSubgraphLive:
      def test_depth_0_returns_root_only(): ...
      def test_depth_1_returns_direct_neighbors(): ...
      def test_depth_2_includes_object(): ...
      def test_follows_both_directions(): ...
      def test_preserves_edges_between_visited_nodes(): ...
      def test_node_has_labels_string(): ...
      def test_edge_has_type(): ...

  class TestSubgraphFromFrameLive: ...
  class TestSubgraphFromNodeLive: ...
  class TestObjectHistoryLive: ...
  ```

- `tests/unit/reporting/test_graph_api.py` (extend): Live routing tests
  ```python
  class TestLiveRouting:
      def test_frame_graph_routes_live(): ...
      def test_frame_graph_routes_historical(): ...
      def test_node_graph_routes_live(): ...
      def test_object_history_routes_live(): ...
      def test_live_frame_404_missing_tick(): ...
      def test_live_node_404_missing(): ...
      def test_live_object_404_missing(): ...
  ```

**Implementation**:

- `roc/reporting/graph_service.py` (modify): Add live query static methods
  ```python
  @staticmethod
  def _find_frame_live(tick: int) -> Node: ...
  @staticmethod
  def _bfs_subgraph_live(root: Node, depth: int) -> nx.DiGraph: ...
  @staticmethod
  def subgraph_from_frame_live(tick: int, depth: int = 2) -> nx.DiGraph: ...
  @staticmethod
  def subgraph_from_node_live(node_id: int, depth: int = 2) -> nx.DiGraph: ...
  @staticmethod
  def object_history_live(uuid: int) -> nx.DiGraph: ...
  ```

- `roc/reporting/graph_api.py` (modify): Add live/historical routing in all endpoints
  ```python
  if _is_run_live(run_name):
      sub = GraphService.subgraph_from_frame_live(tick, depth=depth)
  else:
      svc = _get_graph_service()
      sub = svc.subgraph_from_frame(run_name, tick, depth=depth)
  ```

**Dependencies**:
- External: None
- Internal: Phase 1 (GraphService), Phase 2 (graph API routing)

**Verification**:
1. Run: `make test` -- all tests pass
2. Start servers: `make run`
3. Start a live game via REST: `curl -X POST .../api/game/start`
4. While game is running, query the live frame:
   ```bash
   curl https://dev.ato.ms:<port>/api/runs/<live_run>/graph/frame/<step>?depth=2 | python -m json.tool
   ```
5. Verify response has graph data (nodes and edges), depth parameter is respected
6. After game ends, verify the same endpoint now returns data from the archive

---

### Phase 4: Frontend -- Basic Cytoscape.js Panel

**What this phase accomplishes**: A `GraphVisualization` component integrated into the dashboard accordion. Fetches graph data from the frame endpoint and renders it with Cytoscape.js using preset layout (manual positioning). Frames pinned horizontally, click-to-expand interaction model, node type styling (shapes/colors). No detail panel yet -- just the interactive graph canvas. This phase ports the core rendering logic from the prototype, replacing mock data with live API calls.

**Duration**: 3 days

**Tests to Write First**:

- `dashboard-ui/src/components/panels/GraphVisualization.test.tsx`: Component rendering tests
  ```typescript
  describe("GraphVisualization", () => {
    it("renders loading state when data is undefined", () => {
      // Mock useFrameGraph to return { data: undefined, isLoading: true }
      // Verify loading indicator is shown
    });

    it("renders 'no data' message when API returns empty graph", () => {
      // Mock useFrameGraph to return { data: { elements: { nodes: [], edges: [] } } }
      // Verify "No graph data" text
    });

    it("renders Cytoscape component when data is available", () => {
      // Mock useFrameGraph with valid Cytoscape JSON
      // Verify CytoscapeComponent is rendered
    });

    it("re-fetches when step changes", () => {
      // Render with step=1, then update to step=2
      // Verify useFrameGraph was called with new tick
    });
  });
  ```

- `dashboard-ui/src/api/queries.test.ts` (extend): Graph query hooks
  ```typescript
  describe("useFrameGraph", () => {
    it("passes run, tick, and game to fetchFrameGraph", () => { ... });
    it("sets staleTime to Infinity (immutable data)", () => { ... });
    it("is disabled when run is empty", () => { ... });
  });
  ```

**Implementation**:

- `dashboard-ui/src/api/client.ts` (modify): Add fetch function
  ```typescript
  export async function fetchFrameGraph(
    run: string, tick: number, game?: number, depth?: number
  ): Promise<CytoscapeData> { ... }
  ```

- `dashboard-ui/src/api/queries.ts` (modify): Add query hook
  ```typescript
  export function useFrameGraph(run: string, tick: number, game?: number) { ... }
  ```

- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (new file): Main visualization component. Port from prototype:
  - Cytoscape stylesheet (node type colors/shapes, edge type colors/styles)
  - `buildElements()` function computing positioned Cytoscape elements from API data
  - `expandedNodes` state for click-to-expand
  - Manual hierarchical layout with `measureFrameWidth()` and level positioning
  - Frame nodes pinned in horizontal timeline row
  - Note: base/fallback node style MUST come before type-specific styles (Cytoscape applies in array order, not CSS specificity)

- `dashboard-ui/src/App.tsx` (modify): Add GraphVisualization to accordion
  ```typescript
  import { GraphVisualization } from "./components/panels/GraphVisualization";
  // Add Network icon import from lucide-react
  // Add Section with value="graph-visualization"
  ```

**Dependencies**:
- External: `cytoscape`, `react-cytoscapejs` (already installed from prototyping)
- Internal: Phase 2 (graph API frame endpoint), Phase 3 (live snapshot support)

**Verification**:
1. Run: `cd dashboard-ui && npx vitest run` -- all frontend tests pass
2. Run: `cd dashboard-ui && npx tsc --noEmit` -- no type errors
3. Start servers: `make run`
4. Open dashboard in browser: `npx servherd info roc-ui` for URL
5. Scroll to "Graph Visualization" section in accordion -- it should appear
6. Navigate to a run that has graph.json -- graph should render with blue Frame nodes
7. Click a Frame node -- it should expand showing Action (red triangle) and ObjectInstances (lime circles)
8. Click an ObjectInstance -- it should expand showing Object, FeatureGroup, RelationshipGroup
9. Pan and zoom should work (mouse wheel + drag)

---

### Phase 5: Frontend -- Detail Panel and Polish

**What this phase accomplishes**: Adds the right-side detail panel that appears when clicking a node or edge. Shows context-appropriate information: Frame intrinsics, ObjectInstance features/relationships, Object history, FeatureNode values. Also adds the action name mapping (from `/api/runs/{run}/action-map`), glyph character rendering, edge labels, and visual polish (collapse animation, selected node highlighting).

**Duration**: 2-3 days

**Tests to Write First**:

- `dashboard-ui/src/components/panels/GraphVisualization.test.tsx` (extend):
  ```typescript
  describe("Detail Panel", () => {
    it("shows frame details when a Frame node is clicked", () => {
      // Click Frame node -> detail panel shows tick, action badge, intrinsic bars
    });

    it("shows object instance details when ObjectInstance is clicked", () => {
      // Click ObjectInstance -> shows glyph, features table, relationships table
    });

    it("shows object details when Object is clicked", () => {
      // Click Object -> shows UUID, resolve count, first/last seen
    });

    it("shows feature details when FeatureNode is clicked", () => {
      // Click FeatureNode -> shows name and value
    });

    it("closes detail panel on background click", () => {
      // Click empty canvas -> detail panel closes
    });

    it("uses action map for action name display", () => {
      // Verify TakeAction node label shows action name, not just ID
    });
  });
  ```

**Implementation**:

- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (modify): Add detail panel
  - `selectedNode` state tracking clicked node type + data
  - Conditional detail panel (`<Paper>` sibling) with content per node type:
    - Frame: action Badge, intrinsic Progress bars (raw + normalized), object list
    - ObjectInstance: colored glyph header, features KVTable, relationships KVTable, Object UUID
    - Object: UUID, resolve count, last position, annotation list
    - FeatureGroup/RelationshipGroup: contained features as a table
    - FeatureNode: name, kind (PHYSICAL/RELATIONAL), value
  - Background click handler to close panel
  - Selected node visual highlight (thicker border)
  - Edge click handler showing edge type and endpoint labels

- `dashboard-ui/src/api/queries.ts` (modify if needed): Ensure `useActionMap` is available for action name lookup

- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (modify): Edge labels
  - All edges show their type name (NextFrame, Situated, ObservedAs, etc.)
  - Edge label styling: small font, matching edge color, positioned at midpoint

**Dependencies**:
- External: None
- Internal: Phase 4 (base GraphVisualization component), existing `useActionMap` hook

**Verification**:
1. Run: `cd dashboard-ui && npx vitest run` -- all tests pass
2. Open dashboard, navigate to a run with graph data
3. Click a Frame node -- detail panel appears on right with intrinsics as progress bars
4. Click an ObjectInstance -- detail panel shows glyph character, features table
5. Click canvas background -- detail panel closes
6. Verify TakeAction nodes show action names (e.g., "Action (West)") not raw IDs
7. Verify edge labels are visible on all connections
8. Click an Object node -- shows UUID, resolve count

---

### Phase 6: Object History Endpoint in Frontend

**What this phase accomplishes**: Integrates the object history API endpoint into the frontend. Adds an "Object History" view accessible from the detail panel (click an Object to see all frames where it appeared). This is an advanced query feature that differentiates the production implementation from the prototype.

**Duration**: 2-3 days

**Tests to Write First**:

- `dashboard-ui/src/components/panels/GraphVisualization.test.tsx` (extend):
  ```typescript
  describe("Object History", () => {
    it("shows 'View History' button when Object node is selected", () => { ... });

    it("fetches and displays object history graph on button click", () => {
      // Verify useObjectHistory hook is called with correct uuid
      // Verify graph switches to showing object's frame timeline
    });

    it("provides a 'Back to Frame' button to return to normal view", () => { ... });
  });
  ```

- `dashboard-ui/src/api/queries.test.ts` (extend):
  ```typescript
  describe("useObjectHistoryGraph", () => {
    it("fetches from /graph/object/{uuid}", () => { ... });
    it("is disabled when uuid is null", () => { ... });
  });
  ```

**Implementation**:

- `dashboard-ui/src/api/client.ts` (modify): Add fetch functions
  ```typescript
  export async function fetchObjectHistoryGraph(
    run: string, uuid: number
  ): Promise<CytoscapeData> { ... }
  ```

- `dashboard-ui/src/api/queries.ts` (modify): Add query hooks
  ```typescript
  export function useObjectHistoryGraph(run: string, uuid: number | null) { ... }
  ```

- `dashboard-ui/src/types/step-data.ts` (modify): Add types
  ```typescript
  export interface CytoscapeData {
    elements: { nodes: CytoscapeNode[]; edges: CytoscapeEdge[] };
    meta: { root_id: number | null; node_count: number; edge_count: number };
  }
  ```

- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (modify):
  - Object detail panel: "View History" button that switches the graph to object-centric view
  - Object history view: horizontal frame timeline with object instances highlighted

**Dependencies**:
- External: None
- Internal: Phase 2 (object history API endpoint), Phase 5 (detail panel)

**Verification**:
1. Run: `cd dashboard-ui && npx vitest run` -- all tests pass
2. Open dashboard, navigate to a run with graph data
3. Click a Frame, then click an ObjectInstance, then click the Object node
4. In the detail panel, click "View History" -- graph switches to show all frames for that object
5. Click "Back to Frame" -- returns to normal frame view

---

### Phase 7: Cleanup, Performance, and Integration Testing

**What this phase accomplishes**: Remove the prototype files (now superseded by the production implementation). Remove the `cytoscape-dagre` dependency (unused). Add integration and e2e tests that exercise the full pipeline: game run -> graph archive -> API -> frontend rendering. Optimize large graph rendering (virtualization, node count limits). Ensure the panel works correctly with live games, historical scrubbing, and run switching.

**Duration**: 2 days

**Tests to Write First**:

- `tests/integration/reporting/test_graph_integration.py`:
  ```python
  def test_graph_archive_round_trip():
      """Export graph, load via GraphService, extract subgraph, verify structure."""

  def test_graph_api_serves_archived_data(test_client):
      """Full stack: write graph.json, hit API endpoint, verify Cytoscape JSON."""

  def test_graph_api_live_snapshot_routing(test_client):
      """Push StepData with graph_snapshot, verify API returns it."""

  def test_large_graph_subgraph_extraction():
      """Graph with 1000+ nodes: subgraph extraction completes in <100ms."""
  ```

- `dashboard-ui/e2e/graph-visualization.spec.ts` (new file): Playwright e2e
  ```typescript
  test("graph panel renders nodes for a historical run", async ({ page }) => {
    // Navigate to a run with graph data
    // Open Graph Visualization accordion section
    // Wait for Cytoscape canvas to appear
    // Verify at least one node is visible
  });

  test("click-to-expand works", async ({ page }) => {
    // Click a Frame node
    // Verify child nodes appear (node count increases)
  });

  test("detail panel opens on node click", async ({ page }) => {
    // Click a node
    // Verify detail panel appears with node information
  });
  ```

**Implementation**:

- Delete `dashboard-ui/src/prototype/` directory (GraphPrototype.tsx, mock-graph-data.ts, graph-main.tsx)
- Delete `dashboard-ui/graph-prototype.html`
- `dashboard-ui/package.json` (modify): Remove `cytoscape-dagre` dependency
- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (modify): Performance guardrails
  - Warn when subgraph exceeds ~200 nodes (suggest reducing depth)
  - Debounce re-fetches during rapid step scrubbing (300ms)
  - Memoize Cytoscape stylesheet and layout config
- `roc/reporting/graph_service.py` (modify): Add node count limit parameter
  ```python
  def subgraph_from_frame(self, run_name, tick, depth=2, max_nodes=500): ...
  ```

**Dependencies**:
- External: None
- Internal: All previous phases

**Verification**:
1. Run: `make test` -- all Python tests pass
2. Run: `cd dashboard-ui && npx vitest run` -- all frontend tests pass
3. Run: `cd dashboard-ui && npx playwright test` -- e2e tests pass
4. Verify `dashboard-ui/src/prototype/` no longer exists
5. Verify `cytoscape-dagre` is not in package.json
6. Run: `make lint` -- no lint errors
7. Open dashboard, navigate through multiple runs -- graph panel works for each
8. Start a live game with `roc_emit_state_graph=true` -- graph updates in real time
9. Scrub rapidly through steps -- graph updates without errors or excessive API calls

## Common Utilities Needed

- **`_bfs_subgraph()` (Python, GraphService)**: BFS traversal on NetworkX DiGraph following both directions. Used by `subgraph_from_frame()`, `subgraph_from_node()`. Shared static method.
- **`_bfs_collect()` (Python, gymnasium.py)**: BFS traversal on live Node objects following src_edges/dst_edges. Used by `_collect_graph_snapshot()`. Cannot share with GraphService because it operates on different data structures (live Node vs NetworkX node).
- **`buildElements()` (TypeScript, GraphVisualization)**: Converts API Cytoscape JSON into positioned Cytoscape elements respecting expand state. Ported from prototype. This is the core layout engine.
- **`measureFrameWidth()` (TypeScript, GraphVisualization)**: Counts leaf-level columns per frame subtree to compute dynamic frame spacing. Ported from prototype.
- **Cytoscape stylesheet constant (TypeScript)**: Array of style rules for all node/edge types. Defined once, memoized, shared across normal view and object history view.

## External Libraries Assessment

- **cytoscape + react-cytoscapejs**: Already installed. Core visualization library. No alternatives needed.
- **cytoscape-dagre**: Already installed but UNUSED. Remove in Phase 7. Dagre's global optimization fights with pinned frame nodes -- manual positioning is strictly better for this use case.
- **networkx**: Already installed (Python). Used for graph archive format (node-link JSON), subgraph extraction, and Cytoscape JSON conversion. The right tool for this job.
- No new external libraries are needed for this implementation.

## Risk Mitigation

- **Large graph archives**: A long game (1000+ steps) could produce a graph.json in the tens of MB range. **Mitigation**: The subgraph extraction limits what the API returns. Add a `max_nodes` parameter to `_bfs_subgraph()` that stops early. The archive itself is write-once-read-rarely, so file size is acceptable.

- **Cytoscape.js performance with many nodes**: Rendering 200+ nodes with labels and animations can lag. **Mitigation**: Depth 2 produces 20-50 nodes (design doc). The expand-on-click model means users control complexity. Add a warning when node count exceeds ~200.

- **Serialization overhead in game loop**: `_collect_graph_snapshot()` runs every step when `emit_state_graph=True`. **Mitigation**: Config flag defaults to False. At depth 2 with 20-50 nodes, serialization is ~1-5ms (design doc). The game loop already spends more time on screen rendering. If it becomes an issue, the "demand-driven collection" future consideration from the design doc addresses it.

- **graph.json not written on abnormal exit**: If the game subprocess crashes, `_handle_game_over()` may not run. **Mitigation**: The live snapshot path covers running games. For historical analysis, users can re-run the game. A future improvement could flush periodically, but this is not needed for the initial implementation.

- **Cytoscape stylesheet ordering**: Cytoscape.js applies styles in array order (later wins), not by CSS specificity. **Mitigation**: Document this in code comments. The base/fallback style MUST be the first entry. This was a bug found and fixed during prototyping.

- **Breaking existing tests**: All modifications to existing files (config.py, gymnasium.py, run_store.py, api_server.py, data_store.py) are additive. **Mitigation**: New config flag defaults to False, new StepData field defaults to None, new API router is mounted alongside existing routes. Existing behavior is unchanged.
