import { fireEvent, render, screen } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import type { CytoscapeData } from "../../types/step-data";
import {
    buildElements,
    buildPropertyChangeRows,
    DetailPanel,
    findConnectedNodes,
    findTransformChildren,
    findTransformFrameTicks,
    getActionId,
    getDescendants,
    getFeatureNodeDisplay,
    getNodeType,
    GraphVisualization,
    lookupActionName,
    makeLabel,
    mergeGraphData,
    type SelectedElement,
} from "./GraphVisualization";

// Mock react-cytoscapejs -- Cytoscape.js cannot run in jsdom.
//
// The real react-cytoscapejs calls the `cy` prop callback on EVERY React
// update (not just on mount), so this mock does the same thing to catch
// listener-accumulation regressions in handleCyInit.
const capturedCyCallbacks: Array<(cy: unknown) => void> = [];

/** A node-collection-shaped stub that supports the methods our code path
 *  exercises (forEach, sort, length). Returning empty mimics an unmounted
 *  cytoscape instance, which is fine for our component-mount tests. */
function emptyCollection() {
    const arr: unknown[] = [];
    const collection = Object.assign(arr, {
        forEach: arr.forEach.bind(arr) as (
            cb: (n: unknown, i: number) => void,
        ) => void,
        sort: () => collection,
        map: arr.map.bind(arr) as <T>(cb: (n: unknown) => T) => T[],
    });
    return collection;
}

const mockCyInstance = {
    _listenerCount: 0,
    _listeners: new Map<string, unknown[]>(),
    on(event: string, ...rest: unknown[]) {
        this._listenerCount += 1;
        const list = this._listeners.get(event) ?? [];
        list.push(rest[rest.length - 1]);
        this._listeners.set(event, list);
    },
    fit() { /* no-op */ },
    nodes() { return emptyCollection(); },
    edges() { return emptyCollection(); },
    zoom() { return 1; },
    pan() { return { x: 0, y: 0 }; },
    layout() {
        return {
            one: () => undefined,
            run: () => undefined,
        };
    },
    getElementById() {
        return { length: 0 };
    },
};

function resetMockCy() {
    mockCyInstance._listenerCount = 0;
    mockCyInstance._listeners = new Map();
    capturedCyCallbacks.length = 0;
}

vi.mock("react-cytoscapejs", () => ({
    __esModule: true,
    default: function MockCytoscape(props: {
        elements?: unknown[];
        "data-testid"?: string;
        cy?: (cy: unknown) => void;
    }) {
        const count = Array.isArray(props.elements) ? props.elements.length : 0;
        // Simulate react-cytoscapejs's behaviour: call the cy prop on every
        // render, passing the same cy instance. Without the handleCyInit
        // idempotency guard, this accumulates listeners.
        if (props.cy) {
            capturedCyCallbacks.push(props.cy);
            props.cy(mockCyInstance);
        }
        return <div data-testid="cytoscape-component" data-element-count={count} />;
    },
}));

// Mock the query hooks
vi.mock("../../api/queries", () => ({
    useFrameGraph: vi.fn(),
    useActionMap: vi.fn(),
    useObjectHistoryGraph: vi.fn(),
}));

import { useActionMap, useFrameGraph, useObjectHistoryGraph } from "../../api/queries";

const mockUseFrameGraph = vi.mocked(useFrameGraph);
const mockUseActionMap = vi.mocked(useActionMap);
const mockUseObjectHistoryGraph = vi.mocked(useObjectHistoryGraph);

function Wrapper({ children }: Readonly<{ children: ReactNode }>) {
    return <MantineProvider>{children}</MantineProvider>;
}

function makeCytoscapeData(
    overrides: Partial<CytoscapeData> = {},
): CytoscapeData {
    return {
        elements: {
            nodes: [
                { data: { id: "1", labels: "Frame", tick: 42 } },
                { data: { id: "2", labels: "ObjectInstance" } },
                { data: { id: "3", labels: "TakeAction", action_id: 19 } },
            ],
            edges: [
                { data: { id: "1-2", source: "1", target: "2", type: "SituatedObjectInstance" } },
                { data: { id: "1-3", source: "1", target: "3", type: "FrameAttribute" } },
            ],
        },
        meta: { root_id: 1, node_count: 3, edge_count: 2 },
        ...overrides,
    };
}

describe("GraphVisualization", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockUseActionMap.mockReturnValue({
            data: undefined,
            isLoading: false,
        } as ReturnType<typeof useActionMap>);
        mockUseObjectHistoryGraph.mockReturnValue({
            data: undefined,
            isLoading: false,
        } as ReturnType<typeof useObjectHistoryGraph>);
    });

    it("renders loading state when data is undefined", () => {
        mockUseFrameGraph.mockReturnValue({
            data: undefined,
            isLoading: true,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={1} />, { wrapper: Wrapper });
        expect(screen.getByText("Loading graph data...")).toBeInTheDocument();
    });

    it("renders 'No graph data' when API returns empty graph", () => {
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData({
                elements: { nodes: [], edges: [] },
                meta: { root_id: null, node_count: 0, edge_count: 0 },
            }),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={1} />, { wrapper: Wrapper });
        expect(screen.getByText("No graph data")).toBeInTheDocument();
    });

    it("renders 'No graph data' when data is undefined and not loading", () => {
        mockUseFrameGraph.mockReturnValue({
            data: undefined,
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={1} />, { wrapper: Wrapper });
        expect(screen.getByText("No graph data")).toBeInTheDocument();
    });

    it("renders Cytoscape component when data is available", () => {
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={1} />, { wrapper: Wrapper });
        expect(screen.getByTestId("cytoscape-component")).toBeInTheDocument();
    });

    it("re-fetches immediately when step prop changes (no internal debounce)", () => {
        // Step debouncing is handled by the parent (App.tsx) via
        // useDebouncedValue. GraphVisualization uses the step prop directly.
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        const { rerender } = render(
            <GraphVisualization run="test-run" step={1} />,
            { wrapper: Wrapper },
        );

        expect(mockUseFrameGraph).toHaveBeenCalledWith("test-run", 1, undefined, 1);

        rerender(
            <Wrapper>
                <GraphVisualization run="test-run" step={2} />
            </Wrapper>,
        );

        // No debounce: hook immediately called with step=2
        expect(mockUseFrameGraph).toHaveBeenCalledWith("test-run", 2, undefined, 1);
    });

    it("uses new step immediately on run change -- no stale cross-run requests", () => {
        // Bug regression: switching from a 573-step run at step 507 to a
        // 13-step run must NOT fire a request for step 507 on the new run.
        // With no internal debounce, the step prop arrives already correct
        // from the parent's useDebouncedValue(step, 200, run) reset.
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        const { rerender } = render(
            <GraphVisualization run="long-run" step={507} />,
            { wrapper: Wrapper },
        );

        expect(mockUseFrameGraph).toHaveBeenCalledWith("long-run", 507, undefined, 1);

        // Switch to a short run with step reset to 1
        rerender(
            <Wrapper>
                <GraphVisualization run="short-run" step={1} />
            </Wrapper>,
        );

        // The hook must NEVER be called with step=507 on the new run.
        expect(mockUseFrameGraph).not.toHaveBeenCalledWith(
            "short-run", 507, expect.anything(), expect.anything(),
        );
        // It MUST have been called with the correct step on the new run
        expect(mockUseFrameGraph).toHaveBeenCalledWith("short-run", 1, undefined, 1);
    });

    it("passes game parameter to useFrameGraph", () => {
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={5} game={2} />, { wrapper: Wrapper });
        // Initial render: debounced step starts at the prop value
        expect(mockUseFrameGraph).toHaveBeenCalledWith("test-run", 5, 2, 1);
    });

    it("shows node count warning when more than 200 nodes", () => {
        // Build data with 201+ frame nodes (each frame becomes a positioned node)
        const manyFrames = Array.from({ length: 210 }, (_, i) => ({
            data: { id: String(i + 1), labels: "Frame", tick: i + 1 },
        }));
        const manyEdges = Array.from({ length: 209 }, (_, i) => ({
            data: { id: `e${i}`, source: String(i + 1), target: String(i + 2), type: "NextFrame" },
        }));

        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData({
                elements: { nodes: manyFrames, edges: manyEdges },
                meta: { root_id: 1, node_count: 210, edge_count: 209 },
            }),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        render(<GraphVisualization run="test-run" step={1} />, { wrapper: Wrapper });
        expect(screen.getByText(/Large graph/)).toBeInTheDocument();
    });

    it("handleCyInit is idempotent across re-renders (no listener accumulation)", () => {
        // Regression test for the click-flicker bug.
        //
        // react-cytoscapejs calls the `cy` prop callback on every React
        // update, not just on mount. Without the idempotency guard in
        // handleCyInit, every re-render attaches another copy of every
        // event listener (zoom, pan, tap node, tap edge, background tap).
        // After N re-renders a single click fires the node-tap handler
        // N times, starting N concurrent fetches whose toggleExpand calls
        // oscillate the node between expanded and collapsed for seconds.
        //
        // The fix bails out of handleCyInit when the cy instance it's
        // being called with is the same one already set up. This test
        // verifies listener counts stay at 1-per-event regardless of how
        // many times the component re-renders.
        resetMockCy();
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        // NOTE: do NOT use `wrapper: Wrapper` with rerender() passing <Wrapper>
        // explicitly -- that causes React to see a different tree shape on
        // rerender and REMOUNT the component, which creates a fresh cyRef
        // and invalidates the idempotency check. Instead wrap manually once
        // and pass the same tree structure to rerender.
        const { rerender } = render(
            <Wrapper>
                <GraphVisualization run="test-run" step={1} />
            </Wrapper>,
        );

        // Force several re-renders with fresh props. Each one triggers
        // react-cytoscapejs's componentDidUpdate path which calls
        // props.cy(cy) again (captured by the mock). Since the tree shape
        // stays the same, React reuses the same component instance and
        // cyRef persists, so handleCyInit should skip from the second call
        // onward.
        for (let i = 0; i < 5; i++) {
            rerender(
                <Wrapper>
                    <GraphVisualization run="test-run" step={1} game={i} />
                </Wrapper>,
            );
        }

        // The mock was called multiple times (mount + updates), but the
        // real handler inside handleCyInit should have skipped the second+
        // invocations via the `cyRef.current === cy` guard.
        expect(capturedCyCallbacks.length).toBeGreaterThan(1);

        // Exactly one listener per event type despite multiple cy prop calls.
        expect(mockCyInstance._listeners.get("zoom pan")?.length).toBe(1);
        expect(mockCyInstance._listeners.get("tap")?.length).toBe(3); // node, edge, background
    });
});

describe("getNodeType", () => {
    it("maps Frame labels", () => {
        expect(getNodeType({ labels: "Frame" })).toBe("frame");
    });

    it("maps ObjectInstance labels (before Object)", () => {
        expect(getNodeType({ labels: "ObjectInstance" })).toBe("object-instance");
    });

    it("maps Object labels", () => {
        expect(getNodeType({ labels: "Object" })).toBe("object");
    });

    it("maps FeatureGroup labels", () => {
        expect(getNodeType({ labels: "FeatureGroup" })).toBe("feature-group");
    });

    it("maps FeatureNode labels", () => {
        expect(getNodeType({ labels: "FeatureNode" })).toBe("feature-node");
    });

    it("maps TakeAction labels", () => {
        expect(getNodeType({ labels: "TakeAction" })).toBe("action");
    });

    it("maps IntrinsicNode labels", () => {
        expect(getNodeType({ labels: "IntrinsicNode" })).toBe("intrinsic");
    });

    it("maps RelationshipGroup labels", () => {
        expect(getNodeType({ labels: "RelationshipGroup" })).toBe("relationship-group");
    });

    it("maps bare Transform labels (graph archive format)", () => {
        expect(getNodeType({ labels: "Transform" })).toBe("transform");
    });

    it("maps ObjectTransform labels", () => {
        expect(getNodeType({ labels: "ObjectTransform" })).toBe("transform");
    });

    it("maps ObjectTransform,Transform compound labels", () => {
        expect(getNodeType({ labels: "ObjectTransform, Transform" })).toBe("transform");
    });

    it("maps IntrinsicTransform labels", () => {
        expect(getNodeType({ labels: "IntrinsicTransform" })).toBe("transform");
    });

    // Regression: PropertyTransformNode contains the substring "Transform",
    // and the old getNodeType used substring matching that put it in the
    // "transform" bucket. Clicking it then opened TransformDetail which
    // showed only "kind: Transform" with nothing else, because none of the
    // expected fields exist on a PropertyTransformNode. The fix is to
    // check the more-specific PropertyTransformNode label first and route
    // it to its own detail panel.
    it("maps PropertyTransformNode to its own type, not 'transform'", () => {
        expect(getNodeType({ labels: "PropertyTransformNode" })).toBe(
            "property-transform",
        );
    });

    it("returns unknown for unrecognized labels", () => {
        expect(getNodeType({ labels: "SomethingElse" })).toBe("unknown");
    });

    it("handles missing labels", () => {
        expect(getNodeType({})).toBe("unknown");
    });
});

describe("buildElements", () => {
    it("returns empty array for empty data", () => {
        const data = makeCytoscapeData({
            elements: { nodes: [], edges: [] },
            meta: { root_id: null, node_count: 0, edge_count: 0 },
        });
        expect(buildElements(data, new Set())).toEqual([]);
    });

    it("emits a frame element with the correct type and label", () => {
        // buildElements no longer assigns positions -- the layout is computed
        // by cytoscape-fcose at render time. We just check the element is
        // present with the right metadata.
        const data = makeCytoscapeData({
            elements: {
                nodes: [{ data: { id: "1", labels: "Frame", tick: 1 } }],
                edges: [],
            },
            meta: { root_id: 1, node_count: 1, edge_count: 0 },
        });

        const els = buildElements(data, new Set());
        expect(els).toHaveLength(1);
        const frame = els[0]!;
        expect(frame.group).toBe("nodes");
        expect(frame.data._type).toBe("frame");
        expect(frame.data._label).toBe("Frame 1");
    });

    it("does not place children when frame is collapsed", () => {
        const data = makeCytoscapeData();
        const els = buildElements(data, new Set());
        // Only frame nodes, no children (frames are collapsed)
        const nodes = els.filter((e) => e.group === "nodes");
        expect(nodes).toHaveLength(1);
        expect(nodes[0]!.data._type).toBe("frame");
    });

    it("includes children when frame is expanded", () => {
        const data = makeCytoscapeData();
        const expanded = new Set(["1"]);
        const els = buildElements(data, expanded);

        // Should have frame + 2 children
        const nodes = els.filter((e) => e.group === "nodes");
        expect(nodes.length).toBeGreaterThanOrEqual(3);

        // Children types should be present alongside the frame
        const childTypes = nodes
            .filter((n) => n.data._type !== "frame")
            .map((n) => n.data._type);
        expect(childTypes.length).toBeGreaterThan(0);
    });

    it("includes edges between placed nodes", () => {
        const data = makeCytoscapeData();
        const expanded = new Set(["1"]);
        const els = buildElements(data, expanded);

        const edges = els.filter((e) => e.group === "edges");
        expect(edges.length).toBeGreaterThanOrEqual(2);
    });

    it("adds NextFrame edges between adjacent frames", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "Frame", tick: 2 } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "NextFrame" } },
                ],
            },
            meta: { root_id: 1, node_count: 2, edge_count: 1 },
        });

        const els = buildElements(data, new Set());
        const edges = els.filter((e) => e.group === "edges");
        expect(edges).toHaveLength(1);
        expect(edges[0]!.data._type).toBe("next-frame");
    });

    it("emits frames sorted by tick", () => {
        // Frame x positions are assigned at layout time by cytoscape-fcose
        // (via fixedNodeConstraint based on the iteration order). We assert
        // here that buildElements outputs the frames in tick order so the
        // downstream layout pin sequence ends up correct.
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "b", labels: "Frame", tick: 5 } },
                    { data: { id: "a", labels: "Frame", tick: 3 } },
                ],
                edges: [],
            },
            meta: { root_id: null, node_count: 2, edge_count: 0 },
        });

        const els = buildElements(data, new Set());
        const nodes = els.filter((e) => e.group === "nodes");
        expect(nodes[0]!.data.id).toBe("a"); // tick 3 first
        expect(nodes[1]!.data.id).toBe("b"); // tick 5 second
    });

    it("sets _type and _label on all elements", () => {
        const data = makeCytoscapeData();
        const els = buildElements(data, new Set(["1"]));

        for (const el of els) {
            expect(el.data._type).toBeDefined();
            expect(el.data._label).toBeDefined();
        }
    });

    it("uses action map to label TakeAction nodes with action name", () => {
        const data = makeCytoscapeData();
        const actionMap = [{ action_id: 19, action_name: "West", action_key: "h" }];
        const els = buildElements(data, new Set(["1"]), actionMap);

        const actionNode = els.find((e) => e.data._type === "action");
        expect(actionNode).toBeDefined();
        expect(actionNode!.data._label).toBe("Action (West)");
    });

    it("falls back to action ID when no action map", () => {
        const data = makeCytoscapeData();
        const els = buildElements(data, new Set(["1"]));

        const actionNode = els.find((e) => e.data._type === "action");
        expect(actionNode).toBeDefined();
        expect(actionNode!.data._label).toBe("Action 19");
    });

    it("filters out IntrinsicNode children from expanded frames", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "ObjectInstance" } },
                    { data: { id: "3", labels: "TakeAction", action_id: 19 } },
                    { data: { id: "4", labels: "IntrinsicNode", name: "hp" } },
                    { data: { id: "5", labels: "IntrinsicNode", name: "hunger" } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "SituatedObjectInstance" } },
                    { data: { id: "1-3", source: "1", target: "3", type: "FrameAttribute" } },
                    { data: { id: "1-4", source: "1", target: "4", type: "FrameAttribute" } },
                    { data: { id: "1-5", source: "1", target: "5", type: "FrameAttribute" } },
                ],
            },
            meta: { root_id: 1, node_count: 5, edge_count: 4 },
        });

        const els = buildElements(data, new Set(["1"]));
        const nodeTypes = els
            .filter((e) => e.group === "nodes")
            .map((e) => e.data._type as string);

        // Frame + ObjInst + Action should be present
        expect(nodeTypes).toContain("frame");
        expect(nodeTypes).toContain("object-instance");
        expect(nodeTypes).toContain("action");
        // IntrinsicNodes should NOT be in the graph (default hides intrinsic)
        expect(nodeTypes).not.toContain("intrinsic");
    });

    it("shows IntrinsicNode when hiddenTypes does not include it", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "IntrinsicNode", name: "hp" } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "FrameAttribute" } },
                ],
            },
            meta: { root_id: 1, node_count: 2, edge_count: 1 },
        });

        const els = buildElements(data, new Set(["1"]), undefined, new Set());
        const nodeTypes = els
            .filter((e) => e.group === "nodes")
            .map((e) => e.data._type as string);

        expect(nodeTypes).toContain("frame");
        expect(nodeTypes).toContain("intrinsic");
    });

    it("hides arbitrary node types when listed in hiddenTypes", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "ObjectInstance" } },
                    { data: { id: "3", labels: "TakeAction", action_id: 19 } },
                    { data: { id: "4", labels: "IntrinsicNode", name: "hp" } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "SituatedObjectInstance" } },
                    { data: { id: "1-3", source: "1", target: "3", type: "FrameAttribute" } },
                    { data: { id: "1-4", source: "1", target: "4", type: "FrameAttribute" } },
                ],
            },
            meta: { root_id: 1, node_count: 4, edge_count: 3 },
        });

        // Hide object-instance AND action, but keep intrinsic visible.
        const els = buildElements(
            data,
            new Set(["1"]),
            undefined,
            new Set(["object-instance", "action"]),
        );
        const nodeTypes = els
            .filter((e) => e.group === "nodes")
            .map((e) => e.data._type as string);

        expect(nodeTypes).toContain("frame");
        expect(nodeTypes).toContain("intrinsic");
        expect(nodeTypes).not.toContain("object-instance");
        expect(nodeTypes).not.toContain("action");
    });

    it("empty hiddenTypes set shows every node type", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "ObjectInstance" } },
                    { data: { id: "3", labels: "TakeAction", action_id: 19 } },
                    { data: { id: "4", labels: "IntrinsicNode", name: "hp" } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "SituatedObjectInstance" } },
                    { data: { id: "1-3", source: "1", target: "3", type: "FrameAttribute" } },
                    { data: { id: "1-4", source: "1", target: "4", type: "FrameAttribute" } },
                ],
            },
            meta: { root_id: 1, node_count: 4, edge_count: 3 },
        });

        const els = buildElements(data, new Set(["1"]), undefined, new Set());
        const nodeTypes = els
            .filter((e) => e.group === "nodes")
            .map((e) => e.data._type as string);

        expect(nodeTypes).toContain("frame");
        expect(nodeTypes).toContain("object-instance");
        expect(nodeTypes).toContain("action");
        expect(nodeTypes).toContain("intrinsic");
    });

    it("frame nodes are never removed by hiddenTypes", () => {
        // Even if a user somehow passes "frame" in hiddenTypes, frames must
        // still render -- they are the timeline backbone of the graph.
        const data = makeCytoscapeData({
            elements: {
                nodes: [{ data: { id: "1", labels: "Frame", tick: 1 } }],
                edges: [],
            },
            meta: { root_id: 1, node_count: 1, edge_count: 0 },
        });

        const els = buildElements(data, new Set(), undefined, new Set(["frame"]));
        const nodeTypes = els
            .filter((e) => e.group === "nodes")
            .map((e) => e.data._type as string);

        expect(nodeTypes).toContain("frame");
    });

    // Regression: a parent Transform between Frame N and Frame N+1 has two
    // outgoing Change edges (one to each frame). When the user expands
    // Frame N, the walk places the Transform and the (Frame N -> Transform)
    // edge but used to skip the (Transform -> Frame N+1) edge -- because
    // the walk only adds edges along the parent->child direction it took.
    // Even though Frame N+1 is on the canvas (it's the timeline backbone),
    // the cross-link wasn't drawn. The fix is a final pass that draws every
    // edge whose endpoints are both already placed.
    it("draws cross-link edges between visible nodes that aren't on the walk path", () => {
        const data: CytoscapeData = {
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "Frame", tick: 2 } },
                    { data: { id: "t", labels: "Transform" } },
                ],
                edges: [
                    // Frame 1 has the outgoing Change edge to Transform.
                    { data: { id: "e1", source: "1", target: "t", type: "Change" } },
                    // Transform's other Change edge points at Frame 2.
                    { data: { id: "e2", source: "t", target: "2", type: "Change" } },
                    { data: { id: "e3", source: "1", target: "2", type: "NextFrame" } },
                ],
            },
            meta: { root_id: 1, node_count: 3, edge_count: 3 },
        };

        // Expand frame 1 only -- the walk goes 1 -> Transform via Change.
        const els = buildElements(data, new Set(["1"]));
        const edges = els.filter((e) => e.group === "edges");

        const edgeIds = new Set(edges.map((e) => String(e.data.id)));
        // The (Frame 1 -> Transform) edge is added by the walk itself.
        expect(edgeIds.has("e1")).toBe(true);
        // The (Transform -> Frame 2) edge is the cross-link -- not on the
        // walk path, but both endpoints are placed (Transform via the walk,
        // Frame 2 as part of the timeline backbone). The final pass must
        // draw it.
        expect(edgeIds.has("e2")).toBe(true);
        // And the NextFrame edge between the two frames.
        expect(edgeIds.has("e3")).toBe(true);
    });

    // Regression sanity: the cross-link pass must NOT pull in edges between
    // nodes that aren't on the canvas. If a node was filtered out (e.g.
    // hidden by visible-types), its incident edges should stay hidden too.
    it("does not draw edges to nodes that weren't placed", () => {
        const data: CytoscapeData = {
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "intr", labels: "IntrinsicNode", name: "hp" } },
                ],
                edges: [
                    { data: { id: "e1", source: "1", target: "intr", type: "FrameAttribute" } },
                ],
            },
            meta: { root_id: 1, node_count: 2, edge_count: 1 },
        };

        // Frame 1 not expanded; default DEFAULT_HIDDEN_TYPES hides intrinsic.
        const els = buildElements(data, new Set());
        const placedNodeIds = new Set(
            els.filter((e) => e.group === "nodes").map((e) => String(e.data.id)),
        );
        expect(placedNodeIds.has("intr")).toBe(false);
        const edges = els.filter((e) => e.group === "edges");
        // The edge to the hidden intrinsic must not be drawn.
        expect(edges.some((e) => String(e.data.id) === "e1")).toBe(false);
    });
});

/** Rich test data with all node types connected. */
function makeRichCytoscapeData(): CytoscapeData {
    return {
        elements: {
            nodes: [
                { data: { id: "1", labels: "Frame", tick: 42 } },
                { data: { id: "2", labels: "ObjectInstance", x: 5, y: 10, glyph: "@" } },
                { data: { id: "3", labels: "TakeAction", action_id: 19 } },
                { data: { id: "4", labels: "IntrinsicNode", name: "hp", raw_value: 15, normalized_value: 0.75 } },
                { data: { id: "5", labels: "IntrinsicNode", name: "hunger", raw_value: 900, normalized_value: 0.45 } },
                { data: { id: "6", labels: "Object", uuid: "12345", resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 } },
                { data: { id: "7", labels: "FeatureGroup" } },
                { data: { id: "8", labels: "FeatureNode", name: "glyph", value: "@", kind: "PHYSICAL" } },
                { data: { id: "9", labels: "RelationshipGroup" } },
                { data: { id: "10", labels: "FeatureNode", name: "distance", value: "3", kind: "RELATIONAL" } },
            ],
            edges: [
                { data: { id: "e1", source: "1", target: "2", type: "SituatedObjectInstance" } },
                { data: { id: "e2", source: "1", target: "3", type: "FrameAttribute" } },
                { data: { id: "e3", source: "1", target: "4", type: "FrameAttribute" } },
                { data: { id: "e4", source: "1", target: "5", type: "FrameAttribute" } },
                { data: { id: "e5", source: "2", target: "6", type: "ObservedAs" } },
                { data: { id: "e6", source: "2", target: "7", type: "Features" } },
                { data: { id: "e7", source: "7", target: "8", type: "Detail" } },
                { data: { id: "e8", source: "2", target: "9", type: "Relationships" } },
                { data: { id: "e9", source: "9", target: "10", type: "Detail" } },
            ],
        },
        meta: { root_id: 1, node_count: 10, edge_count: 9 },
    };
}

const TEST_ACTION_MAP = [
    { action_id: 19, action_name: "West", action_key: "h" },
];

describe("Detail Panel", () => {
    const richData = makeRichCytoscapeData();

    it("shows frame details when a Frame node is clicked", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "frame",
            data: { id: "1", labels: "Frame", tick: 42 },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("Frame 42")).toBeInTheDocument();
        expect(screen.getByText("West")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("hunger")).toBeInTheDocument();
    });

    it("shows object instance details when ObjectInstance is clicked", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "object-instance",
            data: { id: "2", labels: "ObjectInstance", x: 5, y: 10, glyph: "@" },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        // Glyph appears in header and features table -- check at least one
        expect(screen.getAllByText("@").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText(/\(5, 10\)/)).toBeInTheDocument();
    });

    // Regression: ObjectInstance detail panel used to show only position
    // and feature counts -- the parent Object's identity (human_name and
    // uuid) was unreachable from the panel even though the connected
    // Object node was already loaded. The fix walks the ObservedAs edge
    // to the Object and surfaces its name + uuid at the top of the panel.
    it("ObjectInstance panel surfaces parent Object's human_name and uuid when both are loaded", () => {
        const apiData: CytoscapeData = {
            elements: {
                nodes: [
                    { data: { id: "oi", labels: "ObjectInstance", x: 23, y: 9, glyph: "@", object_uuid: "6853809301722933453" } },
                    {
                        data: {
                            id: "obj",
                            labels: "Object",
                            uuid: "6853809301722933453",
                            human_name: "rancorous-fey-devy",
                            resolve_count: 189,
                        },
                    },
                ],
                edges: [
                    { data: { id: "e", source: "oi", target: "obj", type: "ObservedAs" } },
                ],
            },
            meta: { root_id: 0, node_count: 2, edge_count: 1 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "object-instance",
            data: apiData.elements.nodes[0]!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={apiData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Parent Object identity rows. The label words may appear elsewhere
        // in the surrounding panel chrome (e.g. another "object" word in
        // a header), so accept >=1 occurrences.
        expect(screen.getAllByText("object").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("rancorous-fey-devy")).toBeInTheDocument();
        expect(screen.getAllByText("uuid").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("6853809301722933453").length).toBeGreaterThanOrEqual(1);
    });

    it("ObjectInstance panel falls back to its own object_uuid / human_name when Object isn't loaded", () => {
        // Depth-1 fetch from a Frame fetches the ObjectInstance but not the
        // parent Object. The instance still carries object_uuid and
        // human_name (the server adds human_name to every node carrying a
        // uuid), so the panel should fall back to those copies rather than
        // showing nothing.
        const apiData: CytoscapeData = {
            elements: {
                nodes: [{
                    data: {
                        id: "oi",
                        labels: "ObjectInstance",
                        x: 23,
                        y: 9,
                        object_uuid: "6853809301722933453",
                        human_name: "rancorous-fey-devy",
                    },
                }],
                edges: [],
            },
            meta: { root_id: 0, node_count: 1, edge_count: 0 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "object-instance",
            data: apiData.elements.nodes[0]!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={apiData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("rancorous-fey-devy")).toBeInTheDocument();
        expect(screen.getByText("6853809301722933453")).toBeInTheDocument();
    });

    it("shows relationships table in ObjectInstance detail", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "object-instance",
            data: { id: "2", labels: "ObjectInstance", x: 5, y: 10, glyph: "@" },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("Relationships")).toBeInTheDocument();
        expect(screen.getByText("distance")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("shows object details when Object is clicked", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: { id: "6", labels: "Object", uuid: "12345", resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("12345")).toBeInTheDocument();
        expect(screen.getByText("7")).toBeInTheDocument();
    });

    it("shows feature details when FeatureNode is clicked", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "feature-node",
            data: { id: "8", labels: "FeatureNode", name: "glyph", value: "@", kind: "PHYSICAL" },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("glyph")).toBeInTheDocument();
        expect(screen.getByText("PHYSICAL")).toBeInTheDocument();
    });

    it("closes detail panel on background click", () => {
        // Background click in production sets selectedElement to null via cy.on("tap").
        // In jsdom we test the close button which calls the same onClose callback.
        const onClose = vi.fn();
        const selected: SelectedElement = {
            kind: "node",
            type: "frame",
            data: { id: "1", labels: "Frame", tick: 42 },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={onClose} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        fireEvent.click(screen.getByLabelText("Close"));
        expect(onClose).toHaveBeenCalledOnce();
    });

    it("uses action map for action name display", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "frame",
            data: { id: "1", labels: "Frame", tick: 42 },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        // Action badge shows mapped name, not raw ID
        expect(screen.getByText("West")).toBeInTheDocument();
    });

    it("shows edge details when an edge is clicked", () => {
        const selected: SelectedElement = {
            kind: "edge",
            type: "situated",
            data: { id: "e1", source: "1", target: "2", type: "SituatedObjectInstance" },
        };
        render(
            <DetailPanel selected={selected} apiData={richData} actionMap={TEST_ACTION_MAP} onClose={vi.fn()} onViewHistory={vi.fn()} />,
            { wrapper: Wrapper },
        );
        // Edge type appears in both title and Type field
        expect(screen.getAllByText("SituatedObjectInstance").length).toBeGreaterThanOrEqual(1);
        // Source label should be present
        expect(screen.getByText(/Frame 42/)).toBeInTheDocument();
    });
});

// ---------------------------------------------------------------------------
// Transform detail tests
//
// Regression: clicking on a Transform / ObjectTransform node used to show
// only summary counts ("discrete changes: 1, continuous changes: 2") with no
// indication of WHAT actually changed. The user couldn't tell what the
// transform represented. The detail panel must walk TransformDetail edges
// to its PropertyTransformNode children and surface the per-property changes.
// ---------------------------------------------------------------------------

/** Build a CytoscapeData containing a parent Transform, an ObjectTransform
 *  child of the parent, and PropertyTransformNode children of the
 *  ObjectTransform. Mirrors the real graph shape produced by Transformer
 *  for an object that moved diagonally and changed motion direction. */
function makeObjectTransformGraph(): CytoscapeData {
    const nodes = [
        { data: { id: "frame-prev", labels: "Frame", tick: 41 } },
        { data: { id: "frame-cur", labels: "Frame", tick: 42 } },
        { data: { id: "parent", labels: "Transform" } },
        {
            data: {
                id: "ot",
                labels: "ObjectTransform, Transform",
                // 63-bit ROC UUID as a string. A number literal here would
                // be silently truncated to ...000 by the JS parser, which
                // is exactly the precision-loss bug this code is fixing.
                object_uuid: "6853809301722933453",
                num_discrete_changes: 1,
                num_continuous_changes: 3,
            },
        },
        {
            data: {
                id: "ptn-x",
                labels: "PropertyTransformNode",
                property_name: "x",
                change_type: "continuous",
                old_value: null,
                new_value: null,
                delta: -1,
            },
        },
        {
            data: {
                id: "ptn-y",
                labels: "PropertyTransformNode",
                property_name: "y",
                change_type: "continuous",
                old_value: null,
                new_value: null,
                delta: 0,
            },
        },
        {
            data: {
                id: "ptn-distance",
                labels: "PropertyTransformNode",
                property_name: "distance",
                change_type: "continuous",
                old_value: 1,
                new_value: 6,
                delta: 5,
            },
        },
        {
            data: {
                id: "ptn-motion",
                labels: "PropertyTransformNode",
                property_name: "motion_direction",
                change_type: "discrete",
                old_value: "UP_LEFT",
                new_value: "LEFT",
                delta: null,
            },
        },
    ];
    const edges = [
        { data: { id: "e1", source: "frame-prev", target: "parent", type: "Change" } },
        { data: { id: "e2", source: "parent", target: "frame-cur", type: "Change" } },
        { data: { id: "e3", source: "parent", target: "ot", type: "Change" } },
        { data: { id: "e4", source: "ot", target: "ptn-x", type: "TransformDetail" } },
        { data: { id: "e5", source: "ot", target: "ptn-y", type: "TransformDetail" } },
        { data: { id: "e6", source: "ot", target: "ptn-distance", type: "TransformDetail" } },
        { data: { id: "e7", source: "ot", target: "ptn-motion", type: "TransformDetail" } },
    ];
    return {
        elements: { nodes, edges },
        meta: { root_id: 0, node_count: nodes.length, edge_count: edges.length },
    };
}

describe("findTransformFrameTicks", () => {
    it("returns src and dst frame ticks via Change edges", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        // The bare parent Transform is the one connected to both frames.
        const ticks = findTransformFrameTicks("parent", nodeMap, data.elements.edges);
        expect(ticks).toEqual({ srcFrameTick: 41, dstFrameTick: 42 });
    });

    it("returns null on each side when frame edges aren't loaded", () => {
        const ticks = findTransformFrameTicks(
            "ot",
            new Map(),
            [],
        );
        expect(ticks).toEqual({ srcFrameTick: null, dstFrameTick: null });
    });
});

describe("buildPropertyChangeRows", () => {
    it("walks TransformDetail edges to PropertyTransformNode children", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const rows = buildPropertyChangeRows("ot", nodeMap, data.elements.edges);

        expect(rows).toHaveLength(4);
        // Stable sort: continuous (x, y, others) before discrete
        expect(rows.map((r) => r.propertyName)).toEqual([
            "x", "y", "distance", "motion_direction",
        ]);
    });

    it("formats position deltas using only the delta field", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const rows = buildPropertyChangeRows("ot", nodeMap, data.elements.edges);
        const x = rows.find((r) => r.propertyName === "x")!;
        // x has only delta=-1; old_value/new_value are null. Output is just
        // the signed delta, not "null -> null".
        expect(x.summary).toBe("-1");

        const y = rows.find((r) => r.propertyName === "y")!;
        // Zero delta has no sign prefix -- "+0" would lie about direction.
        expect(y.summary).toBe("0");
    });

    it("formats discrete changes as 'old -> new'", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const rows = buildPropertyChangeRows("ot", nodeMap, data.elements.edges);
        const motion = rows.find((r) => r.propertyName === "motion_direction")!;
        expect(motion.summary).toBe("UP_LEFT -> LEFT");
        expect(motion.changeType).toBe("discrete");
    });

    it("formats distance changes with both endpoints AND the delta", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const rows = buildPropertyChangeRows("ot", nodeMap, data.elements.edges);
        const dist = rows.find((r) => r.propertyName === "distance")!;
        expect(dist.summary).toBe("1 -> 6 (+5)");
    });

    it("returns an empty list when the transform has no children loaded", () => {
        const data = makeObjectTransformGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        // The bare parent Transform has Change edges (not TransformDetail).
        const rows = buildPropertyChangeRows("parent", nodeMap, data.elements.edges);
        expect(rows).toEqual([]);
    });
});

describe("Transform Detail panel", () => {
    const data = makeObjectTransformGraph();

    it("shows the actual property changes for an ObjectTransform", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: data.elements.nodes.find((n) => n.data.id === "ot")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={data}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );

        // Header row identifies the transform kind. The label appears in
        // both the panel title and the "kind" row, so accept >=1.
        expect(screen.getAllByText("ObjectTransform").length).toBeGreaterThanOrEqual(1);
        // The per-property change table is the new behaviour: each property
        // appears with its actual change, not just a count.
        expect(screen.getByText("Property changes")).toBeInTheDocument();
        expect(screen.getByText("x")).toBeInTheDocument();
        expect(screen.getByText("y")).toBeInTheDocument();
        expect(screen.getByText("distance")).toBeInTheDocument();
        expect(screen.getByText("motion_direction")).toBeInTheDocument();
        // The actual change values (not just counts) must be visible.
        expect(screen.getByText("-1")).toBeInTheDocument();
        // y delta is 0 -- formatDelta omits the sign for zero.
        expect(screen.getByText("0")).toBeInTheDocument();
        expect(screen.getByText("1 -> 6 (+5)")).toBeInTheDocument();
        expect(screen.getByText("UP_LEFT -> LEFT")).toBeInTheDocument();
    });

    it("hides the discrete/continuous count rows once children are loaded", () => {
        // Regression: when the per-property table is shown, the redundant
        // "discrete changes: N" and "continuous changes: N" rows are noise
        // and used to confuse users. They only appear when children are
        // unfetched (depth=1 frame load).
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: data.elements.nodes.find((n) => n.data.id === "ot")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={data}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.queryByText(/discrete changes/)).not.toBeInTheDocument();
        expect(screen.queryByText(/continuous changes/)).not.toBeInTheDocument();
    });

    it("falls back to count rows when no PropertyTransformNode children are loaded", () => {
        // Simulate the depth=1 frame fetch where TransformDetail edges
        // weren't followed -- the user clicks the transform before
        // expanding it. Counts are the only thing we can show.
        const stripped: CytoscapeData = {
            elements: {
                nodes: data.elements.nodes.filter(
                    (n) => !String(n.data.labels).includes("PropertyTransformNode"),
                ),
                edges: data.elements.edges.filter(
                    (e) => e.data.type !== "TransformDetail",
                ),
            },
            meta: { root_id: 0, node_count: 0, edge_count: 0 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: stripped.elements.nodes.find((n) => n.data.id === "ot")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={stripped}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("discrete changes")).toBeInTheDocument();
        expect(screen.getByText("continuous changes")).toBeInTheDocument();
        expect(screen.queryByText("Property changes")).not.toBeInTheDocument();
    });

    it("shows IntrinsicTransform name and normalized change", () => {
        const intrinsicData: CytoscapeData = {
            elements: {
                nodes: [
                    {
                        data: {
                            id: "it",
                            labels: "IntrinsicTransform, Transform",
                            name: "hp",
                            normalized_change: -0.125,
                        },
                    },
                ],
                edges: [],
            },
            meta: { root_id: 0, node_count: 1, edge_count: 0 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: intrinsicData.elements.nodes[0]!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={intrinsicData}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getAllByText("IntrinsicTransform").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("-0.125")).toBeInTheDocument();
    });

    // Regression: clicking on a bare parent Transform used to render
    // "kind: Transform" plus the from/to frame ticks and *nothing else*
    // -- the user couldn't tell what the transform actually contained.
    // The fix walks outgoing Change edges to find the child ObjectTransform
    // and IntrinsicTransform nodes (the per-object/per-intrinsic changes
    // grouped under this frame transition) and lists them.
    it("bare Transform shows a frame-transition summary with child transforms", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: data.elements.nodes.find((n) => n.data.id === "parent")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={data}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Frame ticks render via the existing srcFrame/dstFrame walk.
        expect(screen.getByText(/from frame/)).toBeInTheDocument();
        expect(screen.getByText(/to frame/)).toBeInTheDocument();
        // The new "Frame transition with N change(s)" header.
        expect(screen.getByText(/Frame transition with 1 change/)).toBeInTheDocument();
        // The child ObjectTransform is listed by its kind.
        expect(screen.getByText("object")).toBeInTheDocument();
        // The change count summarises the property children.
        expect(screen.getByText(/4 props/)).toBeInTheDocument();
    });
});

describe("findTransformChildren", () => {
    function makeMixedTransitionGraph(): CytoscapeData {
        return {
            elements: {
                nodes: [
                    { data: { id: "frame-prev", labels: "Frame", tick: 41 } },
                    { data: { id: "frame-cur", labels: "Frame", tick: 42 } },
                    { data: { id: "parent", labels: "Transform" } },
                    {
                        data: {
                            id: "ot",
                            labels: "ObjectTransform, Transform",
                            object_uuid: "12345",
                            human_name: "rancorous-fey-devy",
                            num_discrete_changes: 1,
                            num_continuous_changes: 2,
                        },
                    },
                    {
                        data: {
                            id: "it",
                            labels: "IntrinsicTransform, Transform",
                            name: "hp",
                            normalized_change: -0.125,
                        },
                    },
                ],
                edges: [
                    { data: { id: "e1", source: "frame-prev", target: "parent", type: "Change" } },
                    { data: { id: "e2", source: "parent", target: "frame-cur", type: "Change" } },
                    { data: { id: "e3", source: "parent", target: "ot", type: "Change" } },
                    { data: { id: "e4", source: "parent", target: "it", type: "Change" } },
                ],
            },
            meta: { root_id: 0, node_count: 5, edge_count: 4 },
        };
    }

    it("returns one entry per child Object/Intrinsic transform", () => {
        const data = makeMixedTransitionGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const children = findTransformChildren("parent", nodeMap, data.elements.edges);
        expect(children).toHaveLength(2);
        expect(children.map((c) => c.kind).sort()).toEqual(["intrinsic", "object"]);
    });

    it("excludes Frame Change edges from the children list", () => {
        // Sanity: the parent's Change edges to its src/dst Frame must NOT
        // be confused with child transforms. The frame transition has 4
        // outgoing Change edges (1 to each frame, 1 per child), and only
        // the 2 non-frame ones should appear as children.
        const data = makeMixedTransitionGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const children = findTransformChildren("parent", nodeMap, data.elements.edges);
        for (const c of children) {
            expect(c.id).not.toBe("frame-cur");
            expect(c.id).not.toBe("frame-prev");
        }
    });

    it("populates ObjectTransform fields (humanName, objectUuid, changeCount)", () => {
        const data = makeMixedTransitionGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const children = findTransformChildren("parent", nodeMap, data.elements.edges);
        const obj = children.find((c) => c.kind === "object")!;
        expect(obj.humanName).toBe("rancorous-fey-devy");
        expect(obj.objectUuid).toBe("12345");
        expect(obj.changeCount).toBe(3); // discrete (1) + continuous (2)
    });

    it("populates IntrinsicTransform fields (intrinsicName, normalizedChange)", () => {
        const data = makeMixedTransitionGraph();
        const nodeMap = new Map(
            data.elements.nodes.map((n) => [n.data.id as string, n.data]),
        );
        const children = findTransformChildren("parent", nodeMap, data.elements.edges);
        const intr = children.find((c) => c.kind === "intrinsic")!;
        expect(intr.intrinsicName).toBe("hp");
        expect(intr.normalizedChange).toBe(-0.125);
    });
});

describe("PropertyTransform Detail panel", () => {
    // Regression: PropertyTransformNode used to be misclassified as
    // 'transform' (substring match on the label) and routed to TransformDetail
    // which had nothing to show for it -- the user saw "kind: Transform"
    // and no actual change details. Now it has its own type and detail panel.
    it("renders the actual property change for a PropertyTransformNode", () => {
        const ptnData: CytoscapeData = {
            elements: {
                nodes: [{
                    data: {
                        id: "ptn",
                        labels: "PropertyTransformNode",
                        property_name: "distance",
                        change_type: "continuous",
                        old_value: 1,
                        new_value: 6,
                        delta: 5,
                    },
                }],
                edges: [],
            },
            meta: { root_id: 0, node_count: 1, edge_count: 0 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "property-transform",
            data: ptnData.elements.nodes[0]!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={ptnData}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // The user must see the actual property name, change type, and values.
        // "PropertyTransformNode" appears in both the panel title (default
        // case in title switch falls back to data.labels) and the kind row.
        expect(screen.getAllByText("PropertyTransformNode").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("distance")).toBeInTheDocument();
        expect(screen.getByText("continuous")).toBeInTheDocument();
        expect(screen.getByText("1")).toBeInTheDocument();
        expect(screen.getByText("6")).toBeInTheDocument();
        expect(screen.getByText("5")).toBeInTheDocument();
        // CRITICAL: must NOT degenerate into the unhelpful "kind: Transform"
        // single-row display from the old substring-match bug.
        expect(screen.queryByText(/kind\s*Transform$/)).not.toBeInTheDocument();
    });

    it("renders a discrete change with old/new strings", () => {
        const ptnData: CytoscapeData = {
            elements: {
                nodes: [{
                    data: {
                        id: "ptn",
                        labels: "PropertyTransformNode",
                        property_name: "motion_direction",
                        change_type: "discrete",
                        old_value: "UP_LEFT",
                        new_value: "LEFT",
                        delta: null,
                    },
                }],
                edges: [],
            },
            meta: { root_id: 0, node_count: 1, edge_count: 0 },
        };
        const selected: SelectedElement = {
            kind: "node",
            type: "property-transform",
            data: ptnData.elements.nodes[0]!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={ptnData}
                actionMap={[]}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("motion_direction")).toBeInTheDocument();
        expect(screen.getByText("discrete")).toBeInTheDocument();
        expect(screen.getByText("UP_LEFT")).toBeInTheDocument();
        expect(screen.getByText("LEFT")).toBeInTheDocument();
    });
});

describe("findConnectedNodes", () => {
    const richData = makeRichCytoscapeData();

    it("finds outgoing connections from a node", () => {
        const connected = findConnectedNodes("1", richData);
        // Frame 1 has outgoing edges to nodes 2, 3, 4, 5
        expect(connected).toHaveLength(4);
        const ids = connected.map((c) => c.data.id);
        expect(ids).toContain("2");
        expect(ids).toContain("3");
        expect(ids).toContain("4");
        expect(ids).toContain("5");
    });

    it("includes edge type information", () => {
        const connected = findConnectedNodes("2", richData);
        const objectConn = connected.find((c) => c.data.id === "6");
        expect(objectConn?.edgeType).toBe("ObservedAs");
    });

    it("returns empty array for nodes with no outgoing edges", () => {
        const connected = findConnectedNodes("8", richData);
        expect(connected).toHaveLength(0);
    });

    it("returns empty array for unknown node ID", () => {
        const connected = findConnectedNodes("999", richData);
        expect(connected).toHaveLength(0);
    });
});

describe("lookupActionName", () => {
    const actionMap = [
        { action_id: 19, action_name: "West", action_key: "h" },
        { action_id: 0, action_name: "Wait", action_key: "." },
    ];

    it("returns action name from map", () => {
        expect(lookupActionName(19, actionMap)).toBe("West");
    });

    it("returns null for unknown action ID", () => {
        expect(lookupActionName(99, actionMap)).toBeNull();
    });

    it("returns null for undefined action ID", () => {
        expect(lookupActionName(undefined, actionMap)).toBeNull();
    });
});

describe("mergeGraphData", () => {
    it("adds new nodes and edges without duplicates", () => {
        const existing = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 42 } },
                    { data: { id: "2", labels: "ObjectInstance" } },
                ],
                edges: [
                    { data: { id: "e1", source: "1", target: "2", type: "Situated" } },
                ],
            },
            meta: { root_id: 1, node_count: 2, edge_count: 1 },
        });

        const incoming = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "2", labels: "ObjectInstance" } }, // duplicate
                    { data: { id: "3", labels: "Object", uuid: 100 } }, // new
                ],
                edges: [
                    { data: { id: "e1", source: "1", target: "2", type: "Situated" } }, // duplicate
                    { data: { id: "e2", source: "2", target: "3", type: "ObservedAs" } }, // new
                ],
            },
            meta: { root_id: 2, node_count: 2, edge_count: 2 },
        });

        const merged = mergeGraphData(existing, incoming);
        expect(merged.elements.nodes).toHaveLength(3); // 1, 2, 3
        expect(merged.elements.edges).toHaveLength(2); // e1, e2
        expect(merged.meta.root_id).toBe(1); // preserves existing root
        expect(merged.meta.node_count).toBe(3);
        expect(merged.meta.edge_count).toBe(2);
    });

    it("returns existing data unchanged when incoming is empty", () => {
        const existing = makeCytoscapeData();
        const empty = makeCytoscapeData({
            elements: { nodes: [], edges: [] },
            meta: { root_id: null, node_count: 0, edge_count: 0 },
        });

        const merged = mergeGraphData(existing, empty);
        expect(merged.elements.nodes).toHaveLength(existing.elements.nodes.length);
        expect(merged.elements.edges).toHaveLength(existing.elements.edges.length);
    });

    it("preserves all data fields on merged nodes", () => {
        const existing = makeCytoscapeData({
            elements: { nodes: [{ data: { id: "1", labels: "Frame", tick: 42 } }], edges: [] },
            meta: { root_id: 1, node_count: 1, edge_count: 0 },
        });
        const incoming = makeCytoscapeData({
            elements: { nodes: [{ data: { id: "2", labels: "Object", uuid: 999, resolve_count: 3 } }], edges: [] },
            meta: { root_id: null, node_count: 1, edge_count: 0 },
        });

        const merged = mergeGraphData(existing, incoming);
        const obj = merged.elements.nodes.find((n) => n.data.id === "2");
        expect(obj?.data.uuid).toBe(999);
        expect(obj?.data.resolve_count).toBe(3);
    });
});

describe("getDescendants", () => {
    it("finds all non-frame descendants via outgoing edges", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "10", labels: "ObjectInstance" } },
                    { data: { id: "20", labels: "Object" } },
                    { data: { id: "50", labels: "FeatureGroup" } },
                ],
                edges: [
                    { data: { id: "1-10", source: "1", target: "10", type: "Situated" } },
                    { data: { id: "10-20", source: "10", target: "20", type: "ObservedAs" } },
                    { data: { id: "10-50", source: "10", target: "50", type: "Features" } },
                ],
            },
            meta: { root_id: 1, node_count: 4, edge_count: 3 },
        });

        const desc = getDescendants("1", data);
        expect(desc).toEqual(new Set(["10", "20", "50"]));
    });

    it("stops at frame nodes", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 1 } },
                    { data: { id: "2", labels: "Frame", tick: 2 } },
                    { data: { id: "10", labels: "ObjectInstance" } },
                ],
                edges: [
                    { data: { id: "1-2", source: "1", target: "2", type: "NextFrame" } },
                    { data: { id: "1-10", source: "1", target: "10", type: "Situated" } },
                ],
            },
            meta: { root_id: 1, node_count: 3, edge_count: 2 },
        });

        const desc = getDescendants("1", data);
        // Should include OI but NOT Frame 2
        expect(desc).toEqual(new Set(["10"]));
    });

    it("returns empty set for leaf nodes", () => {
        const data = makeCytoscapeData({
            elements: {
                nodes: [{ data: { id: "60", labels: "FeatureNode", name: "glyph" } }],
                edges: [],
            },
            meta: { root_id: null, node_count: 1, edge_count: 0 },
        });

        expect(getDescendants("60", data)).toEqual(new Set());
    });
});

describe("Object History", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockUseActionMap.mockReturnValue({
            data: TEST_ACTION_MAP,
            isLoading: false,
        } as ReturnType<typeof useActionMap>);
        mockUseObjectHistoryGraph.mockReturnValue({
            data: undefined,
            isLoading: false,
        } as ReturnType<typeof useObjectHistoryGraph>);
    });

    it("shows 'View History' button when Object node is selected", () => {
        const richData = makeRichCytoscapeData();
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: { id: "6", labels: "Object", uuid: "12345", resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={richData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("View History")).toBeInTheDocument();
    });

    it("calls onViewHistory with uuid when View History is clicked", () => {
        const richData = makeRichCytoscapeData();
        const onViewHistory = vi.fn();
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: { id: "6", labels: "Object", uuid: "12345", resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={richData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={onViewHistory}
            />,
            { wrapper: Wrapper },
        );
        fireEvent.click(screen.getByText("View History"));
        // Callback receives the uuid as a string -- never coerce to Number,
        // that loses precision for 63-bit ROC UUIDs.
        expect(onViewHistory).toHaveBeenCalledWith("12345");
    });

    // Regression: ROC Object UUIDs are 63-bit ints. JS Number.MAX_SAFE_INTEGER
    // is 2^53 - 1 = 9007199254740991, so a 19-digit uuid like
    // 6853809301722933453 silently rounds to 6853809301722933000 if it's
    // ever stored as a number. The full pipeline -- wire format, type,
    // state, and View History callback -- must keep it as a string so
    // the trailing digits survive.
    it("View History round-trips a 63-bit uuid as a string without precision loss", () => {
        const BIG_UUID = "6853809301722933453";
        const richData = makeRichCytoscapeData();
        const onViewHistory = vi.fn();
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: {
                id: "-124",
                labels: "Object",
                uuid: BIG_UUID,
                resolve_count: 189,
                last_x: 23,
                last_y: 9,
                last_tick: 314,
            },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={richData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={onViewHistory}
            />,
            { wrapper: Wrapper },
        );
        fireEvent.click(screen.getByText("View History"));
        expect(onViewHistory).toHaveBeenCalledWith(BIG_UUID);
        // Sanity: the value the callback received is the *exact* string,
        // not the JS-Number-truncated form ("6853809301722933000").
        const arg = onViewHistory.mock.calls[0]![0] as string;
        expect(typeof arg).toBe("string");
        expect(arg).toBe(BIG_UUID);
        expect(arg).not.toBe("6853809301722933000");
        // The uuid is also rendered in the detail table -- check that
        // the unmodified string lands in the DOM (no Number coercion).
        expect(screen.getByText(BIG_UUID)).toBeInTheDocument();
    });

    it("renders human_name in the Object detail panel when present", () => {
        const richData = makeRichCytoscapeData();
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: {
                id: "-124",
                labels: "Object",
                uuid: "6853809301722933453",
                human_name: "rancorous-fey-devy",
                resolve_count: 189,
                last_x: 23,
                last_y: 9,
                last_tick: 314,
            },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={richData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // human_name appears in the detail table for users to see at a glance.
        expect(screen.getAllByText("rancorous-fey-devy").length).toBeGreaterThanOrEqual(1);
    });

    it("does not show View History button for non-Object nodes", () => {
        const richData = makeRichCytoscapeData();
        const selected: SelectedElement = {
            kind: "node",
            type: "frame",
            data: { id: "1", labels: "Frame", tick: 42 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={richData}
                actionMap={TEST_ACTION_MAP}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.queryByText("View History")).not.toBeInTheDocument();
    });

    // Regression: when entering Object History view, the dashboard used to
    // leave expandedNodes empty. The buildElements walk only renders
    // children of expanded nodes, so the user saw the timeline backbone
    // (171 lonely Frame nodes for a long-lived object) and none of the
    // ObjectInstances or the Object itself -- the very content the History
    // view exists to show. The fix auto-expands every node in the history
    // subgraph as soon as it arrives. This test asserts the post-effect
    // expandedNodes set produces the right buildElements output.
    it("buildElements renders ObjectInstances AND Object when all history nodes are expanded", () => {
        const historyData: CytoscapeData = {
            elements: {
                nodes: [
                    { data: { id: "f1", labels: "Frame", tick: 1 } },
                    { data: { id: "f2", labels: "Frame", tick: 2 } },
                    { data: { id: "oi1", labels: "ObjectInstance", x: 5, y: 5 } },
                    { data: { id: "oi2", labels: "ObjectInstance", x: 6, y: 5 } },
                    {
                        data: {
                            id: "obj",
                            labels: "Object",
                            uuid: "6853809301722933453",
                            human_name: "rancorous-fey-devy",
                            resolve_count: 2,
                        },
                    },
                ],
                edges: [
                    { data: { id: "e1", source: "f1", target: "oi1", type: "SituatedObjectInstance" } },
                    { data: { id: "e2", source: "f2", target: "oi2", type: "SituatedObjectInstance" } },
                    { data: { id: "e3", source: "oi1", target: "obj", type: "ObservedAs" } },
                    { data: { id: "e4", source: "oi2", target: "obj", type: "ObservedAs" } },
                    { data: { id: "e5", source: "f1", target: "f2", type: "NextFrame" } },
                ],
            },
            meta: { root_id: 0, node_count: 5, edge_count: 5 },
        };

        // The effect sets expandedNodes to every node id in historyData.
        // Reproduce that here.
        const expanded = new Set(historyData.elements.nodes.map((n) => String(n.data.id)));
        const els = buildElements(historyData, expanded);

        const placedNodeIds = new Set(
            els.filter((e) => e.group === "nodes").map((e) => String(e.data.id)),
        );
        // All three node types render. The bug was that the Object never
        // appeared because ObjectInstances weren't expanded, so the walk
        // never followed their ObservedAs out-edge.
        expect(placedNodeIds.has("f1")).toBe(true);
        expect(placedNodeIds.has("f2")).toBe(true);
        expect(placedNodeIds.has("oi1")).toBe(true);
        expect(placedNodeIds.has("oi2")).toBe(true);
        expect(placedNodeIds.has("obj")).toBe(true);
    });

    // Negative case: with an empty expandedNodes set (the OLD broken state
    // after handleViewHistory cleared it), only the Frame backbone renders
    // and the Object stays hidden. Pins the failure mode so a regression
    // to empty-expand would be caught.
    it("with empty expandedNodes the History view shows only the Frame backbone (regression baseline)", () => {
        const historyData: CytoscapeData = {
            elements: {
                nodes: [
                    { data: { id: "f1", labels: "Frame", tick: 1 } },
                    { data: { id: "oi1", labels: "ObjectInstance", x: 5, y: 5 } },
                    {
                        data: {
                            id: "obj",
                            labels: "Object",
                            uuid: "6853809301722933453",
                            human_name: "rancorous-fey-devy",
                        },
                    },
                ],
                edges: [
                    { data: { id: "e1", source: "f1", target: "oi1", type: "SituatedObjectInstance" } },
                    { data: { id: "e2", source: "oi1", target: "obj", type: "ObservedAs" } },
                ],
            },
            meta: { root_id: 0, node_count: 3, edge_count: 2 },
        };
        const els = buildElements(historyData, new Set());
        const placedNodeIds = new Set(
            els.filter((e) => e.group === "nodes").map((e) => String(e.data.id)),
        );
        expect(placedNodeIds.has("f1")).toBe(true);
        // Without expansion, the buildElements walk doesn't reach
        // ObjectInstance or Object -- this is the old broken state.
        expect(placedNodeIds.has("oi1")).toBe(false);
        expect(placedNodeIds.has("obj")).toBe(false);
    });
});

describe("getActionId", () => {
    it("reads the `action` field from live graph data", () => {
        expect(getActionId({ action: 4 })).toBe(4);
    });

    it("reads the `action_id` field from archived/legacy data", () => {
        expect(getActionId({ action_id: 19 })).toBe(19);
    });

    it("prefers `action` when both are present", () => {
        expect(getActionId({ action: 7, action_id: 19 })).toBe(7);
    });

    it("returns undefined when neither field is present", () => {
        expect(getActionId({ labels: "TakeAction" })).toBeUndefined();
    });
});

describe("getFeatureNodeDisplay", () => {
    it("decodes ShapeNode as a character", () => {
        expect(
            getFeatureNodeDisplay({ labels: "ShapeNode, FeatureNode", type: 64 }),
        ).toEqual({ name: "Shape", value: "@" });
    });

    it("decodes ColorNode by name", () => {
        expect(
            getFeatureNodeDisplay({ labels: "ColorNode, FeatureNode", type: 7 }),
        ).toEqual({ name: "Color", value: "GREY" });
    });

    it("decodes SingleNode as a raw glyph type", () => {
        expect(
            getFeatureNodeDisplay({ labels: "FeatureNode, SingleNode", type: 2364 }),
        ).toEqual({ name: "Single", value: "2364" });
    });

    it("decodes FloodNode as a summary string", () => {
        const result = getFeatureNodeDisplay({
            labels: "FeatureNode, FloodNode",
            type: 1,
            size: 4,
            color: 7,
            shape: 46,
        });
        expect(result.name).toBe("Flood");
        expect(result.value).toContain("size=4");
        expect(result.value).toContain("color=GREY");
        expect(result.value).toContain("shape=.");
    });

    it("decodes LineNode as a summary string", () => {
        const result = getFeatureNodeDisplay({
            labels: "FeatureNode, LineNode",
            type: 2,
            size: 7,
            color: 15,
            shape: 45,
        });
        expect(result.name).toBe("Line");
        expect(result.value).toContain("size=7");
        expect(result.value).toContain("color=WHITE");
        expect(result.value).toContain("shape=-");
    });

    it("decodes DeltaNode as old -> new", () => {
        expect(
            getFeatureNodeDisplay({
                labels: "FeatureNode, DeltaNode",
                old_val: 32,
                new_val: 64,
            }),
        ).toEqual({ name: "Delta", value: "32 -> 64" });
    });

    it("decodes MotionNode as type + direction", () => {
        const result = getFeatureNodeDisplay({
            labels: "FeatureNode, MotionNode",
            type: 64,
            direction: "UP_RIGHT",
        });
        expect(result.name).toBe("Motion");
        expect(result.value).toContain("UP_RIGHT");
    });

    it("decodes DistanceNode as size", () => {
        expect(
            getFeatureNodeDisplay({ labels: "FeatureNode, DistanceNode", size: 5 }),
        ).toEqual({ name: "Distance", value: "5" });
    });

    it("decodes PhonemeNode as type", () => {
        expect(
            getFeatureNodeDisplay({ labels: "FeatureNode, PhonemeNode", type: 42 }),
        ).toEqual({ name: "Phoneme", value: "42" });
    });

    it("falls back to {name, value} fields for legacy/test data", () => {
        expect(
            getFeatureNodeDisplay({ labels: "FeatureNode", name: "glyph", value: "@" }),
        ).toEqual({ name: "glyph", value: "@" });
    });

    it("falls back to 'Feature' when no recognised fields exist", () => {
        expect(getFeatureNodeDisplay({ labels: "FeatureNode" })).toEqual({
            name: "Feature",
            value: "",
        });
    });
});

describe("makeLabel", () => {
    it("uses shape_type for ObjectInstance glyph", () => {
        const label = makeLabel(
            { id: "1", labels: "ObjectInstance", shape_type: 64 },
            "object-instance",
        );
        expect(label).toBe("ObjInst (@)");
    });

    it("falls back to plain ObjInst when no shape_type", () => {
        const label = makeLabel(
            { id: "1", labels: "ObjectInstance" },
            "object-instance",
        );
        expect(label).toBe("ObjInst");
    });

    it("finds Object glyph via ObservedAs edge from ObjectInstance", () => {
        const nodeMap = new Map<string, Record<string, unknown>>();
        nodeMap.set("2", { id: "2", labels: "ObjectInstance", shape_type: 100 });
        nodeMap.set("6", { id: "6", labels: "Object" });
        const allEdges = [
            { data: { id: "e1", source: "2", target: "6", type: "ObservedAs" } },
        ];
        const label = makeLabel(
            { id: "6", labels: "Object" },
            "object",
            undefined,
            { nodeMap, allEdges },
        );
        expect(label).toBe("Object (d)");
    });

    it("falls back to plain Object when no context or no linked instance", () => {
        expect(makeLabel({ id: "6", labels: "Object" }, "object")).toBe("Object");
    });

    it("uses action map with the live `action` field", () => {
        const actionMap = [{ action_id: 4, action_name: "NE", action_key: "u" }];
        const label = makeLabel(
            { id: "3", labels: "TakeAction", action: 4 },
            "action",
            actionMap,
        );
        expect(label).toBe("Action (NE)");
    });

    it("uses action map with the archived `action_id` field", () => {
        const actionMap = [{ action_id: 19, action_name: "Wait", action_key: "." }];
        const label = makeLabel(
            { id: "3", labels: "TakeAction", action_id: 19 },
            "action",
            actionMap,
        );
        expect(label).toBe("Action (Wait)");
    });

    it("formats feature-node labels via getFeatureNodeDisplay", () => {
        expect(
            makeLabel(
                { id: "9", labels: "ColorNode, FeatureNode", type: 7 },
                "feature-node",
            ),
        ).toBe("Color (GREY)");
    });

    it("distinguishes ObjectTransform in the label", () => {
        expect(
            makeLabel(
                { id: "x", labels: "Transform, ObjectTransform" },
                "transform",
            ),
        ).toBe("ObjectTransform");
    });

    it("distinguishes IntrinsicTransform in the label", () => {
        expect(
            makeLabel(
                { id: "x", labels: "Transform, IntrinsicTransform", name: "hp" },
                "transform",
            ),
        ).toBe("IntrinsicTransform");
    });
});

describe("Detail Panel (live graph format)", () => {
    // Builds test data using the field names the live graph API actually
    // returns (shape_type, glyph_type, action, etc.) -- not the legacy
    // {name, value, glyph} shape used by older tests.
    function makeLiveData(): CytoscapeData {
        return {
            elements: {
                nodes: [
                    { data: { id: "1", labels: "Frame", tick: 20 } },
                    // ObjectInstance for `@` (player)
                    {
                        data: {
                            id: "2",
                            labels: "ObjectInstance",
                            object_uuid: 111,
                            x: 5,
                            y: 10,
                            tick: 20,
                            glyph_type: 333,
                            color_type: 15,
                            shape_type: 64, // '@'
                            distance: 0,
                        },
                    },
                    // The Object this instance was resolved to
                    {
                        data: {
                            id: "6",
                            labels: "Object",
                            uuid: 111,
                            resolve_count: 3,
                            last_x: 5,
                            last_y: 10,
                            last_tick: 20,
                        },
                    },
                    // FeatureGroup containing the Color/Shape/Single feature nodes
                    { data: { id: "7", labels: "FeatureGroup" } },
                    { data: { id: "8", labels: "ColorNode, FeatureNode", type: 15 } },
                    { data: { id: "9", labels: "ShapeNode, FeatureNode", type: 64 } },
                    { data: { id: "10", labels: "FeatureNode, SingleNode", type: 333 } },
                    // RelationshipGroup containing a Distance node
                    { data: { id: "11", labels: "RelationshipGroup" } },
                    { data: { id: "12", labels: "FeatureNode, DistanceNode", size: 2 } },
                    // TakeAction using the live `action` field
                    { data: { id: "3", labels: "TakeAction", action: 4 } },
                    // IntrinsicNode
                    {
                        data: {
                            id: "4",
                            labels: "IntrinsicNode",
                            name: "hp",
                            raw_value: 15,
                            normalized_value: 0.85,
                        },
                    },
                    // ObjectTransform
                    {
                        data: {
                            id: "13",
                            labels: "Transform, ObjectTransform",
                            object_uuid: 111,
                            num_discrete_changes: 0,
                            num_continuous_changes: 1,
                        },
                    },
                ],
                edges: [
                    { data: { id: "e1", source: "1", target: "2", type: "SituatedObjectInstance" } },
                    { data: { id: "e2", source: "1", target: "3", type: "FrameAttribute" } },
                    { data: { id: "e3", source: "1", target: "4", type: "FrameAttribute" } },
                    { data: { id: "e4", source: "2", target: "6", type: "ObservedAs" } },
                    { data: { id: "e5", source: "2", target: "7", type: "Features" } },
                    { data: { id: "e6", source: "7", target: "8", type: "Detail" } },
                    { data: { id: "e7", source: "7", target: "9", type: "Detail" } },
                    { data: { id: "e8", source: "7", target: "10", type: "Detail" } },
                    { data: { id: "e9", source: "2", target: "11", type: "Relationships" } },
                    { data: { id: "e10", source: "11", target: "12", type: "Detail" } },
                ],
            },
            meta: { root_id: 1, node_count: 12, edge_count: 10 },
        };
    }

    const liveActionMap = [
        { action_id: 4, action_name: "NE", action_key: "u" },
    ];

    it("renders decoded features for a FeatureGroup (regression for empty rows bug)", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "feature-group",
            data: { id: "7", labels: "FeatureGroup" },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Decoded color
        expect(screen.getByText("Color")).toBeInTheDocument();
        expect(screen.getByText("WHITE")).toBeInTheDocument();
        // Decoded shape (@)
        expect(screen.getByText("Shape")).toBeInTheDocument();
        // "Single" with raw glyph id
        expect(screen.getByText("Single")).toBeInTheDocument();
        expect(screen.getByText("333")).toBeInTheDocument();
    });

    it("renders decoded features for a RelationshipGroup", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "relationship-group",
            data: { id: "11", labels: "RelationshipGroup" },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("Distance")).toBeInTheDocument();
        expect(screen.getByText("2")).toBeInTheDocument();
    });

    it("ObjectInstance detail shows shape_type as character", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "object-instance",
            data: liveData.elements.nodes.find((n) => n.data.id === "2")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // '@' shows in the title and in the big display character
        expect(screen.getAllByText("@").length).toBeGreaterThanOrEqual(1);
        // WHITE appears twice -- once for the ObjectInstance.color_type
        // attribute row and once for the decoded Color feature.
        expect(screen.getAllByText("WHITE").length).toBeGreaterThanOrEqual(1);
        // Both the raw NetHack glyph id (attribute row) and decoded displays
        // are present. Check label key.
        expect(screen.getByText("glyph_type")).toBeInTheDocument();
    });

    it("Object detail title includes glyph from ObservedAs instance", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "object",
            data: liveData.elements.nodes.find((n) => n.data.id === "6")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Title shows "Object (@)" and body shows UUID
        expect(screen.getByText("Object (@)")).toBeInTheDocument();
        expect(screen.getByText("111")).toBeInTheDocument();
    });

    it("Frame detail uses the live `action` field to look up action name", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "frame",
            data: { id: "1", labels: "Frame", tick: 20 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Badge shows mapped name (not "4")
        expect(screen.getByText("NE")).toBeInTheDocument();
    });

    it("Action node detail renders id, name, and key", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "action",
            data: { id: "3", labels: "TakeAction", action: 4 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Title includes the action name
        expect(screen.getByText("Action (NE)")).toBeInTheDocument();
        // Table rows
        expect(screen.getByText("name")).toBeInTheDocument();
        expect(screen.getByText("NE")).toBeInTheDocument();
        expect(screen.getByText("u")).toBeInTheDocument();
    });

    it("Intrinsic node detail renders name, raw value, and normalized", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "intrinsic",
            data: {
                id: "4",
                labels: "IntrinsicNode",
                name: "hp",
                raw_value: 15,
                normalized_value: 0.85,
            },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={makeLiveData()}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("Intrinsic (hp)")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("15")).toBeInTheDocument();
        expect(screen.getByText("0.850")).toBeInTheDocument();
    });

    it("Transform node detail distinguishes ObjectTransform", () => {
        const liveData = makeLiveData();
        const selected: SelectedElement = {
            kind: "node",
            type: "transform",
            data: liveData.elements.nodes.find((n) => n.data.id === "13")!.data,
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={liveData}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        // Title + "kind" row
        expect(screen.getAllByText("ObjectTransform").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("discrete changes")).toBeInTheDocument();
        expect(screen.getByText("continuous changes")).toBeInTheDocument();
    });

    it("FeatureNode detail shows decoded type and value for ColorNode", () => {
        const selected: SelectedElement = {
            kind: "node",
            type: "feature-node",
            data: { id: "8", labels: "ColorNode, FeatureNode", type: 15 },
        };
        render(
            <DetailPanel
                selected={selected}
                apiData={makeLiveData()}
                actionMap={liveActionMap}
                onClose={vi.fn()}
                onViewHistory={vi.fn()}
            />,
            { wrapper: Wrapper },
        );
        expect(screen.getByText("Color (WHITE)")).toBeInTheDocument();
    });
});
