import { fireEvent, render, screen } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import type { CytoscapeData } from "../../types/step-data";
import {
    buildElements,
    DetailPanel,
    findConnectedNodes,
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

    it("re-fetches when step changes after debounce", () => {
        vi.useFakeTimers();
        mockUseFrameGraph.mockReturnValue({
            data: makeCytoscapeData(),
            isLoading: false,
        } as ReturnType<typeof useFrameGraph>);

        const { rerender } = render(
            <GraphVisualization run="test-run" step={1} />,
            { wrapper: Wrapper },
        );

        // Initial render uses step=1 (debounced state initializes to prop)
        expect(mockUseFrameGraph).toHaveBeenCalledWith("test-run", 1, undefined, 1);

        rerender(
            <Wrapper>
                <GraphVisualization run="test-run" step={2} />
            </Wrapper>,
        );

        // After debounce timer fires, hook should be called with step=2
        vi.advanceTimersByTime(300);
        expect(mockUseFrameGraph).toHaveBeenCalledWith("test-run", 2, undefined, 1);

        vi.useRealTimers();
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
                { data: { id: "6", labels: "Object", uuid: 12345, resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 } },
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
            data: { id: "6", labels: "Object", uuid: 12345, resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
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
            data: { id: "6", labels: "Object", uuid: 12345, resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
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
            data: { id: "6", labels: "Object", uuid: 12345, resolve_count: 7, last_x: 5, last_y: 10, last_tick: 42 },
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
        expect(onViewHistory).toHaveBeenCalledWith(12345);
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
