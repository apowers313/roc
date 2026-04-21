/**
 * Graph visualization panel using Cytoscape.js.
 *
 * Fetches a frame subgraph from the graph API and renders it with a manual
 * hierarchical layout. Frames are pinned in a horizontal timeline row.
 * Click a node to expand/collapse its children.
 *
 * Base/fallback node style MUST come before type-specific styles because
 * Cytoscape applies styles in array order, not CSS specificity.
 */

import {
    Alert, Badge, Box, Button, Checkbox, CloseButton, Divider, Group, Paper,
    Popover, Progress, ScrollArea, Stack, Table, Text,
} from "@mantine/core";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import CytoscapeComponent from "react-cytoscapejs";

import { fetchNodeGraph, type ActionMapEntry } from "../../api/client";
import { useActionMap, useFrameGraph, useObjectHistoryGraph } from "../../api/queries";
import type { CytoscapeData } from "../../types/step-data";

// Register the fcose layout extension exactly once at module load.
cytoscape.use(fcose);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Force-directed layout (cytoscape-fcose) parameters. Frames are pinned in a
// horizontal timeline row via fixedNodeConstraint; everything else is placed
// by physics so children fan out around their parent without colliding.
// Tuned via the prototype experiment in tmp/fcose_experiment.html.
const FRAME_Y = 0;
const FRAME_X_GAP = 400;
const NODE_REPULSION = 6500;
const IDEAL_EDGE_LENGTH = 70;
const NODE_SEPARATION = 75;
const LAYOUT_ITERATIONS = 2500;
const LAYOUT_ANIMATION_MS = 400;
const PINNED_BORDER_COLOR = "#FCC419";

const NODE_COUNT_WARNING_THRESHOLD = 200;
// Step debouncing moved to App.tsx (useDebouncedValue with resetKey).
// fcose computes positions; "preset" is just a no-op kickstart layout that
// react-cytoscapejs uses on initial mount. The real layout runs imperatively
// in a useEffect after each elements update.
const PRESET_LAYOUT: cytoscape.LayoutOptions = { name: "preset" };

// All node types are expandable -- clicking any node fetches its neighbors.

/** All filterable node types with display labels, shown in the toolbar
 *  "Show Nodes" dropdown. Frames are intentionally excluded -- they are the
 *  timeline backbone and always visible. */
export const NODE_TYPE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
    { value: "object-instance", label: "Object Instance" },
    { value: "object", label: "Object" },
    { value: "feature-group", label: "Feature Group" },
    { value: "relationship-group", label: "Relationship Group" },
    { value: "feature-node", label: "Feature Node" },
    { value: "action", label: "Action" },
    { value: "intrinsic", label: "Intrinsic" },
    { value: "transform", label: "Transform" },
    { value: "property-transform", label: "Property Change" },
];

/** Node types visible by default: everything except Intrinsic. Intrinsics are
 *  still reachable from the detail panel of their parent node. */
export const DEFAULT_VISIBLE_NODE_TYPES: readonly string[] = NODE_TYPE_OPTIONS
    .filter((o) => o.value !== "intrinsic")
    .map((o) => o.value);

/** Default hidden-types set used by ``buildElements`` when no explicit set
 *  is passed (preserves prior behavior of hiding intrinsic nodes in tests and
 *  other callers that have not opted in to the filter). */
const DEFAULT_HIDDEN_TYPES: ReadonlySet<string> = new Set(["intrinsic"]);

// ---------------------------------------------------------------------------
// Selected element type
// ---------------------------------------------------------------------------

/** Represents a clicked node or edge for the detail panel. */
export interface SelectedElement {
    kind: "node" | "edge";
    type: string;
    data: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Detail panel helpers
// ---------------------------------------------------------------------------

/** Find all nodes connected from nodeId via outgoing edges. */
export function findConnectedNodes(
    nodeId: string,
    apiData: CytoscapeData,
): Array<{ data: Record<string, unknown>; edgeType: string }> {
    const nodeMap = new Map<string, Record<string, unknown>>();
    for (const n of apiData.elements.nodes) {
        nodeMap.set(n.data.id, n.data as Record<string, unknown>);
    }
    const results: Array<{ data: Record<string, unknown>; edgeType: string }> = [];
    for (const e of apiData.elements.edges) {
        if (e.data.source === nodeId) {
            const targetData = nodeMap.get(e.data.target);
            if (targetData) results.push({ data: targetData, edgeType: e.data.type ?? "" });
        }
    }
    return results;
}

/** Look up an action name from the action map. Returns null if not found. */
export function lookupActionName(
    actionId: number | undefined,
    actionMap: ActionMapEntry[],
): string | null {
    if (actionId == null) return null;
    const entry = actionMap.find((e) => e.action_id === actionId);
    return entry?.action_name ?? null;
}

/** Read the action ID from a TakeAction node.
 *
 * Historical archives and legacy test data use `action_id`; the live graph
 * uses `action` (matching the pydantic field name on TakeAction). Accept
 * either to keep both formats working.
 */
export function getActionId(data: Record<string, unknown>): number | undefined {
    const raw = data.action ?? data.action_id;
    return raw != null ? Number(raw) : undefined;
}

/** NetHack color ID -> human-readable name. Matches ColorNode.attr_strs. */
const COLOR_NAMES = [
    "BLACK", "RED", "GREEN", "BROWN", "BLUE", "MAGENTA",
    "CYAN", "GREY", "NO COLOR", "ORANGE", "BRIGHT GREEN",
    "YELLOW", "BRIGHT BLUE", "BRIGHT MAGENTA", "BRIGHT CYAN",
    "WHITE", "MAX",
];

function colorName(val: unknown): string {
    if (typeof val !== "number") return String(val ?? "");
    return COLOR_NAMES[val] ?? String(val);
}

/** Convert a character code to a single-char string, or fall back to the raw value. */
function charFromCode(val: unknown): string {
    if (typeof val !== "number") return String(val ?? "");
    if (val < 32 || val > 126) return String(val);
    return String.fromCharCode(val);
}

/** Extract a human-readable (name, value) pair for a FeatureNode.
 *
 * FeatureNodes don't have a uniform shape -- the specific subtype
 * (ColorNode, ShapeNode, etc.) determines which fields matter and how
 * they should be displayed. We parse the compound label string to pick
 * the right decoder. Keeps legacy `{name, value}` test data working via
 * the fallback at the end.
 */
export function getFeatureNodeDisplay(
    data: Record<string, unknown>,
): { name: string; value: string } {
    const labels = String(data.labels ?? "");
    const type = data.type;
    const size = data.size;

    if (labels.includes("ShapeNode")) {
        return { name: "Shape", value: charFromCode(type) };
    }
    if (labels.includes("ColorNode")) {
        return { name: "Color", value: colorName(type) };
    }
    if (labels.includes("SingleNode")) {
        return { name: "Single", value: String(type ?? "") };
    }
    if (labels.includes("FloodNode") || labels.includes("LineNode")) {
        const kind = labels.includes("FloodNode") ? "Flood" : "Line";
        const parts = [
            typeof type === "number" ? `type=${type}` : null,
            typeof size === "number" ? `size=${size}` : null,
            typeof data.color === "number" ? `color=${colorName(data.color)}` : null,
            typeof data.shape === "number" ? `shape=${charFromCode(data.shape)}` : null,
        ].filter(Boolean);
        return { name: kind, value: parts.join(", ") };
    }
    if (labels.includes("DeltaNode")) {
        return {
            name: "Delta",
            value: `${String(data.old_val ?? "?")} -> ${String(data.new_val ?? "?")}`,
        };
    }
    if (labels.includes("MotionNode")) {
        const dir = data.direction != null ? ` ${String(data.direction)}` : "";
        return { name: "Motion", value: `${String(type ?? "?")}${dir}` };
    }
    if (labels.includes("DistanceNode")) {
        return { name: "Distance", value: String(size ?? "?") };
    }
    if (labels.includes("PhonemeNode")) {
        return { name: "Phoneme", value: String(type ?? "") };
    }
    // Legacy/test data fallback: many tests write {name, value} directly.
    return {
        name: String(data.name ?? "Feature"),
        value: String(data.value ?? ""),
    };
}

/** Get the character representation for an ObjectInstance from shape_type. */
function getInstanceChar(data: Record<string, unknown>): string | null {
    if (typeof data.shape_type === "number") return charFromCode(data.shape_type);
    if (data.glyph != null) return String(data.glyph);
    return null;
}

/** Walk incoming ObservedAs edges to find an ObjectInstance that resolved to this Object.
 *  Used so Object nodes can show the same glyph character as their instances. */
function getObjectChar(
    objectId: string,
    nodeMap: Map<string, Record<string, unknown>>,
    allEdges: Array<{ data: Record<string, unknown> }>,
): string | null {
    for (const e of allEdges) {
        if (e.data.target !== objectId) continue;
        const edgeType = String(e.data.type ?? "");
        if (!edgeType.includes("ObservedAs")) continue;
        const src = nodeMap.get(String(e.data.source));
        if (!src) continue;
        const ch = getInstanceChar(src);
        if (ch != null) return ch;
    }
    return null;
}

/** Merge incoming graph data into existing, deduplicating by node/edge ID. */
export function mergeGraphData(
    existing: CytoscapeData,
    incoming: CytoscapeData,
): CytoscapeData {
    const existingNodeIds = new Set(existing.elements.nodes.map((n) => n.data.id));
    const existingEdgeIds = new Set(existing.elements.edges.map((e) => e.data.id));

    const newNodes = incoming.elements.nodes.filter((n) => !existingNodeIds.has(n.data.id));
    const newEdges = incoming.elements.edges.filter((e) => !existingEdgeIds.has(e.data.id));

    const mergedNodes = [...existing.elements.nodes, ...newNodes];
    const mergedEdges = [...existing.elements.edges, ...newEdges];

    return {
        elements: { nodes: mergedNodes, edges: mergedEdges },
        meta: {
            root_id: existing.meta.root_id,
            node_count: mergedNodes.length,
            edge_count: mergedEdges.length,
        },
    };
}

// ---------------------------------------------------------------------------
// Node / edge type mapping
// ---------------------------------------------------------------------------

/** Map a graph node's labels attribute to a visual type string.
 *
 *  IMPORTANT: order matters because we use substring matching. PropertyTransformNode
 *  must be checked BEFORE the bare-Transform branch -- the substring "Transform"
 *  is in "PropertyTransformNode" too, and a misclassification sends it to the
 *  TransformDetail panel which shows only "kind: Transform" with no useful info. */
export function getNodeType(data: Record<string, unknown>): string {
    const labels = String(data.labels ?? "");
    // Check more specific types first (ObjectInstance before Object,
    // RelationshipGroup before FeatureGroup since both are groups,
    // PropertyTransformNode / ObjectTransform / IntrinsicTransform before bare Transform).
    if (labels.includes("ObjectInstance")) return "object-instance";
    if (labels.includes("PropertyTransformNode")) return "property-transform";
    if (labels.includes("ObjectTransform")) return "transform";
    if (labels.includes("IntrinsicTransform")) return "transform";
    if (labels.includes("Frame")) return "frame";
    if (labels.includes("RelationshipGroup")) return "relationship-group";
    if (labels.includes("FeatureGroup")) return "feature-group";
    if (labels.includes("FeatureNode")) return "feature-node";
    if (labels.includes("TakeAction")) return "action";
    if (labels.includes("IntrinsicNode")) return "intrinsic";
    if (labels.includes("Object")) return "object";
    if (labels.includes("Transform")) return "transform";
    return "unknown";
}

/** Map a graph edge's type attribute to a visual type string. */
function getEdgeVisualType(data: Record<string, unknown>): string {
    const type = String(data.type ?? "");
    if (type.includes("NextFrame")) return "next-frame";
    if (type.includes("Situated")) return "situated";
    if (type.includes("FrameAttribute")) return "frame-attribute";
    if (type.includes("ObservedAs")) return "observed-as";
    if (type.includes("Relationships")) return "relationships";
    if (type.includes("Features")) return "features";
    if (type.includes("Detail")) return "detail";
    if (type.includes("Change")) return "transform";
    return "default";
}

/** Context passed to makeLabel for cross-node lookups (e.g. Object glyph lookup). */
export interface LabelContext {
    nodeMap?: Map<string, Record<string, unknown>>;
    allEdges?: Array<{ data: Record<string, unknown> }>;
}

/** Build a display label for a node based on its type and attributes.
 *  Uses literal node type names with glyph characters, matching the design. */
export function makeLabel(
    data: Record<string, unknown>,
    nodeType: string,
    actionMap?: ActionMapEntry[],
    context?: LabelContext,
): string {
    switch (nodeType) {
        case "frame":
            return `Frame ${data.tick ?? ""}`;
        case "object-instance": {
            const ch = getInstanceChar(data);
            return ch != null ? `ObjInst (${ch})` : "ObjInst";
        }
        case "object": {
            const ch =
                context?.nodeMap && context.allEdges
                    ? getObjectChar(String(data.id), context.nodeMap, context.allEdges)
                    : null;
            return ch != null ? `Object (${ch})` : "Object";
        }
        case "feature-group":
            return "FeatureGroup";
        case "relationship-group":
            return "RelationshipGroup";
        case "action": {
            const actionId = getActionId(data);
            const name = actionMap ? lookupActionName(actionId, actionMap) : null;
            if (name) return `Action (${name})`;
            return actionId != null ? `Action ${actionId}` : "Action";
        }
        case "feature-node": {
            const { name, value } = getFeatureNodeDisplay(data);
            return value ? `${name} (${value})` : name;
        }
        case "intrinsic":
            return String(data.name ?? "Intrinsic");
        case "transform": {
            const labels = String(data.labels ?? "");
            if (labels.includes("ObjectTransform")) return "ObjectTransform";
            if (labels.includes("IntrinsicTransform")) return "IntrinsicTransform";
            return "Transform";
        }
        case "property-transform":
            return String(data.property_name ?? "PropChange");
        default:
            return String(data.labels ?? "Node");
    }
}

/** Build a display label for an edge. */
function makeEdgeLabel(data: Record<string, unknown>): string {
    return String(data.type ?? "");
}

// ---------------------------------------------------------------------------
// Element selection helpers
// ---------------------------------------------------------------------------

/** Get visible children of a node. Frames are always excluded (they are
 *  laid out in the top-level timeline row, not as children of other nodes);
 *  any type in ``hiddenTypes`` is also excluded. */
function getVisibleChildren(
    nodeId: string,
    outEdges: Map<string, Array<{ target: string }>>,
    nodeMap: Map<string, Record<string, unknown>>,
    hiddenTypes: ReadonlySet<string>,
): string[] {
    return (outEdges.get(nodeId) ?? [])
        .map((e) => e.target)
        .filter((id) => {
            const d = nodeMap.get(id);
            if (d == null) return false;
            const type = getNodeType(d);
            return type !== "frame" && !hiddenTypes.has(type);
        });
}

// ---------------------------------------------------------------------------
// Build Cytoscape elements from API data + expand state
// ---------------------------------------------------------------------------

/** Build the cytoscape element list from API data + expand state.
 *
 *  This is a pure data builder -- it does NOT assign positions. The actual
 *  layout is computed by cytoscape-fcose in a side effect after the elements
 *  are added to the cy instance. Frames are pinned in a horizontal row via
 *  fixedNodeConstraint; everything else is force-directed.
 *
 *  Each element has an explicit ``group`` field ("nodes" or "edges") so the
 *  caller can distinguish them without inspecting cytoscape internals.
 */
export function buildElements(
    apiData: CytoscapeData,
    expandedNodes: Set<string>,
    actionMap?: ActionMapEntry[],
    hiddenTypes: ReadonlySet<string> = DEFAULT_HIDDEN_TYPES,
): cytoscape.ElementDefinition[] {
    const elements: cytoscape.ElementDefinition[] = [];

    // Build indexes
    const nodeMap = new Map<string, Record<string, unknown>>();
    for (const n of apiData.elements.nodes) {
        nodeMap.set(n.data.id, n.data);
    }

    const outEdges = new Map<string, Array<{ target: string; data: Record<string, unknown> }>>();
    for (const e of apiData.elements.edges) {
        const arr = outEdges.get(e.data.source) ?? [];
        arr.push({ target: e.data.target, data: e.data });
        outEdges.set(e.data.source, arr);
    }

    // Find and sort frame nodes by tick
    const frames = apiData.elements.nodes
        .filter((n) => getNodeType(n.data) === "frame")
        .sort((a, b) => Number(a.data.tick ?? 0) - Number(b.data.tick ?? 0));

    // Track which nodes/edges have been added to avoid duplicates
    const placedNodes = new Set<string>();
    const placedEdges = new Set<string>();

    const labelContext: LabelContext = {
        nodeMap,
        allEdges: apiData.elements.edges,
    };

    function addNode(id: string): void {
        if (placedNodes.has(id)) return;
        const data = nodeMap.get(id);
        if (!data) return;

        const nodeType = getNodeType(data);
        placedNodes.add(id);
        elements.push({
            group: "nodes",
            data: {
                ...data,
                _type: nodeType,
                _label: makeLabel(data, nodeType, actionMap, labelContext),
            },
        });
    }

    function addEdgeBetween(nodeA: string, nodeB: string): void {
        // Check A -> B
        for (const e of outEdges.get(nodeA) ?? []) {
            if (e.target === nodeB) {
                const edgeId = String(e.data.id ?? `${nodeA}-${nodeB}`);
                if (placedEdges.has(edgeId)) return;
                placedEdges.add(edgeId);
                const eType = getEdgeVisualType(e.data);
                elements.push({
                    group: "edges",
                    data: { ...e.data, _type: eType, _label: makeEdgeLabel(e.data) },
                });
                return;
            }
        }
        // Check B -> A (reverse direction)
        for (const e of outEdges.get(nodeB) ?? []) {
            if (e.target === nodeA) {
                const edgeId = String(e.data.id ?? `${nodeB}-${nodeA}`);
                if (placedEdges.has(edgeId)) return;
                placedEdges.add(edgeId);
                const eType = getEdgeVisualType(e.data);
                elements.push({
                    group: "edges",
                    data: { ...e.data, _type: eType, _label: makeEdgeLabel(e.data) },
                });
                return;
            }
        }
    }

    /** Walk children of an expanded node up to ``maxDepth`` levels. Adds the
     *  child nodes and the parent->child edges. The recursion is BFS-like
     *  (via stack iteration) so each node is visited at most once. */
    function walkExpandedSubtree(rootId: string, maxDepth: number): void {
        if (maxDepth <= 0 || !expandedNodes.has(rootId)) return;
        const stack: Array<{ id: string; depth: number }> = [{ id: rootId, depth: 0 }];
        while (stack.length > 0) {
            const { id: parentId, depth } = stack.pop()!;
            if (depth >= maxDepth) continue;
            const children = getVisibleChildren(parentId, outEdges, nodeMap, hiddenTypes);
            for (const childId of children) {
                addNode(childId);
                addEdgeBetween(parentId, childId);
                if (expandedNodes.has(childId)) {
                    stack.push({ id: childId, depth: depth + 1 });
                }
            }
        }
    }

    // Add all frames + NextFrame edges between them (the timeline backbone)
    frames.forEach((frame, fi) => {
        addNode(frame.data.id);
        if (fi > 0) {
            const prevId = frames[fi - 1]!.data.id;
            addEdgeBetween(prevId, frame.data.id);
        }
        // Walk children of expanded frames up to 3 levels deep
        walkExpandedSubtree(frame.data.id, 3);
    });

    // Final pass: draw every edge whose endpoints are both already placed.
    // Without this, an edge like (Transform -> Frame N+1) is missing from
    // the canvas when Transform was added by walking out of Frame N -- the
    // walk only places parent->child edges, so the Transform's outgoing
    // edge to the next Frame stays invisible even though Frame N+1 is on
    // the canvas. Same problem for any cross-link added by the BFS that
    // isn't on the walk path. Edges already placed are skipped via the
    // placedEdges set in addEdgeBetween.
    for (const e of apiData.elements.edges) {
        const src = String(e.data.source);
        const dst = String(e.data.target);
        if (placedNodes.has(src) && placedNodes.has(dst)) {
            addEdgeBetween(src, dst);
        }
    }

    return elements;
}

// ---------------------------------------------------------------------------
// Collapse helpers
// ---------------------------------------------------------------------------

/** Find all non-frame descendant node IDs reachable from a given node via outgoing edges. */
export function getDescendants(nodeId: string, apiData: CytoscapeData): Set<string> {
    const outEdges = new Map<string, string[]>();
    for (const e of apiData.elements.edges) {
        const arr = outEdges.get(e.data.source) ?? [];
        arr.push(e.data.target);
        outEdges.set(e.data.source, arr);
    }

    const nodeTypes = new Map<string, string>();
    for (const n of apiData.elements.nodes) {
        nodeTypes.set(n.data.id, getNodeType(n.data));
    }

    const descendants = new Set<string>();
    const queue = [...(outEdges.get(nodeId) ?? [])];
    while (queue.length > 0) {
        const current = queue.shift()!;
        if (descendants.has(current)) continue;
        if (nodeTypes.get(current) === "frame") continue;
        descendants.add(current);
        for (const child of outEdges.get(current) ?? []) {
            queue.push(child);
        }
    }
    return descendants;
}

// ---------------------------------------------------------------------------
// Stylesheet
// ---------------------------------------------------------------------------

const BASE_NODE: cytoscape.Css.Node = {
    "background-color": "#ADB5BD",
    width: 32,
    height: 32,
    label: "data(_label)",
    "font-size": "10px",
    color: "#c8c8c8",
    "text-outline-color": "#1a1b1e",
    "text-outline-width": 1.5,
    "text-valign": "bottom",
    "text-halign": "center",
    "text-margin-y": 4,
} as cytoscape.Css.Node;

const STYLESHEET: cytoscape.StylesheetStyle[] = [
    // Base/fallback style MUST be first (Cytoscape applies in array order)
    { selector: "node", style: BASE_NODE },

    {
        selector: "node[_type='frame']",
        style: {
            "background-color": "#4C6EF5", shape: "round-rectangle",
            width: 80, height: 40, "font-size": "11px",
            "text-valign": "center", "text-halign": "center", "text-margin-y": 0,
        } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='object-instance']",
        style: { "background-color": "#82C91E", shape: "ellipse", width: 34, height: 34 } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='object']",
        style: { "background-color": "#40C057", shape: "ellipse", width: 34, height: 34 } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='feature-group']",
        style: { "background-color": "#FAB005", shape: "diamond", width: 28, height: 28 } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='relationship-group']",
        style: { "background-color": "#FD7E14", shape: "diamond", width: 28, height: 28 } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='action']",
        style: { "background-color": "#F03E3E", shape: "triangle", width: 32, height: 32 } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='feature-node']",
        style: { "background-color": "#ADB5BD", shape: "ellipse", width: 24, height: 24, "font-size": "9px" } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='intrinsic']",
        style: { "background-color": "#BE4BDB", shape: "ellipse", width: 26, height: 26, "font-size": "9px" } as cytoscape.Css.Node,
    },
    {
        selector: "node[_type='transform']",
        style: { "background-color": "#15AABF", shape: "round-rectangle", width: 28, height: 28, "font-size": "9px" } as cytoscape.Css.Node,
    },

    { selector: "node:selected", style: { "border-width": 3, "border-color": "#ffffff" } as cytoscape.Css.Node },
    { selector: "node:active", style: { "overlay-opacity": 0.1 } as cytoscape.Css.Node },
    {
        selector: "node.pinned",
        style: {
            "border-width": 2,
            "border-color": PINNED_BORDER_COLOR,
            "border-style": "dashed",
        } as cytoscape.Css.Node,
    },

    // Edges
    {
        selector: "edge",
        style: {
            width: 1.5, "line-color": "#495057", "target-arrow-color": "#495057",
            "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.7,
            label: "data(_label)", "font-size": "8px", color: "#636B74",
            "text-outline-color": "#1a1b1e", "text-outline-width": 1.5, "text-rotation": "autorotate",
        } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='next-frame']",
        style: { width: 2, "line-color": "#4C6EF5", "target-arrow-color": "#4C6EF5", "curve-style": "straight", "arrow-scale": 0.8, color: "#4C6EF5" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='situated']",
        style: { "line-color": "#82C91E", "target-arrow-color": "#82C91E", color: "#82C91E" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='observed-as']",
        style: { "line-color": "#40C057", "target-arrow-color": "#40C057", "line-style": "dashed", color: "#40C057" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='features']",
        style: { "line-color": "#FAB005", "target-arrow-color": "#FAB005", color: "#FAB005" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='relationships']",
        style: { "line-color": "#FD7E14", "target-arrow-color": "#FD7E14", color: "#FD7E14" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='detail']",
        style: { "line-color": "#868E96", "target-arrow-color": "#868E96", color: "#868E96" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='frame-attribute']",
        style: { "line-color": "#F03E3E", "target-arrow-color": "#F03E3E", color: "#F03E3E" } as cytoscape.Css.Edge,
    },
    {
        selector: "edge[_type='transform']",
        style: {
            "line-color": "#15AABF", "target-arrow-color": "#15AABF", "line-style": "dashed",
            "curve-style": "unbundled-bezier", color: "#15AABF",
            "control-point-distances": [40], "control-point-weights": [0.5],
        } as cytoscape.Css.Edge,
    },
    {
        selector: "edge:selected",
        style: { width: 2.5, "line-color": "#ffffff", "target-arrow-color": "#ffffff", color: "#ffffff" } as cytoscape.Css.Edge,
    },
];

// ---------------------------------------------------------------------------
// Detail Panel
// ---------------------------------------------------------------------------

interface DetailPanelProps {
    selected: SelectedElement;
    apiData: CytoscapeData;
    actionMap: ActionMapEntry[];
    onClose: () => void;
    onViewHistory?: (uuid: string) => void;
}

/** Right-side detail panel showing context-appropriate info for the selected node or edge. */
export function DetailPanel({ selected, apiData, actionMap, onClose, onViewHistory }: Readonly<DetailPanelProps>) {
    const nodeMap = useMemo(() => {
        const m = new Map<string, Record<string, unknown>>();
        for (const n of apiData.elements.nodes) {
            m.set(n.data.id, n.data as Record<string, unknown>);
        }
        return m;
    }, [apiData]);

    const title = (() => {
        if (selected.kind === "edge") return String(selected.data.type ?? "Edge");
        switch (selected.type) {
            case "frame": return `Frame ${selected.data.tick ?? ""}`;
            case "object-instance": {
                const ch = getInstanceChar(selected.data);
                return ch != null ? `ObjectInstance (${ch})` : "ObjectInstance";
            }
            case "object": {
                const ch = getObjectChar(
                    String(selected.data.id),
                    nodeMap,
                    apiData.elements.edges,
                );
                return ch != null ? `Object (${ch})` : "Object";
            }
            case "feature-group": return "FeatureGroup";
            case "relationship-group": return "RelationshipGroup";
            case "feature-node": {
                const { name, value } = getFeatureNodeDisplay(selected.data);
                return value ? `${name} (${value})` : name;
            }
            case "action": {
                const actionId = getActionId(selected.data);
                const name = lookupActionName(actionId, actionMap);
                if (name) return `Action (${name})`;
                return actionId != null ? `Action ${actionId}` : "Action";
            }
            case "intrinsic":
                return `Intrinsic (${selected.data.name ?? "?"})`;
            case "transform": {
                const labels = String(selected.data.labels ?? "");
                if (labels.includes("ObjectTransform")) return "ObjectTransform";
                if (labels.includes("IntrinsicTransform")) return "IntrinsicTransform";
                return "Transform";
            }
            default: return String(selected.data.labels ?? "Node");
        }
    })();

    return (
        <Paper
            p="xs"
            withBorder
            style={{
                borderColor: "#495057",
                backgroundColor: "#25262b",
                width: 280,
                flexShrink: 0,
                overflow: "hidden",
            }}
        >
            <Group justify="space-between" mb="xs">
                <Text fw={600} size="sm">{title}</Text>
                <CloseButton size="sm" onClick={onClose} aria-label="Close" />
            </Group>
            <ScrollArea h={380}>
                {selected.kind === "edge"
                    ? <EdgeDetailContent selected={selected} apiData={apiData} actionMap={actionMap} />
                    : <NodeDetailContent selected={selected} apiData={apiData} actionMap={actionMap} onViewHistory={onViewHistory} />}
            </ScrollArea>
        </Paper>
    );
}

function NodeDetailContent({
    selected,
    apiData,
    actionMap,
    onViewHistory,
}: Readonly<{ selected: SelectedElement; apiData: CytoscapeData; actionMap: ActionMapEntry[]; onViewHistory?: (uuid: string) => void }>) {
    switch (selected.type) {
        case "frame": return <FrameDetail nodeId={String(selected.data.id)} apiData={apiData} actionMap={actionMap} />;
        case "object-instance": return <ObjectInstanceDetail selected={selected} apiData={apiData} />;
        case "object": return <ObjectDetail data={selected.data} apiData={apiData} onViewHistory={onViewHistory} />;
        case "feature-node": return <FeatureNodeDetail data={selected.data} />;
        case "feature-group":
        case "relationship-group": return <GroupDetail nodeId={String(selected.data.id)} apiData={apiData} nodeType={selected.type} />;
        case "action": return <ActionDetail data={selected.data} actionMap={actionMap} />;
        case "intrinsic": return <IntrinsicDetail data={selected.data} />;
        case "transform": return <TransformDetail data={selected.data} apiData={apiData} />;
        case "property-transform": return <PropertyTransformDetail data={selected.data} />;
        default: return <RawDataDetail data={selected.data} />;
    }
}

function FrameDetail({
    nodeId,
    apiData,
    actionMap,
}: Readonly<{ nodeId: string; apiData: CytoscapeData; actionMap: ActionMapEntry[] }>) {
    const connected = findConnectedNodes(nodeId, apiData);
    const actionNode = connected.find((c) => getNodeType(c.data) === "action");
    const intrinsics = connected.filter((c) => getNodeType(c.data) === "intrinsic");
    const objects = connected.filter((c) => getNodeType(c.data) === "object-instance");

    const actionId = actionNode ? getActionId(actionNode.data) : undefined;
    const actionName = lookupActionName(actionId, actionMap) ?? String(actionId ?? "--");

    return (
        <Stack gap="xs">
            <Box>
                <Text size="xs" c="dimmed" mb={2}>Action taken</Text>
                <Badge color="red" variant="light" size="sm">{actionName}</Badge>
            </Box>
            {intrinsics.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Box>
                        <Text size="xs" c="dimmed" mb={4}>Intrinsics</Text>
                        <Stack gap={4}>
                            {intrinsics.map((intr) => {
                                const normalized = Number(intr.data.normalized_value ?? 0);
                                return (
                                    <Group key={String(intr.data.name)} gap="xs" justify="space-between">
                                        <Text size="xs" c="#868E96">{String(intr.data.name)}</Text>
                                        <Group gap={4}>
                                            <Progress
                                                value={normalized * 100}
                                                size="sm"
                                                w={80}
                                                color={normalized > 0.5 ? "green" : normalized > 0.25 ? "yellow" : "red"}
                                            />
                                            <Text size="xs" c="white" fw={500} w={30} ta="right">
                                                {String(intr.data.raw_value ?? "--")}
                                            </Text>
                                        </Group>
                                    </Group>
                                );
                            })}
                        </Stack>
                    </Box>
                </>
            )}
            {objects.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Box>
                        <Text size="xs" c="dimmed" mb={4}>Objects</Text>
                        <Stack gap={2}>
                            {objects.map((obj) => (
                                <Group key={String(obj.data.id)} gap="xs">
                                    <Text size="sm" ff="monospace">{getInstanceChar(obj.data) ?? "?"}</Text>
                                    <Text size="xs" c="dimmed">
                                        at ({String(obj.data.x ?? "?")}, {String(obj.data.y ?? "?")})
                                    </Text>
                                </Group>
                            ))}
                        </Stack>
                    </Box>
                </>
            )}
        </Stack>
    );
}

function ObjectInstanceDetail({
    selected,
    apiData,
}: Readonly<{ selected: SelectedElement; apiData: CytoscapeData }>) {
    const connected = findConnectedNodes(String(selected.data.id), apiData);
    const objectNode = connected.find((c) => getNodeType(c.data) === "object");
    const featureGroups = connected.filter((c) => getNodeType(c.data) === "feature-group");
    const relationshipGroups = connected.filter(
        (c) => getNodeType(c.data) === "relationship-group",
    );

    // Gather features from FeatureGroups -> FeatureNodes
    const features: Array<{ name: string; value: string }> = [];
    for (const fg of featureGroups) {
        const fgChildren = findConnectedNodes(String(fg.data.id), apiData);
        for (const child of fgChildren) {
            if (getNodeType(child.data) === "feature-node") {
                features.push(getFeatureNodeDisplay(child.data));
            }
        }
    }

    // Gather relationships from RelationshipGroups -> FeatureNodes
    const relationships: Array<{ name: string; value: string }> = [];
    for (const rg of relationshipGroups) {
        const rgChildren = findConnectedNodes(String(rg.data.id), apiData);
        for (const child of rgChildren) {
            if (getNodeType(child.data) === "feature-node") {
                relationships.push(getFeatureNodeDisplay(child.data));
            }
        }
    }

    // Also surface the intrinsic fields stored directly on ObjectInstance
    // (glyph_type, color_type, shape_type, etc.). These are the extracted
    // feature values the perception layer has already decoded.
    const instanceAttrs: Array<{ name: string; value: string }> = [];
    const d = selected.data;
    if (typeof d.glyph_type === "number") {
        instanceAttrs.push({ name: "glyph_type", value: String(d.glyph_type) });
    }
    if (typeof d.shape_type === "number") {
        instanceAttrs.push({ name: "shape", value: charFromCode(d.shape_type) });
    }
    if (typeof d.color_type === "number") {
        instanceAttrs.push({ name: "color", value: colorName(d.color_type) });
    }
    if (typeof d.flood_size === "number") {
        instanceAttrs.push({ name: "flood_size", value: String(d.flood_size) });
    }
    if (typeof d.line_size === "number") {
        instanceAttrs.push({ name: "line_size", value: String(d.line_size) });
    }
    if (typeof d.delta_old === "number" || typeof d.delta_new === "number") {
        instanceAttrs.push({
            name: "delta",
            value: `${String(d.delta_old ?? "?")} -> ${String(d.delta_new ?? "?")}`,
        });
    }
    if (d.motion_direction != null) {
        instanceAttrs.push({ name: "motion", value: String(d.motion_direction) });
    }
    if (typeof d.distance === "number") {
        instanceAttrs.push({ name: "distance", value: String(d.distance) });
    }

    const instanceChar = getInstanceChar(d) ?? "?";

    // Surface the parent Object's identity at the top of the instance
    // panel: human_name (display) + uuid (precision-safe). Walk the
    // ObservedAs edge to the Object and read its fields. Falls back to
    // the instance's own copies if the Object isn't loaded yet -- both
    // sides carry the same uuid/human_name from the server.
    const parentName =
        (objectNode && typeof objectNode.data.human_name === "string"
            ? objectNode.data.human_name
            : null) ?? (typeof d.human_name === "string" ? d.human_name : null);
    const parentUuid = (objectNode?.data.uuid ?? d.object_uuid) as unknown;

    return (
        <Stack gap="xs">
            <Group gap="xs">
                <Text size="lg" ff="monospace" fw={700}>{instanceChar}</Text>
                <Text size="xs" c="dimmed">
                    Position: ({String(selected.data.x ?? "?")}, {String(selected.data.y ?? "?")})
                </Text>
            </Group>
            {(parentName != null || parentUuid != null) && (
                <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                    <Table.Tbody>
                        {parentName != null && (
                            <Table.Tr>
                                <Table.Td style={{ color: "#868E96" }}>object</Table.Td>
                                <Table.Td style={{ color: "#e0e0e0" }}>{parentName}</Table.Td>
                            </Table.Tr>
                        )}
                        {parentUuid != null && (
                            <Table.Tr>
                                <Table.Td style={{ color: "#868E96" }}>uuid</Table.Td>
                                <Table.Td
                                    style={{
                                        color: "#e0e0e0",
                                        fontFamily: "monospace",
                                        fontSize: "11px",
                                        wordBreak: "break-all",
                                    }}
                                >
                                    {String(parentUuid)}
                                </Table.Td>
                            </Table.Tr>
                        )}
                    </Table.Tbody>
                </Table>
            )}
            {instanceAttrs.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Box>
                        <Text size="xs" c="dimmed" mb={4}>Attributes</Text>
                        <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                            <Table.Tbody>
                                {instanceAttrs.map((a, i) => (
                                    <Table.Tr key={`${a.name}-${i}`}>
                                        <Table.Td style={{ color: "#868E96" }}>{a.name}</Table.Td>
                                        <Table.Td style={{ color: "#e0e0e0" }}>{a.value}</Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>
                    </Box>
                </>
            )}
            {features.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Box>
                        <Text size="xs" c="dimmed" mb={4}>Features</Text>
                        <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                            <Table.Tbody>
                                {features.map((f, i) => (
                                    <Table.Tr key={`${f.name}-${i}`}>
                                        <Table.Td style={{ color: "#868E96" }}>{f.name}</Table.Td>
                                        <Table.Td style={{ color: "#e0e0e0" }}>{f.value}</Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>
                    </Box>
                </>
            )}
            {relationships.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Box>
                        <Text size="xs" c="dimmed" mb={4}>Relationships</Text>
                        <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                            <Table.Tbody>
                                {relationships.map((r, i) => (
                                    <Table.Tr key={`${r.name}-${i}`}>
                                        <Table.Td style={{ color: "#868E96" }}>{r.name}</Table.Td>
                                        <Table.Td style={{ color: "#e0e0e0" }}>{r.value}</Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>
                    </Box>
                </>
            )}
            {objectNode && (
                <>
                    <Divider color="#2C2E33" />
                    <Group gap="xs">
                        <Text size="xs" c="dimmed">Object UUID:</Text>
                        <Text size="xs" c="white">{String(objectNode.data.uuid ?? "--")}</Text>
                    </Group>
                    <Group gap="xs">
                        <Text size="xs" c="dimmed">Resolve count:</Text>
                        <Text size="xs" c="white">{String(objectNode.data.resolve_count ?? "--")}</Text>
                    </Group>
                </>
            )}
        </Stack>
    );
}

function ObjectDetail({
    data,
    apiData,
    onViewHistory,
}: Readonly<{
    data: Record<string, unknown>;
    apiData: CytoscapeData;
    onViewHistory?: (uuid: string) => void;
}>) {
    const nodeMap = useMemo(() => {
        const m = new Map<string, Record<string, unknown>>();
        for (const n of apiData.elements.nodes) {
            m.set(n.data.id, n.data as Record<string, unknown>);
        }
        return m;
    }, [apiData]);
    const glyphChar = getObjectChar(String(data.id), nodeMap, apiData.elements.edges);

    return (
        <Stack gap="xs">
            {glyphChar != null && (
                <Group gap="xs">
                    <Text size="lg" ff="monospace" fw={700}>{glyphChar}</Text>
                    {data.human_name != null && (
                        <Text size="xs" c="dimmed">{String(data.human_name)}</Text>
                    )}
                </Group>
            )}
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    {data.human_name != null && (
                        <Table.Tr>
                            <Table.Td style={{ color: "#868E96" }}>name</Table.Td>
                            <Table.Td style={{ color: "#e0e0e0" }}>{String(data.human_name)}</Table.Td>
                        </Table.Tr>
                    )}
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>UUID</Table.Td>
                        <Table.Td
                            style={{
                                color: "#e0e0e0",
                                fontFamily: "monospace",
                                fontSize: "11px",
                                wordBreak: "break-all",
                            }}
                        >
                            {/* uuid is a string for precision -- never coerce to Number */}
                            {data.uuid != null ? String(data.uuid) : "--"}
                        </Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>resolve count</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{String(data.resolve_count ?? "--")}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>last position</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>
                            ({String(data.last_x ?? "?")}, {String(data.last_y ?? "?")})
                        </Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>last tick</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{String(data.last_tick ?? "--")}</Table.Td>
                    </Table.Tr>
                </Table.Tbody>
            </Table>
            {onViewHistory && data.uuid != null && (
                <>
                    <Divider color="#2C2E33" />
                    <Button
                        size="xs"
                        variant="light"
                        color="cyan"
                        onClick={() => onViewHistory(String(data.uuid))}
                    >
                        View History
                    </Button>
                </>
            )}
        </Stack>
    );
}

function FeatureNodeDetail({ data }: Readonly<{ data: Record<string, unknown> }>) {
    const { name, value } = getFeatureNodeDisplay(data);
    // Show the raw feature subtype (e.g. "ColorNode, FeatureNode") so it's
    // obvious which decoder produced the display value.
    const labels = String(data.labels ?? "--");
    // Surface any raw numeric / string fields not already summarized.
    const summarizedKeys = new Set([
        "id", "labels", "_type", "_label",
        "name", "value", "kind", // legacy test fields already covered by getFeatureNodeDisplay
    ]);
    const extras = Object.entries(data).filter(([k]) => !summarizedKeys.has(k));
    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>type</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{name}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>value</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{value || "--"}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>labels</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{labels}</Table.Td>
                    </Table.Tr>
                    {data.kind != null && (
                        <Table.Tr>
                            <Table.Td style={{ color: "#868E96" }}>kind</Table.Td>
                            <Table.Td style={{ color: "#e0e0e0" }}>{String(data.kind)}</Table.Td>
                        </Table.Tr>
                    )}
                    {extras.map(([k, v]) => (
                        <Table.Tr key={k}>
                            <Table.Td style={{ color: "#868E96" }}>{k}</Table.Td>
                            <Table.Td style={{ color: "#e0e0e0" }}>{String(v ?? "--")}</Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </Stack>
    );
}

function GroupDetail({
    nodeId,
    apiData,
    nodeType,
}: Readonly<{ nodeId: string; apiData: CytoscapeData; nodeType: string }>) {
    const children = findConnectedNodes(nodeId, apiData);
    const featureNodes = children.filter((c) => getNodeType(c.data) === "feature-node");

    // FeatureGroups/RelationshipGroups on depth-1 frame loads have no
    // children yet. When that happens, the tap handler kicks off a fetch
    // for neighbors and this panel re-renders once the data arrives.
    const loading = featureNodes.length === 0;

    return (
        <Stack gap="xs">
            <Text size="xs" c="dimmed" mb={4}>
                {nodeType === "feature-group" ? "Features" : "Relationships"}
            </Text>
            {loading ? (
                <Text size="xs" c="dimmed">Loading...</Text>
            ) : (
                <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                    <Table.Tbody>
                        {featureNodes.map((f) => {
                            const { name, value } = getFeatureNodeDisplay(f.data);
                            return (
                                <Table.Tr key={String(f.data.id)}>
                                    <Table.Td style={{ color: "#868E96" }}>{name}</Table.Td>
                                    <Table.Td style={{ color: "#e0e0e0" }}>{value}</Table.Td>
                                </Table.Tr>
                            );
                        })}
                    </Table.Tbody>
                </Table>
            )}
        </Stack>
    );
}

function ActionDetail({
    data,
    actionMap,
}: Readonly<{ data: Record<string, unknown>; actionMap: ActionMapEntry[] }>) {
    const actionId = getActionId(data);
    const entry = actionId != null ? actionMap.find((e) => e.action_id === actionId) : undefined;
    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>action id</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{String(actionId ?? "--")}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>name</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{entry?.action_name ?? "--"}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>key</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0", fontFamily: "monospace" }}>
                            {entry?.action_key ?? "--"}
                        </Table.Td>
                    </Table.Tr>
                </Table.Tbody>
            </Table>
        </Stack>
    );
}

function IntrinsicDetail({ data }: Readonly<{ data: Record<string, unknown> }>) {
    const raw = data.raw_value;
    const normalized = data.normalized_value;
    const normalizedNum = typeof normalized === "number" ? normalized : null;
    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>name</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{String(data.name ?? "--")}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>raw value</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>{String(raw ?? "--")}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                        <Table.Td style={{ color: "#868E96" }}>normalized</Table.Td>
                        <Table.Td style={{ color: "#e0e0e0" }}>
                            {normalizedNum != null ? normalizedNum.toFixed(3) : "--"}
                        </Table.Td>
                    </Table.Tr>
                </Table.Tbody>
            </Table>
            {normalizedNum != null && (
                <Progress
                    value={normalizedNum * 100}
                    size="sm"
                    color={normalizedNum > 0.5 ? "green" : normalizedNum > 0.25 ? "yellow" : "red"}
                />
            )}
        </Stack>
    );
}

/** Walk Change edges from a Transform node to its source and destination
 *  Frame neighbors. Returns the (src_tick, dst_tick) pair, or null when
 *  either side isn't reachable in the loaded graph. */
export function findTransformFrameTicks(
    transformId: string,
    nodeMap: Map<string, Record<string, unknown>>,
    edges: ReadonlyArray<{ data: Record<string, unknown> }>,
): { srcFrameTick: number | null; dstFrameTick: number | null } {
    let srcFrameTick: number | null = null;
    let dstFrameTick: number | null = null;
    for (const e of edges) {
        if (String(e.data.type ?? "") !== "Change") continue;
        if (String(e.data.target) === transformId) {
            const frame = nodeMap.get(String(e.data.source));
            if (frame && typeof frame.tick === "number") srcFrameTick = frame.tick;
        }
        if (String(e.data.source) === transformId) {
            const frame = nodeMap.get(String(e.data.target));
            if (frame && typeof frame.tick === "number") dstFrameTick = frame.tick;
        }
    }
    return { srcFrameTick, dstFrameTick };
}

/** Each row of the per-property change table inside a Transform detail. */
export interface PropertyChangeRow {
    propertyName: string;
    changeType: string;
    summary: string;
}

/** Walk TransformDetail edges from an ObjectTransform to its
 *  PropertyTransformNode children and convert each into a display row.
 *
 *  PropertyTransformNode shapes vary by change type:
 *  - position changes (x/y): old/new are null, only `delta` is set
 *  - size/distance changes: old, new, and delta are all set
 *  - discrete changes (motion_direction, glyph_type, ...): old/new strings, no delta
 */
export function buildPropertyChangeRows(
    transformId: string,
    nodeMap: Map<string, Record<string, unknown>>,
    edges: ReadonlyArray<{ data: Record<string, unknown> }>,
): PropertyChangeRow[] {
    const rows: PropertyChangeRow[] = [];
    for (const e of edges) {
        if (String(e.data.type ?? "") !== "TransformDetail") continue;
        if (String(e.data.source) !== transformId) continue;
        const child = nodeMap.get(String(e.data.target));
        if (!child) continue;
        const labels = String(child.labels ?? "");
        if (!labels.includes("PropertyTransformNode")) continue;
        rows.push(propertyChangeRowFromNode(child));
    }
    // Stable order: continuous first (position before others), then discrete.
    // Inside each group, alphabetical by property name. This keeps the panel
    // readable when the same Transform is inspected across steps.
    const order = (r: PropertyChangeRow): number => {
        if (r.changeType === "continuous") {
            if (r.propertyName === "x") return 0;
            if (r.propertyName === "y") return 1;
            return 2;
        }
        return 3;
    };
    rows.sort((a, b) => {
        const oa = order(a);
        const ob = order(b);
        if (oa !== ob) return oa - ob;
        return a.propertyName.localeCompare(b.propertyName);
    });
    return rows;
}

function propertyChangeRowFromNode(
    child: Record<string, unknown>,
): PropertyChangeRow {
    const propertyName = String(child.property_name ?? "?");
    const changeType = String(child.change_type ?? "?");
    const oldVal = child.old_value;
    const newVal = child.new_value;
    const delta = child.delta;

    let summary: string;
    if (oldVal != null && newVal != null) {
        // Discrete-style: old -> new
        summary = `${formatPropValue(oldVal)} -> ${formatPropValue(newVal)}`;
        if (typeof delta === "number") summary += ` (${formatDelta(delta)})`;
    } else if (typeof delta === "number") {
        // Position/relative change: only delta is set
        summary = formatDelta(delta);
    } else {
        summary = `${formatPropValue(oldVal)} -> ${formatPropValue(newVal)}`;
    }
    return { propertyName, changeType, summary };
}

function formatPropValue(v: unknown): string {
    if (v == null) return "--";
    if (Array.isArray(v)) return `[${v.map(formatPropValue).join(",")}]`;
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
    return String(v);
}

function formatDelta(d: number): string {
    const sign = d > 0 ? "+" : "";
    const body = Number.isInteger(d) ? String(d) : d.toFixed(3);
    return `${sign}${body}`;
}

/** A child transform reachable from a bare parent Transform via an outgoing
 *  Change edge. Each entry summarises one ObjectTransform or IntrinsicTransform
 *  that the parent groups together for a single frame transition. */
export interface TransformChildSummary {
    id: string;
    /** "object" | "intrinsic" -- determines which fields are populated. */
    kind: "object" | "intrinsic" | "other";
    /** human_name from FlexiHumanHash for object transforms. */
    humanName?: string;
    /** Object UUID string for object transforms. */
    objectUuid?: string;
    /** Intrinsic name for intrinsic transforms. */
    intrinsicName?: string;
    /** Normalized change for intrinsic transforms. */
    normalizedChange?: number;
    /** Number of property changes summarised in the child node attributes. */
    changeCount?: number;
}

/** Walk outgoing Change edges from a bare parent Transform to its child
 *  ObjectTransform / IntrinsicTransform nodes. Used by TransformDetail to
 *  show a meaningful summary instead of a bare "kind: Transform" row.
 *
 *  This is the "container" half of the Transform multiplexing -- a parent
 *  Transform represents a frame transition and groups one child per object
 *  or intrinsic that changed. The Change edges to its children are
 *  distinct from the two Change edges that link it to the source/destination
 *  Frames. */
export function findTransformChildren(
    transformId: string,
    nodeMap: Map<string, Record<string, unknown>>,
    edges: ReadonlyArray<{ data: Record<string, unknown> }>,
): TransformChildSummary[] {
    const out: TransformChildSummary[] = [];
    for (const e of edges) {
        if (String(e.data.type ?? "") !== "Change") continue;
        if (String(e.data.source) !== transformId) continue;
        const child = nodeMap.get(String(e.data.target));
        if (!child) continue;
        const labels = String(child.labels ?? "");
        // Skip the parent->Frame Change edges (frames are not children).
        if (labels.includes("Frame")) continue;
        const id = String(child.id);
        if (labels.includes("ObjectTransform")) {
            const discrete = typeof child.num_discrete_changes === "number"
                ? child.num_discrete_changes : 0;
            const continuous = typeof child.num_continuous_changes === "number"
                ? child.num_continuous_changes : 0;
            out.push({
                id,
                kind: "object",
                humanName: typeof child.human_name === "string" ? child.human_name : undefined,
                objectUuid: child.object_uuid != null ? String(child.object_uuid) : undefined,
                changeCount: discrete + continuous,
            });
        } else if (labels.includes("IntrinsicTransform")) {
            out.push({
                id,
                kind: "intrinsic",
                intrinsicName: typeof child.name === "string" ? child.name : undefined,
                normalizedChange: typeof child.normalized_change === "number"
                    ? child.normalized_change : undefined,
            });
        } else {
            out.push({ id, kind: "other" });
        }
    }
    return out;
}

function TransformDetail({
    data,
    apiData,
}: Readonly<{ data: Record<string, unknown>; apiData: CytoscapeData }>) {
    const labels = String(data.labels ?? "");
    const kind = labels.includes("ObjectTransform")
        ? "ObjectTransform"
        : labels.includes("IntrinsicTransform")
            ? "IntrinsicTransform"
            : "Transform";

    const nodeMap = useMemo(() => {
        const m = new Map<string, Record<string, unknown>>();
        for (const n of apiData.elements.nodes) {
            m.set(n.data.id, n.data as Record<string, unknown>);
        }
        return m;
    }, [apiData]);

    const selfId = String(data.id);
    const { srcFrameTick, dstFrameTick } = findTransformFrameTicks(
        selfId,
        nodeMap,
        apiData.elements.edges,
    );
    const propertyRows = buildPropertyChangeRows(
        selfId,
        nodeMap,
        apiData.elements.edges,
    );
    // For a bare parent Transform (frame transition container), enumerate
    // its child ObjectTransform / IntrinsicTransform nodes via outgoing
    // Change edges. ObjectTransform / IntrinsicTransform leaves don't
    // have child transforms, so this list is always empty for them.
    const childTransforms = kind === "Transform"
        ? findTransformChildren(selfId, nodeMap, apiData.elements.edges)
        : [];

    // Header rows describe the transform itself; the per-property table
    // below it (for ObjectTransforms) shows the actual changes.
    const headerRows: Array<[string, string]> = [["kind", kind]];
    // human_name is the friendly display name; show it first when present.
    if (typeof data.human_name === "string") {
        headerRows.push(["object", data.human_name]);
    }
    // object_uuid arrives as a string from the wire format. Accept the
    // legacy number / bigint shapes too so this code keeps working if we
    // ever load an older graph fixture.
    if (data.object_uuid != null) {
        const ot = typeof data.object_uuid;
        if (ot === "string" || ot === "number" || ot === "bigint") {
            headerRows.push(["object uuid", String(data.object_uuid)]);
        }
    }
    if (typeof data.name === "string") {
        headerRows.push(["intrinsic", data.name]);
    }
    if (typeof data.normalized_change === "number") {
        headerRows.push(["normalized change", data.normalized_change.toFixed(3)]);
    }
    if (srcFrameTick != null) headerRows.push(["from frame", `tick ${srcFrameTick}`]);
    if (dstFrameTick != null) headerRows.push(["to frame", `tick ${dstFrameTick}`]);

    // Show the count rows only when we don't have the actual per-property
    // children loaded -- otherwise the children table makes the counts
    // redundant noise. The depth-1 frame fetch leaves PropertyTransformNodes
    // unfetched, so the counts give the user *some* signal until they click
    // the transform to expand it.
    if (propertyRows.length === 0) {
        if (typeof data.num_discrete_changes === "number") {
            headerRows.push(["discrete changes", String(data.num_discrete_changes)]);
        }
        if (typeof data.num_continuous_changes === "number") {
            headerRows.push(["continuous changes", String(data.num_continuous_changes)]);
        }
    }

    const totalAdvertised =
        (typeof data.num_discrete_changes === "number" ? data.num_discrete_changes : 0) +
        (typeof data.num_continuous_changes === "number" ? data.num_continuous_changes : 0);
    // PositionChange contributes 2 PropertyTransformNodes (x and y) but counts
    // as 1 continuous change, so node-count > advertised-count by up to 1.
    const childrenIncomplete =
        propertyRows.length > 0 &&
        totalAdvertised > 0 &&
        propertyRows.length < totalAdvertised;

    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    {headerRows.map(([k, v]) => (
                        <Table.Tr key={k}>
                            <Table.Td style={{ color: "#868E96" }}>{k}</Table.Td>
                            <Table.Td style={{ color: "#e0e0e0" }}>{v}</Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
            {propertyRows.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Text size="xs" c="dimmed">Property changes</Text>
                    <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th style={{ color: "#868E96" }}>property</Table.Th>
                                <Table.Th style={{ color: "#868E96" }}>type</Table.Th>
                                <Table.Th style={{ color: "#868E96" }}>change</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {propertyRows.map((row) => (
                                <Table.Tr key={row.propertyName}>
                                    <Table.Td style={{ color: "#e0e0e0" }}>
                                        {row.propertyName}
                                    </Table.Td>
                                    <Table.Td style={{ color: "#868E96" }}>
                                        {row.changeType}
                                    </Table.Td>
                                    <Table.Td
                                        style={{
                                            color: "#e0e0e0",
                                            fontFamily: "monospace",
                                        }}
                                    >
                                        {row.summary}
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                    {childrenIncomplete && (
                        <Text size="xs" c="dimmed">
                            Showing {propertyRows.length} of {totalAdvertised}
                            {" "}property changes -- expand the transform node
                            to load the rest.
                        </Text>
                    )}
                </>
            )}
            {childTransforms.length > 0 && (
                <>
                    <Divider color="#2C2E33" />
                    <Text size="xs" c="dimmed">
                        Frame transition with {childTransforms.length}
                        {childTransforms.length === 1 ? " change" : " changes"}
                    </Text>
                    <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th style={{ color: "#868E96" }}>kind</Table.Th>
                                <Table.Th style={{ color: "#868E96" }}>target</Table.Th>
                                <Table.Th style={{ color: "#868E96" }}>change</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {childTransforms.map((c) => (
                                <Table.Tr key={c.id}>
                                    <Table.Td style={{ color: "#868E96" }}>
                                        {c.kind}
                                    </Table.Td>
                                    <Table.Td style={{ color: "#e0e0e0" }}>
                                        {c.kind === "object"
                                            ? (c.humanName ?? c.objectUuid ?? c.id)
                                            : c.kind === "intrinsic"
                                                ? (c.intrinsicName ?? c.id)
                                                : c.id}
                                    </Table.Td>
                                    <Table.Td
                                        style={{
                                            color: "#e0e0e0",
                                            fontFamily: "monospace",
                                        }}
                                    >
                                        {c.kind === "object" && c.changeCount != null
                                            ? `${c.changeCount} prop${c.changeCount === 1 ? "" : "s"}`
                                            : c.kind === "intrinsic" && c.normalizedChange != null
                                                ? formatDelta(c.normalizedChange)
                                                : "--"}
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </>
            )}
        </Stack>
    );
}

function PropertyTransformDetail({
    data,
}: Readonly<{ data: Record<string, unknown> }>) {
    const propertyName = String(data.property_name ?? "?");
    const changeType = String(data.change_type ?? "?");
    const oldVal = data.old_value;
    const newVal = data.new_value;
    const delta = data.delta;

    const rows: Array<[string, string]> = [
        ["kind", "PropertyTransformNode"],
        ["property", propertyName],
        ["change type", changeType],
    ];
    if (oldVal != null) rows.push(["old value", formatPropDisplayValue(oldVal)]);
    if (newVal != null) rows.push(["new value", formatPropDisplayValue(newVal)]);
    if (typeof delta === "number") {
        rows.push(["delta", Number.isInteger(delta) ? String(delta) : delta.toFixed(3)]);
    }

    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    {rows.map(([k, v]) => (
                        <Table.Tr key={k}>
                            <Table.Td style={{ color: "#868E96" }}>{k}</Table.Td>
                            <Table.Td
                                style={{
                                    color: "#e0e0e0",
                                    fontFamily: "monospace",
                                }}
                            >
                                {v}
                            </Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </Stack>
    );
}

/** Display-format a PropertyTransformNode old/new value, which may be a
 *  string, number, array (delta pair), or null. */
function formatPropDisplayValue(v: unknown): string {
    if (v == null) return "--";
    if (Array.isArray(v)) return `[${v.map(formatPropDisplayValue).join(", ")}]`;
    if (typeof v === "number") {
        return Number.isInteger(v) ? String(v) : v.toFixed(3);
    }
    return String(v);
}

function RawDataDetail({ data }: Readonly<{ data: Record<string, unknown> }>) {
    const entries = Object.entries(data).filter(
        ([k]) => !["_type", "_label"].includes(k),
    );
    return (
        <Stack gap="xs">
            <Table verticalSpacing={2} horizontalSpacing="xs" style={{ fontSize: "12px" }}>
                <Table.Tbody>
                    {entries.map(([k, v]) => (
                        <Table.Tr key={k}>
                            <Table.Td style={{ color: "#868E96" }}>{k}</Table.Td>
                            <Table.Td style={{ color: "#e0e0e0" }}>{String(v ?? "--")}</Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </Stack>
    );
}

function EdgeDetailContent({
    selected,
    apiData,
    actionMap,
}: Readonly<{ selected: SelectedElement; apiData: CytoscapeData; actionMap: ActionMapEntry[] }>) {
    const nodeMap = new Map<string, Record<string, unknown>>();
    for (const n of apiData.elements.nodes) {
        nodeMap.set(n.data.id, n.data as Record<string, unknown>);
    }
    const labelContext: LabelContext = {
        nodeMap,
        allEdges: apiData.elements.edges,
    };

    const sourceData = nodeMap.get(String(selected.data.source));
    const targetData = nodeMap.get(String(selected.data.target));
    const sourceType = sourceData ? getNodeType(sourceData) : "unknown";
    const targetType = targetData ? getNodeType(targetData) : "unknown";

    return (
        <Stack gap="xs">
            <Group gap="xs">
                <Text size="xs" c="dimmed">Type:</Text>
                <Text size="xs" c="white">{String(selected.data.type ?? "--")}</Text>
            </Group>
            <Divider color="#2C2E33" />
            <Group gap="xs">
                <Text size="xs" c="dimmed">Source:</Text>
                <Text size="xs" c="white">
                    {makeLabel(sourceData ?? {}, sourceType, actionMap, labelContext)}
                </Text>
            </Group>
            <Group gap="xs">
                <Text size="xs" c="dimmed">Target:</Text>
                <Text size="xs" c="white">
                    {makeLabel(targetData ?? {}, targetType, actionMap, labelContext)}
                </Text>
            </Group>
        </Stack>
    );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface GraphVisualizationProps {
    run: string;
    step: number;
    game?: number;
}

function GraphVisualizationInner({ run, step, game }: Readonly<GraphVisualizationProps>) {
    const cyRef = useRef<cytoscape.Core | null>(null);
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
    const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null);
    const [accumulatedData, setAccumulatedData] = useState<CytoscapeData | null>(null);

    // Per-type visibility filter. Drives the "Show Nodes" toolbar dropdown.
    // Starts with every node type selected except Intrinsic -- intrinsics are
    // still reachable via the detail panel of their parent node.
    const [visibleNodeTypes, setVisibleNodeTypes] = useState<string[]>(
        () => [...DEFAULT_VISIBLE_NODE_TYPES],
    );
    const hiddenTypes = useMemo(() => {
        const set = new Set<string>();
        for (const opt of NODE_TYPE_OPTIONS) {
            if (!visibleNodeTypes.includes(opt.value)) {
                set.add(opt.value);
            }
        }
        return set;
    }, [visibleNodeTypes]);
    const toggleNodeType = useCallback((value: string) => {
        setVisibleNodeTypes((prev) =>
            prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value],
        );
    }, []);

    // Object history view mode
    // Object UUIDs are 63-bit ints stored as strings -- never coerce to
    // Number, that loses precision and View History fetches the wrong key.
    const [historyUuid, setHistoryUuid] = useState<string | null>(null);
    const isHistoryView = historyUuid != null;

    // Step debouncing is handled by the parent (App.tsx passes
    // ``useDebouncedValue(step, 200, run)``). The parent's hook resets
    // immediately on run change (via resetKey) so no stale cross-run
    // step can leak into the query. A second debounce layer here would
    // re-introduce the same stale-state-on-run-switch race it claims to
    // prevent, because React render-time setState does not update the
    // return value until the NEXT render -- long enough for useFrameGraph
    // to fire one request with (newRun, oldStep).

    // Depth 1: frame + direct neighbors. Deeper nodes fetched on expand.
    const { data, isLoading } = useFrameGraph(run, step, game, 1);
    const { data: actionMapData } = useActionMap(run);
    const actionMap = actionMapData ?? [];

    // Object history graph (only fetched when in history view)
    const { data: historyData, isLoading: historyLoading } = useObjectHistoryGraph(
        run,
        historyUuid,
    );

    // Refs for use in stable callbacks without causing re-creation
    const runRef = useRef(run);
    runRef.current = run;
    const accumulatedDataRef = useRef(accumulatedData);
    accumulatedDataRef.current = accumulatedData;
    const expandedNodesRef = useRef(expandedNodes);
    expandedNodesRef.current = expandedNodes;

    // User-pinned positions. Frames are pinned by the layout's
    // fixedNodeConstraint -- this map only tracks pins on non-frame nodes
    // (created by clicking or dragging). The position recorded here is
    // re-applied to the layout on every subsequent run via fixedNodeConstraint
    // so the node stays put even as the rest of the graph reflows.
    const pinnedNodesRef = useRef<Map<string, { x: number; y: number }>>(new Map());

    // Track which node IDs were rendered in the previous layout pass so the
    // imperative fcose effect can identify newly-added nodes and seed them
    // at a parent's position (rather than letting them appear at 0,0).
    const prevRenderedNodeIdsRef = useRef<Set<string>>(new Set());

    /** Pin a node at its current cytoscape position. Frames are skipped --
     *  they are already pinned by the layout's fixedNodeConstraint. */
    const pinNodeAtCurrentPosition = useCallback((nodeId: string) => {
        const cy = cyRef.current;
        if (!cy) return;
        const node = cy.getElementById(nodeId);
        if (node.length === 0) return;
        if (node.data("_type") === "frame") return;
        pinnedNodesRef.current.set(nodeId, {
            x: node.position("x"),
            y: node.position("y"),
        });
        node.addClass("pinned");
    }, []);

    /** Remove a single node's user pin. */
    const unpinNode = useCallback((nodeId: string) => {
        pinnedNodesRef.current.delete(nodeId);
        const cy = cyRef.current;
        if (!cy) return;
        const node = cy.getElementById(nodeId);
        if (node.length > 0) node.removeClass("pinned");
    }, []);

    /** Remove every user pin. Frame pins (the layout's fixedNodeConstraint)
     *  are unaffected. */
    const unpinAll = useCallback(() => {
        pinnedNodesRef.current.clear();
        const cy = cyRef.current;
        if (!cy) return;
        cy.nodes().removeClass("pinned");
    }, []);


    // DEBUG: sequence counter + timestamp for tracking the click flicker bug.
    // Every log line prints a monotonic counter so we can see event ordering
    // even when React batching or async timing shuffles the console output.
    //
    // Gated behind localStorage so the console stays quiet on a normal page
    // load (BUG-L2: this used to spam 25+ "[GraphViz]" lines per load). To
    // re-enable, run in DevTools:
    //     localStorage.setItem("graphviz-debug", "1")
    // and reload. Clear with localStorage.removeItem("graphviz-debug").
    const debugSeq = useRef(0);
    const dbg = useCallback((event: string, data: Record<string, unknown> = {}) => {
        if (!import.meta.env.DEV) return;
        try {
            if (globalThis.localStorage?.getItem("graphviz-debug") == null) return;
        } catch {
            return;
        }
        debugSeq.current += 1;
        const payload = { seq: debugSeq.current, t: performance.now().toFixed(1), event, ...data };
        // eslint-disable-next-line no-console
        console.log(`[GraphViz] ${JSON.stringify(payload)}`);
    }, []);

    // Viewport preservation: once the user zooms or pans, auto-fit stops
    // firing so their view isn't snapped back on every live step update.
    // Programmatic fit() calls set suppressViewportEvents first, so the
    // zoom/pan events they trigger don't count as user interaction.
    const userAdjustedViewport = useRef(false);
    const suppressViewportEvents = useRef(false);

    // Reset the user-adjusted flag when the run changes so a new run still
    // auto-fits on load. Step/frame changes within the same run preserve
    // the user's current zoom.
    useEffect(() => {
        userAdjustedViewport.current = false;
    }, [run]);

    const fitGraph = useCallback((force: boolean = false) => {
        const cy = cyRef.current;
        if (!cy) return;
        if (!force && userAdjustedViewport.current) {
            dbg("fitGraph:SKIPPED-userAdjusted");
            return;
        }
        dbg("fitGraph:FIRED", { force });
        suppressViewportEvents.current = true;
        cy.fit(undefined, 40);
        // Clear on the next tick so any synchronous zoom/pan events the
        // fit() call emits are ignored rather than flipping the flag.
        setTimeout(() => {
            suppressViewportEvents.current = false;
        }, 0);
        if (force) {
            userAdjustedViewport.current = false;
        }
    }, [dbg]);

    /** Run the cytoscape-fcose layout against the current cy state. Frames
     *  go in fixedNodeConstraint at evenly-spaced positions; user pins also
     *  go in fixedNodeConstraint at their stored positions. Animates from
     *  the current positions to the new ones. */
    const runFcoseLayout = useCallback(() => {
        const cy = cyRef.current;
        if (!cy || cy.nodes().length === 0) return;

        // Collect frame IDs in tick order from the live cy state
        const frames = cy.nodes("[_type='frame']")
            .sort((a, b) => Number(a.data("tick") ?? 0) - Number(b.data("tick") ?? 0));
        const frameIds: string[] = [];
        frames.forEach((n) => {
            frameIds.push(n.id());
        });

        // Pull user pins (filter out any whose node has since disappeared)
        const customPins: Array<{ nodeId: string; position: { x: number; y: number } }> = [];
        for (const [id, pos] of pinnedNodesRef.current.entries()) {
            if (cy.getElementById(id).length > 0) {
                customPins.push({ nodeId: id, position: pos });
            }
        }

        const layoutOpts = {
            name: "fcose",
            animate: "end",
            animationDuration: LAYOUT_ANIMATION_MS,
            animationEasing: "ease-out",
            fit: false,
            randomize: false,
            nodeRepulsion: () => NODE_REPULSION,
            idealEdgeLength: () => IDEAL_EDGE_LENGTH,
            nodeSeparation: NODE_SEPARATION,
            gravity: 0.25,
            numIter: LAYOUT_ITERATIONS,
            fixedNodeConstraint: [
                ...frameIds.map((id, i) => ({
                    nodeId: id,
                    position: { x: i * FRAME_X_GAP, y: FRAME_Y },
                })),
                ...customPins,
            ],
        } as unknown as cytoscape.LayoutOptions;

        const layout = cy.layout(layoutOpts);
        layout.one("layoutstop", () => fitGraph());
        layout.run();
    }, [fitGraph]);

    // When new frame data arrives (step change), reset accumulated data.
    //
    // IMPORTANT: skip the reset while the user is inspecting a selected
    // element. Otherwise, live frames arriving ~every 100ms clobber the
    // user's click state -- the detail panel and node expansion flicker
    // open/closed as frames arrive and async expand fetches resolve, and
    // the final state is non-deterministic. Pausing the graph refresh
    // while the user is engaged with a selection gives them a stable view
    // to inspect; the next frame's data flows in once they close it.
    const rootId = data?.meta.root_id;
    const prevRootId = useRef<number | null | undefined>(undefined);
    useEffect(() => {
        if (selectedElement != null) {
            if (rootId !== prevRootId.current) {
                dbg("rootId-reset:SKIPPED-selectedElement", {
                    prev: prevRootId.current,
                    next: rootId,
                });
            }
            return;
        }
        if (rootId !== prevRootId.current) {
            dbg("rootId-reset:FIRED", { prev: prevRootId.current, next: rootId });
            prevRootId.current = rootId;
            setAccumulatedData(data ?? null);
            setHistoryUuid(null);
            if (rootId != null) {
                setExpandedNodes(new Set([String(rootId)]));
            } else {
                setExpandedNodes(new Set());
            }
        }
    }, [rootId, data, selectedElement, dbg]);

    // Use history data when in history view, otherwise accumulated data
    const graphData = isHistoryView ? (historyData ?? null) : accumulatedData;

    // When the History view's data arrives, auto-expand every non-leaf
    // node so the entire curated subgraph is visible. The History subgraph
    // is curated server-side to be exactly the Frames the Object appeared
    // in, the linking ObjectInstances, and the Object itself. Without
    // expansion the user sees only the timeline backbone (171 lonely
    // Frame nodes for a long-lived object) because the buildElements walk
    // only renders children of expanded nodes -- and:
    //   - Frames must be expanded so their ObjectInstance children render.
    //   - ObjectInstances must be expanded so the Object node renders
    //     (Object is reachable via the ObservedAs out-edge from each
    //     instance). Without this the panel shows ObjectInstances but
    //     not the Object the user navigated to inspect.
    const historyAutoExpandedFor = useRef<string | null>(null);
    useEffect(() => {
        if (!isHistoryView) {
            historyAutoExpandedFor.current = null;
            return;
        }
        if (historyData == null) return;
        // Only auto-expand once per History session, so a user collapse
        // sticks. Re-runs only when historyUuid changes (new History view).
        if (historyAutoExpandedFor.current === historyUuid) return;
        historyAutoExpandedFor.current = historyUuid;
        // Expand every node that has any outgoing edge (i.e. anything
        // that's not a leaf in the curated history subgraph). This pulls
        // in Frames -> ObjectInstances -> Object in one shot.
        const expandable = new Set<string>();
        for (const n of historyData.elements.nodes) {
            expandable.add(String(n.data.id));
        }
        setExpandedNodes(expandable);
    }, [isHistoryView, historyData, historyUuid]);

    // Build positioned elements from graph data + expand state + action map.
    // The ``hiddenTypes`` set controls the "Show Nodes" filter from the toolbar.
    const elements = useMemo(() => {
        const result = graphData
            ? buildElements(graphData, expandedNodes, actionMap, hiddenTypes)
            : [];
        dbg("elements:rebuild", {
            nodeCount: result.filter((e) => e.group === "nodes").length,
            totalCount: result.length,
            graphDataNodeCount: graphData?.elements.nodes.length ?? 0,
            expandedSize: expandedNodes.size,
            hiddenTypes: [...hiddenTypes],
        });
        return result;
    }, [graphData, expandedNodes, actionMap, hiddenTypes, dbg]);

    // Warn when the rendered node count is large
    const nodeCount = useMemo(
        () => elements.filter((e) => e.group === "nodes").length,
        [elements],
    );
    const showNodeWarning = nodeCount > NODE_COUNT_WARNING_THRESHOLD;

    // Run the fcose layout whenever the elements set changes (a step
    // change, an expand/collapse, or a hidden-types toggle). Before running,
    // seed any newly-added nodes at a parent's current position so they
    // animate outward instead of materializing at (0, 0), and re-apply the
    // "pinned" class on nodes that survived the rebuild and are still in the
    // user pin map.
    useEffect(() => {
        const cy = cyRef.current;
        if (!cy || elements.length === 0) {
            prevRenderedNodeIdsRef.current = new Set();
            return;
        }

        // Identify newly-added nodes by diffing against the previous render.
        const currentNodeIds = new Set<string>();
        cy.nodes().forEach((n) => {
            currentNodeIds.add(n.id());
        });

        // Seed new nodes at a parent's current position. Skip frames --
        // the fixedNodeConstraint will place them precisely.
        currentNodeIds.forEach((id) => {
            if (prevRenderedNodeIdsRef.current.has(id)) return;
            const node = cy.getElementById(id);
            if (node.data("_type") === "frame") return;
            const incoming = node.incomers("node");
            for (let i = 0; i < incoming.length; i++) {
                const parent = incoming[i]!;
                if (prevRenderedNodeIdsRef.current.has(parent.id())) {
                    node.position(parent.position());
                    break;
                }
            }
        });

        // Restore pinned class on nodes that survived the rebuild and are
        // still user-pinned. (cy.json() preserves classes for unchanged
        // nodes, but a re-added node loses them.)
        for (const id of pinnedNodesRef.current.keys()) {
            const node = cy.getElementById(id);
            if (node.length > 0) node.addClass("pinned");
        }

        prevRenderedNodeIdsRef.current = currentNodeIds;

        runFcoseLayout();
    }, [elements, runFcoseLayout]);

    // Re-fit when detail panel opens/closes (container width changes).
    // Also respects userAdjustedViewport so the user's zoom survives the
    // panel toggle.
    const panelOpen = selectedElement != null;
    useEffect(() => {
        const timer = setTimeout(() => fitGraph(), 100);
        return () => clearTimeout(timer);
    }, [panelOpen, fitGraph]);

    const toggleExpand = useCallback((nodeId: string) => {
        setExpandedNodes((prev) => {
            const wasIn = prev.has(nodeId);
            const next = new Set(prev);
            let descendantsRemoved = 0;
            if (wasIn) {
                // Collapse: remove this node AND all graph descendants
                next.delete(nodeId);
                const currentData = accumulatedDataRef.current;
                if (currentData) {
                    for (const d of getDescendants(nodeId, currentData)) {
                        next.delete(d);
                        descendantsRemoved += 1;
                    }
                }
            } else {
                next.add(nodeId);
            }
            dbg("toggleExpand", {
                nodeId,
                action: wasIn ? "collapse" : "expand",
                prevSize: prev.size,
                nextSize: next.size,
                descendantsRemoved,
            });
            return next;
        });
    }, [dbg]);

    const handleCyInit = useCallback(
        (cy: cytoscape.Core) => {
            // CRITICAL: react-cytoscapejs calls the `cy` prop callback on
            // EVERY React update (see its updateCytoscape), not just on
            // mount. Without this guard, every re-render of this component
            // attaches another full set of event listeners to the same
            // cytoscape instance. After N re-renders, a single click fires
            // the tap handler N times, starting N concurrent fetches whose
            // toggleExpand calls oscillate the graph between expanded and
            // collapsed for seconds -- the click-flicker bug. We bail out
            // if the cy instance we're setting up is the same one we
            // already initialized; cytoscape reuses the same instance for
            // the component's lifetime (it destroys and recreates only on
            // unmount/remount).
            if (cyRef.current === cy) return;
            cyRef.current = cy;
            // Expose to window in dev mode so Playwright tests can drive
            // cytoscape's API directly (window.__cy.nodes()[0].emit('tap')).
            // Canvas-rendered nodes are unreachable via normal DOM selectors,
            // so this escape hatch is the only reliable way to simulate
            // real user interaction in tests. DEV-only; stripped in prod builds.
            if (import.meta.env.DEV) {
                (window as unknown as { __cy?: cytoscape.Core }).__cy = cy;
            }

            // Track user zoom/pan so auto-fit effects can skip themselves
            // once the user has taken control of the viewport. Programmatic
            // fit() calls set suppressViewportEvents first, so only genuine
            // user interaction flips the userAdjustedViewport flag.
            cy.on("zoom pan", () => {
                if (!suppressViewportEvents.current) {
                    userAdjustedViewport.current = true;
                }
            });

            cy.on("tap", "node", (evt) => {
                const nodeData = evt.target.data() as Record<string, unknown>;
                const nodeType = nodeData._type as string;
                const nodeId = nodeData.id as string;

                // Shift-click = unpin only. Does not toggle expand or open
                // the detail panel; it's a quick "release this node" gesture.
                const native = evt.originalEvent as MouseEvent | undefined;
                if (native && native.shiftKey) {
                    dbg("tap:node:unpin", { nodeId });
                    unpinNode(nodeId);
                    // Re-run layout so the released node can rejoin the
                    // force-directed flow.
                    runFcoseLayout();
                    return;
                }

                const isExpanded = expandedNodesRef.current.has(nodeId);
                dbg("tap:node", {
                    nodeId,
                    nodeType,
                    isExpanded,
                    expandedSize: expandedNodesRef.current.size,
                });

                // Pin at current position before the layout reshuffles
                // anything. Frames are skipped inside the helper.
                pinNodeAtCurrentPosition(nodeId);

                // Show detail panel immediately
                setSelectedElement({ kind: "node", type: nodeType, data: nodeData });

                if (!isExpanded) {
                    // Fetch neighbors, merge, then expand
                    dbg("fetch:start", { nodeId });
                    fetchNodeGraph(runRef.current, nodeId, 1)
                        .then((neighbors) => {
                            dbg("fetch:then", {
                                nodeId,
                                nodeCount: neighbors?.elements?.nodes?.length ?? 0,
                                edgeCount: neighbors?.elements?.edges?.length ?? 0,
                            });
                            setAccumulatedData((prev) => {
                                const merged = prev ? mergeGraphData(prev, neighbors) : neighbors;
                                dbg("setAccumulatedData", {
                                    cause: "fetch:then",
                                    nodeId,
                                    prevNodes: prev?.elements.nodes.length ?? 0,
                                    mergedNodes: merged?.elements.nodes.length ?? 0,
                                });
                                return merged;
                            });
                        })
                        .catch((err) => {
                            console.error("[GraphViz] node expand failed:", err);
                            dbg("fetch:error", { nodeId, err: String(err) });
                        })
                        .finally(() => {
                            dbg("fetch:finally", { nodeId });
                            toggleExpand(nodeId);
                        });
                } else {
                    toggleExpand(nodeId);
                }
            });

            cy.on("tap", "edge", (evt) => {
                const edgeData = evt.target.data() as Record<string, unknown>;
                const edgeType = edgeData._type as string;
                setSelectedElement({ kind: "edge", type: edgeType, data: edgeData });
            });

            // Pin a node wherever the user drags it. dragfree fires once
            // when a drag ends; this avoids spamming pins during the drag.
            cy.on("dragfree", "node", (evt) => {
                const nodeId = evt.target.id() as string;
                pinNodeAtCurrentPosition(nodeId);
            });

            // Background click clears selection
            cy.on("tap", (evt) => {
                if (evt.target === cy) {
                    setSelectedElement(null);
                }
            });

            // Initial fit goes through fitGraph so the programmatic zoom
            // event it emits is suppressed and doesn't flip the user-adjusted
            // flag on startup.
            fitGraph();
        },
        [toggleExpand, fitGraph, pinNodeAtCurrentPosition, unpinNode, runFcoseLayout, dbg],
    );

    const handleViewHistory = useCallback((uuid: string) => {
        setHistoryUuid(uuid);
        setSelectedElement(null);
        // The History view returns the Object's full lifetime: every Frame
        // it appeared in, plus the ObjectInstances linking the two. The
        // buildElements walk only renders children of *expanded* frames,
        // so leaving expandedNodes empty here would show only the timeline
        // backbone (171 lonely Frame nodes for a long-lived object) and
        // hide the very ObjectInstances and Object the user came to see.
        // The actual expansion is populated by an effect once historyData
        // arrives -- this just clears stale state from the frame view.
        setExpandedNodes(new Set());
    }, []);

    const handleBackToFrame = useCallback(() => {
        setHistoryUuid(null);
        setSelectedElement(null);
        setAccumulatedData(data ?? null);
        if (data?.meta.root_id != null) {
            setExpandedNodes(new Set([String(data.meta.root_id)]));
        } else {
            setExpandedNodes(new Set());
        }
    }, [data]);

    if (isHistoryView && historyLoading) {
        return <Text size="xs" c="dimmed">Loading object history...</Text>;
    }

    if (isLoading && !data) {
        return <Text size="xs" c="dimmed">Loading graph data...</Text>;
    }

    if (!isHistoryView && !graphData?.elements.nodes.length && !data?.elements.nodes.length) {
        return <Text size="xs" c="dimmed">No graph data</Text>;
    }

    return (
        <Paper
            withBorder
            style={{
                backgroundColor: "#141517",
                borderColor: "#2C2E33",
                height: 450,
                position: "relative",
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
            }}
        >
            {/* Toolbar: Fit View, Show Nodes filter, Back to Frame */}
            <Group gap="xs" p={4} style={{ borderBottom: "1px solid #2C2E33", flexShrink: 0 }}>
                <Button
                    size="compact-xs"
                    variant="light"
                    color="gray"
                    onClick={() => fitGraph(true)}
                    title="Reset zoom and fit the entire graph in view"
                >
                    Fit View
                </Button>
                <Button
                    size="compact-xs"
                    variant="light"
                    color="gray"
                    onClick={() => {
                        unpinAll();
                        // Re-run the layout so released nodes can rejoin the
                        // force-directed flow.
                        runFcoseLayout();
                    }}
                    title="Release every user-pinned node (frames stay pinned)"
                >
                    Unpin All
                </Button>
                <Popover position="bottom-start" shadow="md" withArrow>
                    <Popover.Target>
                        <Button
                            size="compact-xs"
                            variant="light"
                            color="gray"
                            title="Toggle which node types are rendered in the graph"
                        >
                            Show Nodes ({visibleNodeTypes.length}/{NODE_TYPE_OPTIONS.length})
                        </Button>
                    </Popover.Target>
                    <Popover.Dropdown p="xs">
                        <Stack gap={4}>
                            {NODE_TYPE_OPTIONS.map((opt) => (
                                <Checkbox
                                    key={opt.value}
                                    size="xs"
                                    label={opt.label}
                                    checked={visibleNodeTypes.includes(opt.value)}
                                    onChange={() => toggleNodeType(opt.value)}
                                />
                            ))}
                        </Stack>
                    </Popover.Dropdown>
                </Popover>
                {isHistoryView && (
                    <Button size="compact-xs" variant="light" color="gray" onClick={handleBackToFrame}>
                        Back to Frame
                    </Button>
                )}
                {isHistoryView && (
                    <Text size="xs" c="dimmed">Object History (UUID: {historyUuid})</Text>
                )}
            </Group>
            {showNodeWarning && (
                <Alert variant="light" color="yellow" p={4} style={{ flexShrink: 0, fontSize: "11px" }}>
                    Large graph ({nodeCount} nodes) -- consider reducing depth or collapsing subtrees for better performance.
                </Alert>
            )}
            <div style={{ flex: 1, minWidth: 0, display: "flex" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <CytoscapeComponent
                        elements={elements}
                        stylesheet={STYLESHEET}
                        layout={PRESET_LAYOUT}
                        cy={handleCyInit}
                        style={{ width: "100%", height: "100%" }}
                        userPanningEnabled={true}
                        userZoomingEnabled={true}
                        boxSelectionEnabled={false}
                    />
                </div>
                {selectedElement && graphData && (
                    <DetailPanel
                        selected={selectedElement}
                        apiData={graphData}
                        actionMap={actionMap}
                        onClose={() => setSelectedElement(null)}
                        onViewHistory={handleViewHistory}
                    />
                )}
            </div>
        </Paper>
    );
}

// GraphVisualization is the heaviest panel in the dashboard (Cytoscape.js
// graph layout, ~2500 LOC, expensive re-renders). On step navigation the
// parent App re-renders and passes a new ``step`` prop on every keystroke,
// which cascades into a useless inner re-render of the whole Cytoscape
// subtree before the internal debounce even decides whether to refetch.
// Wrap the component in ``React.memo`` so props equality ends the
// reconciliation at this boundary. The component still reacts to step
// changes via its internal ``debouncedStep`` effect; memo just removes the
// extra work when props haven't actually shifted.
export const GraphVisualization = memo(GraphVisualizationInner);
