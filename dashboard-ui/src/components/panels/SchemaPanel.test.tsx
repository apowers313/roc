import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { GraphSchema, SchemaEdge, SchemaNode } from "../../api/client";
import { renderWithProviders } from "../../test-utils";
import { SchemaPanel } from "./SchemaPanel";

vi.mock("../../api/queries", () => ({
    useSchema: vi.fn(),
}));

// Mock DiagramViewer so we don't pull in mermaid / react-zoom-pan-pinch
vi.mock("../common/DiagramViewer", () => ({
    DiagramViewer: ({ definition }: { definition: string }) => (
        <div data-testid="diagram-viewer">{definition}</div>
    ),
}));

import { useSchema } from "../../api/queries";

function mockSchema(
    data: GraphSchema | undefined,
    opts: { isLoading?: boolean; error?: Error | null } = {},
) {
    vi.mocked(useSchema).mockReturnValue({
        data,
        isLoading: opts.isLoading ?? false,
        error: opts.error ?? null,
    } as ReturnType<typeof useSchema>);
}

const FIELD_LOCAL = { name: "hp", type: "int", default: "100", local: true, exclude: false };
const FIELD_INHERITED = { name: "id", type: "str", default: null, local: false, exclude: false };
const FIELD_EXCLUDED = { name: "secret", type: "str", default: null, local: true, exclude: true };

const METHOD_LOCAL = { name: "update", params: "self, val", return_type: "None", local: true };
const METHOD_INHERITED = { name: "save", params: "self", return_type: "bool", local: false };

const NODE_SIMPLE: SchemaNode = {
    name: "Intrinsic",
    parents: [],
    fields: [FIELD_LOCAL],
    methods: [],
};

const NODE_WITH_PARENTS: SchemaNode = {
    name: "HPIntrinsic",
    parents: ["Intrinsic", "Node"],
    fields: [FIELD_LOCAL, FIELD_INHERITED, FIELD_EXCLUDED],
    methods: [METHOD_LOCAL, METHOD_INHERITED],
};

const EDGE_SIMPLE: SchemaEdge = {
    name: "HAS_INTRINSIC",
    type: "HAS_INTRINSIC",
    connections: [["Frame", "Intrinsic"]],
    fields: [],
};

const EDGE_WITH_FIELDS: SchemaEdge = {
    name: "CONNECTS",
    type: "LINK",
    connections: [
        ["A", "B"],
        ["C", "D"],
    ],
    fields: [
        { name: "weight", type: "float", default: "1.0", local: true, exclude: false },
        { name: "hidden", type: "str", default: null, local: true, exclude: true },
    ],
};

const FULL_SCHEMA: GraphSchema = {
    mermaid: "classDiagram\nclass Intrinsic",
    nodes: [NODE_SIMPLE, NODE_WITH_PARENTS],
    edges: [EDGE_SIMPLE, EDGE_WITH_FIELDS],
};

describe("SchemaPanel", () => {
    it("shows loading state", () => {
        mockSchema(undefined, { isLoading: true });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("Loading schema...")).toBeInTheDocument();
    });

    it("shows error message when schema is null", () => {
        mockSchema(undefined);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("No schema available for this run")).toBeInTheDocument();
    });

    it("shows error message when query errors", () => {
        mockSchema(undefined, { error: new Error("fetch failed") });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("No schema available for this run")).toBeInTheDocument();
    });

    it("renders summary badges with correct counts", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("2 Nodes")).toBeInTheDocument();
        expect(screen.getByText("2 Edges")).toBeInTheDocument();
        // total connections: 1 + 2 = 3
        expect(screen.getByText("3 Connections")).toBeInTheDocument();
    });

    it("renders the DiagramViewer when mermaid text is present", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByTestId("diagram-viewer")).toBeInTheDocument();
        expect(screen.getByTestId("diagram-viewer").textContent).toBe(
            "classDiagram\nclass Intrinsic",
        );
    });

    it("does not render DiagramViewer when mermaid is empty", () => {
        mockSchema({ ...FULL_SCHEMA, mermaid: "" });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.queryByTestId("diagram-viewer")).not.toBeInTheDocument();
    });

    it("renders node names in accordion controls", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        // "Intrinsic" appears both as a node name and in "extends Intrinsic, Node"
        expect(screen.getAllByText("Intrinsic").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("HPIntrinsic")).toBeInTheDocument();
    });

    it("renders 'extends' text for nodes with parents", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("extends Intrinsic, Node")).toBeInTheDocument();
    });

    it("does not render 'extends' for nodes without parents", () => {
        mockSchema({ ...FULL_SCHEMA, nodes: [NODE_SIMPLE], edges: [] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.queryByText(/extends/)).not.toBeInTheDocument();
    });

    it("renders field count badge excluding excluded fields", () => {
        mockSchema({ ...FULL_SCHEMA, nodes: [NODE_WITH_PARENTS], edges: [] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        // NODE_WITH_PARENTS has 3 fields, 1 excluded, 1 local non-excluded = "1 fields" badge
        expect(screen.getByText("1 fields")).toBeInTheDocument();
    });

    it("renders method count badge for nodes with local methods", () => {
        mockSchema({ ...FULL_SCHEMA, nodes: [NODE_WITH_PARENTS], edges: [] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("1 methods")).toBeInTheDocument();
    });

    it("does not render method badge when no local methods", () => {
        const nodeNoLocalMethods: SchemaNode = {
            ...NODE_SIMPLE,
            methods: [METHOD_INHERITED],
        };
        mockSchema({ ...FULL_SCHEMA, nodes: [nodeNoLocalMethods], edges: [] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.queryByText(/methods/)).not.toBeInTheDocument();
    });

    it("renders edge names in accordion controls", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("HAS_INTRINSIC")).toBeInTheDocument();
        expect(screen.getByText("CONNECTS")).toBeInTheDocument();
    });

    it("shows edge type when different from name", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        // EDGE_WITH_FIELDS has name=CONNECTS, type=LINK
        expect(screen.getByText("type: LINK")).toBeInTheDocument();
    });

    it("does not show edge type when same as name", () => {
        mockSchema({ ...FULL_SCHEMA, edges: [EDGE_SIMPLE] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.queryByText(/type:/)).not.toBeInTheDocument();
    });

    it("renders connection count badge on edges", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("1 connections")).toBeInTheDocument();
        expect(screen.getByText("2 connections")).toBeInTheDocument();
    });

    it("renders schema with no edges", () => {
        mockSchema({ mermaid: "", nodes: [NODE_SIMPLE], edges: [] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("Intrinsic")).toBeInTheDocument();
        expect(screen.getByText("0 Edges")).toBeInTheDocument();
    });

    it("renders schema with no nodes", () => {
        mockSchema({ mermaid: "", nodes: [], edges: [EDGE_SIMPLE] });
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("0 Nodes")).toBeInTheDocument();
        expect(screen.getByText("HAS_INTRINSIC")).toBeInTheDocument();
    });

    it("renders Summary heading", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("Summary")).toBeInTheDocument();
    });

    it("renders Nodes and Edges section headings", () => {
        mockSchema(FULL_SCHEMA);
        renderWithProviders(<SchemaPanel run="test-run" />);
        expect(screen.getByText("Nodes")).toBeInTheDocument();
        expect(screen.getByText("Edges")).toBeInTheDocument();
    });
});
