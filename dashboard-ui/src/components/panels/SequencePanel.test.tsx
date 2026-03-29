import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { SequencePanel } from "./SequencePanel";

// SequencePanel uses ObjectLink which imports ObjectModal -> useObjectHistory
vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn().mockReturnValue({
        data: { info: { uuid: 1, resolve_count: 0 }, states: [], transforms: [] },
        isLoading: false,
    }),
}));

describe("SequencePanel", () => {
    it("shows empty state when data is undefined", () => {
        renderWithProviders(<SequencePanel data={undefined} />);
        expect(screen.getByText("No sequence data")).toBeInTheDocument();
    });

    it("displays all objects in frame as table rows", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 7,
                object_count: 3,
                objects: [
                    { id: "1", glyph: "@", color: 15, x: 10, y: 5, matched_previous: true, resolve_count: 42 },
                    { id: "2", glyph: "d", color: 1, x: 15, y: 8, matched_previous: true, resolve_count: 7 },
                    { id: "3", glyph: ".", color: 7, x: 22, y: 1, matched_previous: false, resolve_count: 1 },
                ],
                intrinsic_count: 1,
                intrinsics: { hp: 0.8 },
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        // 1 obj header + 3 objs + 1 intr header + 1 intr = 6 rows
        expect(screen.getAllByRole("row")).toHaveLength(6);
    });

    it("shows resolve count column for objects", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [
                    { id: "1", glyph: "@", color: 15, x: 10, y: 5, matched_previous: true, resolve_count: 42 },
                ],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("Resolves")).toBeInTheDocument();
        expect(screen.getByText("42")).toBeInTheDocument();
    });

    it("renders position as combined (x,y) format", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [
                    { id: "1", glyph: "@", color: 15, x: 10, y: 5, matched_previous: true, resolve_count: 1 },
                ],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("(10, 5)")).toBeInTheDocument();
    });

    it("displays shape column", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [
                    { id: "1", glyph: "@", color: 15, shape: 64, x: 10, y: 5, matched_previous: true, resolve_count: 1 },
                ],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("Shape")).toBeInTheDocument();
        expect(screen.getByText("64")).toBeInTheDocument();
    });

    it("shows intrinsics as textual rows with Raw, Normalized, Matched columns", () => {
        const data = makeStepData({
            intrinsics: { raw: { hp: 15, energy: 10 }, normalized: { hp: 0.8, energy: 0.5 } },
            sequence_summary: {
                tick: 1,
                object_count: 0,
                objects: [],
                intrinsic_count: 2,
                intrinsics: { hp: 0.8, energy: 0.5 },
            },
            transform_summary: {
                count: 1,
                changes: [{ description: "hp changed", type: "intrinsic", name: "hp", normalized_change: -0.05 }],
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        // Column headers
        expect(screen.getByText("Name")).toBeInTheDocument();
        expect(screen.getByText("Raw")).toBeInTheDocument();
        expect(screen.getByText("Normalized")).toBeInTheDocument();
        // Intrinsic names
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("energy")).toBeInTheDocument();
        // Raw values
        expect(screen.getByText("15")).toBeInTheDocument();
        expect(screen.getByText("10")).toBeInTheDocument();
        // Normalized values
        expect(screen.getByText("0.8000")).toBeInTheDocument();
        expect(screen.getByText("0.5000")).toBeInTheDocument();
        // Matched: hp was in transform_summary.changes, energy was not
        const yesElements = screen.getAllByText("Yes");
        const noElements = screen.getAllByText("No");
        expect(yesElements.length).toBeGreaterThanOrEqual(1);
        expect(noElements.length).toBeGreaterThanOrEqual(1);
        // No progress bar elements
        expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    });

    it("shows frame counts in header as single-line format", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 7,
                object_count: 3,
                objects: [
                    { id: "1", glyph: "@", color: 15, x: 10, y: 5, matched_previous: true, resolve_count: 1 },
                    { id: "2", glyph: "d", color: 1, x: 15, y: 8, matched_previous: true, resolve_count: 1 },
                    { id: "3", glyph: ".", color: 7, x: 22, y: 1, matched_previous: false, resolve_count: 1 },
                ],
                intrinsic_count: 4,
                intrinsics: { hp: 0.8, energy: 0.5, ac: 0.9, gold: 0.1 },
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        // Single-line header: "Frame: tick=7 | 3 objects | 4 intrinsics"
        expect(screen.getByText(/Frame: tick=7/)).toBeInTheDocument();
        expect(screen.getByText(/3 objects/)).toBeInTheDocument();
        expect(screen.getByText(/4 intrinsics/)).toBeInTheDocument();
    });

    it("shows -- for raw values when intrinsics data not available", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 0,
                objects: [],
                intrinsic_count: 1,
                intrinsics: { hp: 0.8 },
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("0.8000")).toBeInTheDocument();
        // Raw should show "--" since intrinsics.raw is not provided
        expect(screen.getAllByText("--").length).toBeGreaterThanOrEqual(1);
    });
});
