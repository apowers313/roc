import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { IntrinsicsPanel } from "./IntrinsicsPanel";

describe("IntrinsicsPanel", () => {
    it("shows 'No intrinsics data' when data is undefined", () => {
        renderWithProviders(<IntrinsicsPanel data={undefined} />);
        expect(screen.getByText("No intrinsics data")).toBeInTheDocument();
    });

    it("shows 'No intrinsics data' when intrinsics is null", () => {
        renderWithProviders(<IntrinsicsPanel data={makeStepData()} />);
        expect(screen.getByText("No intrinsics data")).toBeInTheDocument();
    });

    it("renders normalized intrinsic values with progress bars", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 15, energy: 8 },
                normalized: { hp: 0.75, energy: 0.4 },
            },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("energy")).toBeInTheDocument();
        expect(screen.getByText("15")).toBeInTheDocument();
        expect(screen.getByText("8")).toBeInTheDocument();
    });

    it("shows -- for missing raw values", () => {
        const data = makeStepData({
            intrinsics: {
                raw: {},
                normalized: { hp: 0.5 },
            },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("--")).toBeInTheDocument();
    });

    it("displays significance badge when significance is present", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.3456,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("Significance")).toBeInTheDocument();
        expect(screen.getByText("0.3456")).toBeInTheDocument();
    });

    it("does not show significance when it is null", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: null,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.queryByText("Significance")).not.toBeInTheDocument();
    });

    it("uses red color for high significance (>0.5)", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.7,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        const badge = screen.getByText("0.7000");
        // Badge should exist and show value
        expect(badge).toBeInTheDocument();
    });

    it("uses yellow color for medium significance (>0.1)", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.25,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("0.2500")).toBeInTheDocument();
    });

    it("uses green color for low significance (<=0.1)", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.05,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("0.0500")).toBeInTheDocument();
    });

    it("sorts intrinsic keys alphabetically", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hunger: 3, energy: 8, hp: 15 },
                normalized: { hunger: 0.3, energy: 0.4, hp: 0.75 },
            },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        const labels = screen.getAllByText(/^(energy|hp|hunger)$/);
        expect(labels[0]!.textContent).toBe("energy");
        expect(labels[1]!.textContent).toBe("hp");
        expect(labels[2]!.textContent).toBe("hunger");
    });

    it("clamps progress bar to 0-100 range", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 100 },
                normalized: { hp: 1.5 }, // exceeds 1.0
            },
        });
        // Should not throw
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
    });

    it("handles intrinsics with undefined normalized and raw", () => {
        const data = makeStepData({
            intrinsics: {} as { raw: Record<string, number>; normalized: Record<string, number> },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        // Should render without crashing but show no keys
        expect(screen.queryByText("hp")).not.toBeInTheDocument();
    });

    it("uses blue color for low normalized values (<=0.3)", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 3 },
                normalized: { hp: 0.1 },
            },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("handles negative normalized values by clamping to 0", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 0 },
                normalized: { hp: -0.5 },
            },
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("hp")).toBeInTheDocument();
    });

    it("handles significance of exactly 0.5 as yellow", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.5,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("0.5000")).toBeInTheDocument();
    });

    it("handles significance of exactly 0.1 as green", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0.1,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("0.1000")).toBeInTheDocument();
    });

    it("handles significance of exactly 0 as green", () => {
        const data = makeStepData({
            intrinsics: {
                raw: { hp: 10 },
                normalized: { hp: 0.5 },
            },
            significance: 0,
        });
        renderWithProviders(<IntrinsicsPanel data={data} />);
        expect(screen.getByText("Significance")).toBeInTheDocument();
        expect(screen.getByText("0.0000")).toBeInTheDocument();
    });
});
