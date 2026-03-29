import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeGridData, makeStepData, renderWithProviders } from "../../test-utils";
import { SaliencyMap } from "./SaliencyMap";

describe("SaliencyMap", () => {
    it("shows 'No saliency data' when data is undefined", () => {
        renderWithProviders(<SaliencyMap data={undefined} />);
        expect(screen.getByText("No saliency data")).toBeInTheDocument();
    });

    it("shows 'No saliency data' when saliency is null", () => {
        renderWithProviders(
            <SaliencyMap data={makeStepData({ saliency: null })} />,
        );
        expect(screen.getByText("No saliency data")).toBeInTheDocument();
    });

    it("renders CharGrid when saliency data exists", () => {
        const data = makeStepData({ saliency: makeGridData() });
        const { container } = renderWithProviders(<SaliencyMap data={data} />);

        expect(container.querySelector("pre")).toBeTruthy();
    });

    it("uses saliency_cycles when cycleIndex is provided", () => {
        const cycleGrid = makeGridData({
            chars: [[90, 91], [92, 93]], // Z [ \ ]
        });
        const data = makeStepData({
            saliency: makeGridData(),
            saliency_cycles: [
                { saliency: cycleGrid, attenuation: {} },
            ],
        });
        const { container } = renderWithProviders(
            <SaliencyMap data={data} cycleIndex={0} />,
        );
        expect(container.querySelector("pre")).toBeTruthy();
    });

    it("falls back to top-level saliency when no cycles (backward compat)", () => {
        const data = makeStepData({ saliency: makeGridData() });
        const { container } = renderWithProviders(
            <SaliencyMap data={data} cycleIndex={undefined} />,
        );
        expect(container.querySelector("pre")).toBeTruthy();
    });
});
