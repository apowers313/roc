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

    it("renders CharGrid and legend when saliency data exists", () => {
        const data = makeStepData({ saliency: makeGridData() });
        const { container } = renderWithProviders(<SaliencyMap data={data} />);

        expect(container.querySelector("pre")).toBeTruthy();
        expect(screen.getByText("Low")).toBeInTheDocument();
        expect(screen.getByText("High")).toBeInTheDocument();
    });
});
