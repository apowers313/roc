import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeGridData, makeStepData, renderWithProviders } from "../../test-utils";
import { GameScreen } from "./GameScreen";

describe("GameScreen", () => {
    it("shows 'No screen data' when data is undefined", () => {
        renderWithProviders(<GameScreen data={undefined} />);
        expect(screen.getByText("No screen data")).toBeInTheDocument();
    });

    it("shows 'No screen data' when screen is null", () => {
        renderWithProviders(
            <GameScreen data={makeStepData({ screen: null })} />,
        );
        expect(screen.getByText("No screen data")).toBeInTheDocument();
    });

    it("renders CharGrid when screen data exists", () => {
        const data = makeStepData({ screen: makeGridData() });
        const { container } = renderWithProviders(<GameScreen data={data} />);

        expect(container.querySelector("pre")).toBeTruthy();
        expect(screen.queryByText("No screen data")).not.toBeInTheDocument();
    });
});
