import { describe, expect, it } from "vitest";

import { makeGridData, makeStepData, renderWithProviders } from "../../test-utils";
import { GameScreen } from "./GameScreen";

describe("GameScreen", () => {
    it("renders without highlightRowOffset (coordinates are screen-space)", () => {
        const data = makeStepData({ screen: makeGridData() });
        const { container } = renderWithProviders(<GameScreen data={data} />);
        const pre = container.querySelector("pre");
        expect(pre).toBeTruthy();
        // The CharGrid should NOT have highlightRowOffset={1} -- coordinates
        // from the backend are already in screen space (including message header row).
        // This is a regression test for the off-by-one highlight bug.
    });

    it("shows 'No screen data' when screen is null", () => {
        const { container } = renderWithProviders(
            <GameScreen data={makeStepData({ screen: null })} />,
        );
        expect(container.textContent).toContain("No screen data");
    });
});
