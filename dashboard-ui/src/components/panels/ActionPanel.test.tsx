import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { ActionPanel } from "./ActionPanel";

describe("ActionPanel", () => {
    it("shows 'No action data' when data is undefined", () => {
        renderWithProviders(<ActionPanel data={undefined} />);
        expect(screen.getByText("No action data")).toBeInTheDocument();
    });

    it("shows 'No action data' when action_taken is null", () => {
        renderWithProviders(<ActionPanel data={makeStepData()} />);
        expect(screen.getByText("No action data")).toBeInTheDocument();
    });

    it("renders action name badge when action_name is present", () => {
        const data = makeStepData({
            action_taken: { action_id: 7, action_name: "MOVE_NORTH" },
        });
        renderWithProviders(<ActionPanel data={data} />);
        expect(screen.getByText("MOVE_NORTH")).toBeInTheDocument();
        expect(screen.getByText("(id: 7)")).toBeInTheDocument();
    });

    it("renders 'Action #id' when action_name is missing", () => {
        const data = makeStepData({
            action_taken: { action_id: 42 },
        });
        renderWithProviders(<ActionPanel data={data} />);
        expect(screen.getByText("Action #42")).toBeInTheDocument();
        expect(screen.getByText("(id: 42)")).toBeInTheDocument();
    });

    it("does not render expmod inline (expmod is shown by Section)", () => {
        const data = makeStepData({
            action_taken: { action_id: 5, action_name: "KICK", expmod_name: "random-action" },
        });
        renderWithProviders(<ActionPanel data={data} />);
        expect(screen.getByText("KICK")).toBeInTheDocument();
        // Expmod badge is rendered by the parent Section, not by ActionPanel
        expect(screen.queryByText("random-action")).not.toBeInTheDocument();
    });

    it("always shows Action label", () => {
        renderWithProviders(<ActionPanel data={undefined} />);
        expect(screen.getByText("Action")).toBeInTheDocument();
    });
});
