import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { MessageLog } from "./MessageLog";

describe("MessageLog", () => {
    it("shows 'No message this step' when data is undefined", () => {
        renderWithProviders(<MessageLog data={undefined} />);
        expect(screen.getByText("No message this step")).toBeInTheDocument();
    });

    it("shows 'No message this step' when message is null", () => {
        renderWithProviders(<MessageLog data={makeStepData()} />);
        expect(screen.getByText("No message this step")).toBeInTheDocument();
    });

    it("shows 'No message this step' when message is empty string", () => {
        renderWithProviders(<MessageLog data={makeStepData({ message: "" })} />);
        expect(screen.getByText("No message this step")).toBeInTheDocument();
    });

    it("renders message text in a paper container", () => {
        const data = makeStepData({ message: "You hear a distant rumble." });
        renderWithProviders(<MessageLog data={data} />);
        expect(screen.getByText("You hear a distant rumble.")).toBeInTheDocument();
    });

    it("renders long messages", () => {
        const longMsg = "The gnome hits you with a club. You are hit. The gnome picks up a gem.";
        const data = makeStepData({ message: longMsg });
        renderWithProviders(<MessageLog data={data} />);
        expect(screen.getByText(longMsg)).toBeInTheDocument();
    });
});
