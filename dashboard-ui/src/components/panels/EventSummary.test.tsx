import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { EventSummary } from "./EventSummary";

describe("EventSummary", () => {
    it("shows 'No event data' when data is undefined", () => {
        renderWithProviders(<EventSummary data={undefined} />);
        expect(screen.getByText("No event data")).toBeInTheDocument();
    });

    it("shows 'No event data' when event_summary is empty", () => {
        renderWithProviders(
            <EventSummary data={makeStepData({ event_summary: [] })} />,
        );
        expect(screen.getByText("No event data")).toBeInTheDocument();
    });

    it("renders event counts from first entry", () => {
        const data = makeStepData({
            event_summary: [{ perception: 5, attention: 3 }],
        });
        renderWithProviders(<EventSummary data={data} />);
        expect(screen.getByText("Events")).toBeInTheDocument();
        expect(screen.getByText("perception")).toBeInTheDocument();
        expect(screen.getByText("5")).toBeInTheDocument();
    });
});
