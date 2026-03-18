import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { FocusPoints } from "./FocusPoints";

describe("FocusPoints", () => {
    it("shows 'No focus data' when data is undefined", () => {
        renderWithProviders(<FocusPoints data={undefined} />);
        expect(screen.getByText("No focus data")).toBeInTheDocument();
    });

    it("shows 'No focus data' when focus_points is empty", () => {
        renderWithProviders(
            <FocusPoints data={makeStepData({ focus_points: [] })} />,
        );
        expect(screen.getByText("No focus data")).toBeInTheDocument();
    });

    it("shows 'No focus data' when raw is not parseable", () => {
        renderWithProviders(
            <FocusPoints
                data={makeStepData({ focus_points: [{ raw: "invalid" }] })}
            />,
        );
        expect(screen.getByText("No focus data")).toBeInTheDocument();
    });

    it("parses and renders focus points from raw DataFrame string", () => {
        const raw =
            "    x   y  strength  label\n0  28  18  0.933333      3\n1  40  12  0.500000      2";
        const data = makeStepData({ focus_points: [{ raw }] });
        renderWithProviders(<FocusPoints data={data} />);

        expect(screen.getByText("Focus Points")).toBeInTheDocument();
        expect(screen.getByText("28")).toBeInTheDocument();
        expect(screen.getByText("18")).toBeInTheDocument();
        expect(screen.getByText("0.93")).toBeInTheDocument();
        expect(screen.getByText("40")).toBeInTheDocument();
        expect(screen.getByText("12")).toBeInTheDocument();
        expect(screen.getByText("0.50")).toBeInTheDocument();
    });
});
