import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { AttenuationPanel } from "./AttenuationPanel";

describe("AttenuationPanel", () => {
    it("shows empty state when no data", () => {
        renderWithProviders(<AttenuationPanel data={undefined} />);
        expect(screen.getByText("No attenuation data")).toBeInTheDocument();
    });

    it("shows empty state when attenuation is null", () => {
        renderWithProviders(
            <AttenuationPanel data={makeStepData({ attenuation: null })} />,
        );
        expect(screen.getByText("No attenuation data")).toBeInTheDocument();
    });

    it("renders peak_count when attenuation data is present", () => {
        renderWithProviders(
            <AttenuationPanel
                data={makeStepData({
                    attenuation: { flavor: "linear-decline", peak_count: 5 },
                })}
            />,
        );
        // flavor is shown at Section level (expmod prop), not inside AttenuationPanel
        expect(screen.getByText("peak_count")).toBeInTheDocument();
        expect(screen.getByText("5")).toBeInTheDocument();
    });

    // Regression: field names must match actual Python emission keys
    it("renders entropy_at_focus, entropy_max, entropy_min (not entropy/max_entropy)", () => {
        renderWithProviders(
            <AttenuationPanel
                data={makeStepData({
                    attenuation: {
                        flavor: "active-inference",
                        entropy_at_focus: 1.234,
                        entropy_max: 2.5,
                        entropy_min: 0.1,
                    },
                })}
            />,
        );
        expect(screen.getByText("entropy_at_focus")).toBeInTheDocument();
        expect(screen.getByText("entropy_max")).toBeInTheDocument();
        expect(screen.getByText("entropy_min")).toBeInTheDocument();
        expect(screen.getByText("1.2340")).toBeInTheDocument();
    });

    // Regression: pre_peak and post_peak should be displayed
    it("renders pre_peak and post_peak as clickable point entries", () => {
        renderWithProviders(
            <AttenuationPanel
                data={makeStepData({
                    attenuation: {
                        flavor: "linear-decline",
                        peak_count: 3,
                        pre_peak: [10, 20],
                        post_peak: [15, 25],
                    },
                })}
            />,
        );
        expect(screen.getByText("pre_peak")).toBeInTheDocument();
        expect(screen.getByText("(10, 20)")).toBeInTheDocument();
        expect(screen.getByText("post_peak")).toBeInTheDocument();
        expect(screen.getByText("(15, 25)")).toBeInTheDocument();
    });

    // Regression: history table should be rendered when present
    it("renders attended locations history table", () => {
        renderWithProviders(
            <AttenuationPanel
                data={makeStepData({
                    attenuation: {
                        flavor: "linear-decline",
                        history: [
                            { x: 5, y: 10, tick: 100 },
                            { x: 15, y: 20, tick: 101 },
                        ],
                        history_size: 2,
                    },
                })}
            />,
        );
        expect(
            screen.getByText("Attended Locations (attenuated)"),
        ).toBeInTheDocument();
        expect(screen.getByText("100")).toBeInTheDocument();
        expect(screen.getByText("101")).toBeInTheDocument();
    });

    it("renders beliefs_tracked and vocab_size", () => {
        renderWithProviders(
            <AttenuationPanel
                data={makeStepData({
                    attenuation: {
                        flavor: "active-inference",
                        beliefs_tracked: 42,
                        vocab_size: 100,
                        omega: 0.5,
                    },
                })}
            />,
        );
        expect(screen.getByText("beliefs_tracked")).toBeInTheDocument();
        expect(screen.getByText("42")).toBeInTheDocument();
        expect(screen.getByText("vocab_size")).toBeInTheDocument();
        expect(screen.getByText("100")).toBeInTheDocument();
        expect(screen.getByText("omega")).toBeInTheDocument();
    });
});
