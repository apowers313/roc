import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { StatusBar } from "./StatusBar";

describe("StatusBar", () => {
    it("shows step/game when no metrics", () => {
        renderWithProviders(
            <StatusBar data={makeStepData({ step: 5, game_number: 2 })} playbackState="historical" />,
        );
        expect(screen.getByText(/Step 5/)).toBeInTheDocument();
        expect(screen.getByText(/Game 2/)).toBeInTheDocument();
    });

    it("shows -- for step/game when data is undefined", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="historical" />,
        );
        expect(screen.getByText(/Step --/)).toBeInTheDocument();
    });

    it("renders metrics when game_metrics is present", () => {
        const data = makeStepData({
            game_metrics: {
                hp: 15,
                hp_max: 20,
                score: 100,
                depth: 1,
                gold: 50,
                energy: 10,
                hunger: "Not Hungry",
            },
        });
        renderWithProviders(
            <StatusBar data={data} playbackState="historical" />,
        );
        expect(screen.getByText("HP")).toBeInTheDocument();
        expect(screen.getByText("15/20")).toBeInTheDocument();
        expect(screen.getByText("100")).toBeInTheDocument();
    });

    it("shows LIVE badge when live_following", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="live_following" />,
        );
        expect(screen.getByText("LIVE")).toBeInTheDocument();
    });

    it("shows PAUSED badge when live_paused", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="live_paused" />,
        );
        expect(screen.getByText("PAUSED")).toBeInTheDocument();
    });

    it("shows CATCHING UP badge when live_catchup", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="live_catchup" />,
        );
        expect(screen.getByText("CATCHING UP")).toBeInTheDocument();
    });

    it("shows no badge for historical playback", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="historical" />,
        );
        expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
        expect(screen.queryByText("PAUSED")).not.toBeInTheDocument();
    });
});
