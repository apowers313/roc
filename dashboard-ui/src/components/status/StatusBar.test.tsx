import { screen, fireEvent } from "@testing-library/react";
import { useEffect, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useDashboard } from "../../state/context";
import { makeStepData, renderWithProviders } from "../../test-utils";
import { StatusBar } from "./StatusBar";

/** Helper that sets liveGameActive=true in context before rendering children. */
function SetLiveActive({ children }: Readonly<{ children: ReactNode }>) {
    const { setLiveGameActive } = useDashboard();
    useEffect(() => { setLiveGameActive(true); }, [setLiveGameActive]);
    return <>{children}</>;
}

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
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="live_following" />
            </SetLiveActive>,
        );
        expect(screen.getByText("LIVE")).toBeInTheDocument();
    });

    it("LIVE badge is not clickable", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="live_following" />
            </SetLiveActive>,
        );
        const badge = screen.getByText("LIVE");
        // Should not have a click handler / button role
        expect(badge.closest("button")).toBeNull();
    });

    it("shows GO LIVE badge when live_paused", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="live_paused" onGoLive={() => {}} />
            </SetLiveActive>,
        );
        expect(screen.getByText("GO LIVE")).toBeInTheDocument();
    });

    it("shows GO LIVE badge when live_catchup", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="live_catchup" onGoLive={() => {}} />
            </SetLiveActive>,
        );
        expect(screen.getByText("GO LIVE")).toBeInTheDocument();
    });

    it("shows GO LIVE badge when historical but live game is active", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="historical" onGoLive={() => {}} />
            </SetLiveActive>,
        );
        expect(screen.getByText("GO LIVE")).toBeInTheDocument();
    });

    it("GO LIVE badge is clickable and calls onGoLive", () => {
        const onGoLive = vi.fn();
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="live_paused" onGoLive={onGoLive} />
            </SetLiveActive>,
        );
        fireEvent.click(screen.getByText("GO LIVE"));
        expect(onGoLive).toHaveBeenCalledOnce();
    });

    it("GO LIVE from historical mode calls onGoLive", () => {
        const onGoLive = vi.fn();
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} playbackState="historical" onGoLive={onGoLive} />
            </SetLiveActive>,
        );
        fireEvent.click(screen.getByText("GO LIVE"));
        expect(onGoLive).toHaveBeenCalledOnce();
    });

    it("shows no badge for historical playback without live game", () => {
        renderWithProviders(
            <StatusBar data={undefined} playbackState="historical" />,
        );
        expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
        expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
    });
});
