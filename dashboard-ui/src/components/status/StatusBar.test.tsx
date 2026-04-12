import { screen, fireEvent } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { StatusBar } from "./StatusBar";

// Phase 3: tail_growing comes from useStepRange. Mock the queries
// module so tests can drive the value without setting up a real query
// client and HTTP layer.
vi.mock("../../api/queries", () => ({
    useStepRange: vi.fn(() => ({ data: { min: 1, max: 5, tail_growing: false } })),
}));

import { useStepRange } from "../../api/queries";

const mockUseStepRange = vi.mocked(useStepRange);

/** Helper that sets the mocked step-range tail_growing to true for the children. */
function SetLiveActive({ children }: Readonly<{ children: ReactNode }>) {
    // Each render call resets the mock so this only affects subsequent
    // useStepRange calls inside `children` for the duration of the test.
    mockUseStepRange.mockReturnValue({
        data: { min: 1, max: 5, tail_growing: true },
    } as ReturnType<typeof useStepRange>);
    return <>{children}</>;
}

beforeEach(() => {
    mockUseStepRange.mockReturnValue({
        data: { min: 1, max: 5, tail_growing: false },
    } as ReturnType<typeof useStepRange>);
});

afterEach(() => {
    vi.clearAllMocks();
});

describe("StatusBar", () => {
    it("shows step/game when no metrics", () => {
        renderWithProviders(
            <StatusBar data={makeStepData({ step: 5, game_number: 2 })} autoFollow={false} />,
        );
        expect(screen.getByText(/Step 5/)).toBeInTheDocument();
        expect(screen.getByText(/Game 2/)).toBeInTheDocument();
    });

    it("shows -- for step/game when data is undefined", () => {
        renderWithProviders(
            <StatusBar data={undefined} autoFollow={false} />,
        );
        expect(screen.getByText(/Step --/)).toBeInTheDocument();
    });

    // Regression: navigating to a step beyond the data range (URL ?step=2500
    // against a 314-step game) hits the API which returns game_number=0 and
    // screen=null. The StatusBar used to render "Step 2500 | Game 0" which
    // looks like a real reading and made users believe the game had 2k+
    // steps. Surface this as an explicit "no data" state instead.
    it("shows 'No data at step N' when game_number is 0 and screen is null", () => {
        renderWithProviders(
            <StatusBar
                data={makeStepData({ step: 2500, game_number: 0, screen: null })}
                autoFollow={false}
            />,
        );
        expect(screen.getByText("No data at step 2500")).toBeInTheDocument();
        expect(screen.queryByText(/Game 0/)).not.toBeInTheDocument();
    });

    it("does not show 'No data' when screen is present even without metrics", () => {
        renderWithProviders(
            <StatusBar
                data={makeStepData({
                    step: 5,
                    game_number: 1,
                    screen: { chars: [[65]], fg: [["fff"]], bg: [["000"]] },
                })}
                autoFollow={false}
            />,
        );
        expect(screen.getByText(/Step 5/)).toBeInTheDocument();
        expect(screen.queryByText(/No data/)).not.toBeInTheDocument();
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
            <StatusBar data={data} autoFollow={false} />,
        );
        expect(screen.getByText("HP")).toBeInTheDocument();
        expect(screen.getByText("15/20")).toBeInTheDocument();
        expect(screen.getByText("100")).toBeInTheDocument();
    });

    it("shows LIVE badge when tail_growing and autoFollow", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} autoFollow={true} />
            </SetLiveActive>,
        );
        expect(screen.getByText("LIVE")).toBeInTheDocument();
    });

    it("LIVE badge is not clickable", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} autoFollow={true} />
            </SetLiveActive>,
        );
        const badge = screen.getByText("LIVE");
        // Should not have a click handler / button role
        expect(badge.closest("button")).toBeNull();
    });

    it("shows GO LIVE badge when tail_growing but autoFollow=false", () => {
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} autoFollow={false} onGoLive={() => {}} />
            </SetLiveActive>,
        );
        expect(screen.getByText("GO LIVE")).toBeInTheDocument();
    });

    it("GO LIVE badge is clickable and calls onGoLive", () => {
        const onGoLive = vi.fn();
        renderWithProviders(
            <SetLiveActive>
                <StatusBar data={undefined} autoFollow={false} onGoLive={onGoLive} />
            </SetLiveActive>,
        );
        fireEvent.click(screen.getByText("GO LIVE"));
        expect(onGoLive).toHaveBeenCalledOnce();
    });

    it("shows no badge when tail_growing is false", () => {
        renderWithProviders(
            <StatusBar data={undefined} autoFollow={false} />,
        );
        expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
        expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
    });

    // ----------------------------------------------------------------
    // Phase 5: autoFollow + tail_growing drive the badge directly.
    //
    // Phase 3 added tail_growing as the only liveness signal. Phase 5
    // collapsed the four-state playback machine to two booleans, so
    // the badge derivation is now:
    //   LIVE    = tail_growing && autoFollow
    //   GO LIVE = tail_growing && !autoFollow
    // ----------------------------------------------------------------
    describe("tail_growing + autoFollow badges", () => {
        it("shows LIVE when tail_growing=true and autoFollow=true", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 5, tail_growing: true },
            } as ReturnType<typeof useStepRange>);
            renderWithProviders(
                <StatusBar data={undefined} autoFollow={true} />,
            );
            expect(screen.getByText("LIVE")).toBeInTheDocument();
            expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
        });

        it("shows GO LIVE when tail_growing=true and autoFollow=false", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 5, tail_growing: true },
            } as ReturnType<typeof useStepRange>);
            renderWithProviders(
                <StatusBar data={undefined} autoFollow={false} onGoLive={() => {}} />,
            );
            expect(screen.getByText("GO LIVE")).toBeInTheDocument();
            expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
        });

        it("hides both badges when tail_growing=false regardless of autoFollow", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 5, tail_growing: false },
            } as ReturnType<typeof useStepRange>);
            renderWithProviders(
                <StatusBar data={undefined} autoFollow={true} />,
            );
            expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
            expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
        });

        it("hides both badges when step-range data is undefined", () => {
            mockUseStepRange.mockReturnValue({
                data: undefined,
            } as ReturnType<typeof useStepRange>);
            renderWithProviders(
                <StatusBar data={undefined} autoFollow={false} />,
            );
            expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
            expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
        });
    });

    // ----------------------------------------------------------------
    // Regression: REST step fetch errors must be visible.
    //
    // Until 2026-04-08, /api/runs/{run}/step/{n} could throw 500
    // (e.g. pd.NA serialization bug) and the StatusBar would silently
    // render "Step -- | Game --" indistinguishably from "no data
    // selected". The user complained about "errors that aren't
    // visible in the UI". The ERROR badge surfaces these.
    // ----------------------------------------------------------------
    describe("fetchError badge", () => {
        it("shows ERROR badge when fetchError is an Error", () => {
            renderWithProviders(
                <StatusBar
                    data={undefined}
                    autoFollow={false}
                    fetchError={new Error("API error: 500 Internal Server Error")}
                />,
            );
            expect(screen.getByText("ERROR")).toBeInTheDocument();
        });

        it("shows ERROR badge when fetchError is a string", () => {
            renderWithProviders(
                <StatusBar
                    data={undefined}
                    autoFollow={false}
                    fetchError="something went wrong"
                />,
            );
            expect(screen.getByText("ERROR")).toBeInTheDocument();
        });

        it("does NOT show ERROR badge when fetchError is null", () => {
            renderWithProviders(
                <StatusBar
                    data={undefined}
                    autoFollow={false}
                    fetchError={null}
                />,
            );
            expect(screen.queryByText("ERROR")).not.toBeInTheDocument();
        });

        it("does NOT show ERROR badge when fetchError is undefined", () => {
            renderWithProviders(
                <StatusBar data={undefined} autoFollow={false} />,
            );
            expect(screen.queryByText("ERROR")).not.toBeInTheDocument();
        });

        it("ERROR badge appears alongside metrics if data exists", () => {
            // Even when stale data is shown (placeholderData from
            // previous step), the user must see that the *current*
            // fetch failed.
            const data = makeStepData({
                game_metrics: { hp: 10, hp_max: 20 },
            });
            renderWithProviders(
                <StatusBar
                    data={data}
                    autoFollow={false}
                    fetchError={new Error("boom")}
                />,
            );
            expect(screen.getByText("ERROR")).toBeInTheDocument();
            expect(screen.getByText("HP")).toBeInTheDocument();
        });
    });
});
