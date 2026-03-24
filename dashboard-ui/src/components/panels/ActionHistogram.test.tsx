/** Regression tests for ActionHistogram bin-building logic. */

import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { ActionHistogram, buildBins } from "./ActionHistogram";

// ---------------------------------------------------------------------------
// buildBins unit tests (pure logic, no rendering)
// ---------------------------------------------------------------------------

describe("buildBins", () => {
    it("uses action map for names and fills all IDs from 0 to max", () => {
        const history = [
            { step: 1, action_id: 2, action_name: "S", action_key: "j" },
        ];
        const actionMap = [
            { action_id: 0, action_name: "N", action_key: "k" },
            { action_id: 1, action_name: "E", action_key: "l" },
            { action_id: 2, action_name: "S", action_key: "j" },
            { action_id: 3, action_name: "W", action_key: "h" },
        ];
        const bins = buildBins(history, actionMap);

        // All 4 IDs present even though only 1 was taken
        expect(bins).toHaveLength(4);
        expect(bins[0]).toEqual({ action_id: 0, action_name: "N", action_key: "k", count: 0 });
        expect(bins[2]).toEqual({ action_id: 2, action_name: "S", action_key: "j", count: 1 });
        expect(bins[3]).toEqual({ action_id: 3, action_name: "W", action_key: "h", count: 0 });
    });

    it("x-axis grows without action map (regression: only taken IDs shown)", () => {
        // Without action map, maxId comes from history. At game start only
        // a few actions are taken, so the x-axis is small and grows.
        const bins = buildBins(
            [{ step: 1, action_id: 3, action_name: "W", action_key: "h" }],
            undefined,
        );
        // Only IDs 0-3 (maxId=3 from history)
        expect(bins).toHaveLength(4);
        expect(bins[3]!.count).toBe(1);
        expect(bins[0]!.count).toBe(0);
    });

    it("action map fixes x-axis to full range from the start", () => {
        const actionMap = Array.from({ length: 86 }, (_, i) => ({
            action_id: i,
            action_name: `ACT_${i}`,
        }));
        const bins = buildBins(
            [{ step: 1, action_id: 3 }],
            actionMap,
        );
        // Full range 0-85 even with only 1 action taken
        expect(bins).toHaveLength(86);
        expect(bins[85]!.action_name).toBe("ACT_85");
    });

    it("does not produce 'Action #N' for taken actions (regression)", () => {
        // When action map is empty/undefined, history backfill must
        // provide names for all taken actions.
        const bins = buildBins(
            [
                { step: 1, action_id: 80, action_name: "PRAY" },
                { step: 2, action_id: 50, action_name: "PICKUP", action_key: "," },
            ],
            [], // empty action map (race condition)
        );
        const bin80 = bins.find((b) => b.action_id === 80);
        const bin50 = bins.find((b) => b.action_id === 50);
        expect(bin80!.action_name).toBe("PRAY");
        expect(bin50!.action_name).toBe("PICKUP");
        expect(bin50!.action_key).toBe(",");
    });

    it("falls back to 'Action #N' only for untaken IDs with no map", () => {
        const bins = buildBins(
            [{ step: 1, action_id: 5, action_name: "SE", action_key: "n" }],
            undefined,
        );
        // IDs 0-4 were never taken and have no map entry
        expect(bins[0]!.action_name).toBe("Action #0");
        expect(bins[4]!.action_name).toBe("Action #4");
        // ID 5 was taken and has a name from history
        expect(bins[5]!.action_name).toBe("SE");
        expect(bins[5]!.action_key).toBe("n");
    });

    it("prefers action map names over history backfill", () => {
        const bins = buildBins(
            [{ step: 1, action_id: 0, action_name: "N" }],
            [{ action_id: 0, action_name: "NORTH", action_key: "k" }],
        );
        // Map name takes precedence
        expect(bins[0]!.action_name).toBe("NORTH");
        expect(bins[0]!.action_key).toBe("k");
    });

    it("includes action_key from history when map is unavailable", () => {
        // Regression: tooltip should show "key: k" even without action map.
        const bins = buildBins(
            [{ step: 1, action_id: 0, action_name: "N", action_key: "k" }],
            undefined,
        );
        expect(bins[0]!.action_key).toBe("k");
    });
});

// ---------------------------------------------------------------------------
// Component smoke tests (mocked queries)
// ---------------------------------------------------------------------------

vi.mock("../../api/queries", () => ({
    useActionHistory: vi.fn(() => ({ data: undefined })),
    useActionMap: vi.fn(() => ({ data: undefined })),
}));

import { useActionHistory, useActionMap } from "../../api/queries";

const mockHistory = useActionHistory as ReturnType<typeof vi.fn>;
const mockMap = useActionMap as ReturnType<typeof vi.fn>;

describe("ActionHistogram component", () => {
    it("shows empty state when history is empty", () => {
        mockHistory.mockReturnValue({ data: [] });
        mockMap.mockReturnValue({ data: undefined });
        renderWithProviders(<ActionHistogram run="test-run" game={1} />);
        expect(screen.getByText("No action history")).toBeInTheDocument();
    });

    it("renders chart when history has data", () => {
        mockHistory.mockReturnValue({
            data: [{ step: 1, action_id: 0, action_name: "N", action_key: "k" }],
        });
        mockMap.mockReturnValue({ data: undefined });
        renderWithProviders(<ActionHistogram run="test-run" game={1} />);
        expect(screen.getByText("Action Frequency")).toBeInTheDocument();
    });
});
