/** Shared test utilities -- Mantine wrapper, mock data factories, etc. */

import { MantineProvider } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { DashboardProvider } from "./state/context";
import type { GridData, LogEntry, StepData } from "./types/step-data";

/** Wraps components with Mantine + QueryClient + DashboardProvider. */
function AllProviders({ children }: { children: ReactNode }) {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    return (
        <MantineProvider>
            <QueryClientProvider client={queryClient}>
                <DashboardProvider>{children}</DashboardProvider>
            </QueryClientProvider>
        </MantineProvider>
    );
}

/** Custom render that wraps with all providers. */
export function renderWithProviders(
    ui: ReactElement,
    options?: Omit<RenderOptions, "wrapper">,
) {
    return render(ui, { wrapper: AllProviders, ...options });
}

/** Factory for minimal StepData. Override fields as needed. */
export function makeStepData(overrides: Partial<StepData> = {}): StepData {
    return {
        step: 1,
        game_number: 1,
        timestamp: null,
        screen: null,
        saliency: null,
        features: null,
        object_info: null,
        focus_points: null,
        attenuation: null,
        resolution_metrics: null,
        graph_summary: null,
        event_summary: null,
        game_metrics: null,
        logs: null,
        ...overrides,
    };
}

/** Factory for a small GridData. */
export function makeGridData(overrides: Partial<GridData> = {}): GridData {
    return {
        chars: [[65, 66], [67, 68]], // A B / C D
        fg: [["ffffff", "ff0000"], ["00ff00", "0000ff"]],
        bg: [["000000", "000000"], ["000000", "000000"]],
        ...overrides,
    };
}

/** Factory for log entries. */
export function makeLog(overrides: Partial<LogEntry> = {}): LogEntry {
    return {
        body: "test message",
        severity_text: "INFO",
        timestamp: 1000,
        ...overrides,
    };
}
