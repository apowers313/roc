/** Application entry point -- providers and React root. */

import "@mantine/core/styles.css";

import { MantineProvider } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { compactTheme } from "@graphty/compact-mantine";

import { App } from "./App";
import { DashboardProvider } from "./state/context";
import { HighlightProvider } from "./state/highlight";

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: 1,
            refetchOnWindowFocus: false,
        },
    },
});

const root = document.getElementById("root");
if (root) {
    createRoot(root).render(
        <StrictMode>
            <MantineProvider theme={compactTheme} defaultColorScheme="dark">
                <QueryClientProvider client={queryClient}>
                    <DashboardProvider>
                        <HighlightProvider>
                            <App />
                        </HighlightProvider>
                    </DashboardProvider>
                </QueryClientProvider>
            </MantineProvider>
        </StrictMode>,
    );
}
