import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { ErrorBoundary } from "./ErrorBoundary";

function ThrowingChild(): never {
    throw new Error("boom");
}

describe("ErrorBoundary", () => {
    it("renders children when no error", () => {
        renderWithProviders(
            <ErrorBoundary>
                <span>ok</span>
            </ErrorBoundary>,
        );
        expect(screen.getByText("ok")).toBeInTheDocument();
    });

    it("renders error message on child throw", () => {
        // Suppress React error boundary console noise
        const spy = vi.spyOn(console, "error").mockImplementation(() => {});
        renderWithProviders(
            <ErrorBoundary>
                <ThrowingChild />
            </ErrorBoundary>,
        );
        expect(screen.getByText(/Render error: boom/)).toBeInTheDocument();
        spy.mockRestore();
    });

    it("renders custom fallback on error", () => {
        const spy = vi.spyOn(console, "error").mockImplementation(() => {});
        renderWithProviders(
            <ErrorBoundary fallback={<span>custom fallback</span>}>
                <ThrowingChild />
            </ErrorBoundary>,
        );
        expect(screen.getByText("custom fallback")).toBeInTheDocument();
        spy.mockRestore();
    });
});
