import { screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { BookmarkBar } from "./BookmarkBar";
import type { Bookmark } from "../../types/api";

describe("BookmarkBar", () => {
    const bookmarks: Bookmark[] = [
        { step: 5, game: 1, annotation: "start", created: "2026-01-01" },
        { step: 15, game: 1, annotation: "", created: "2026-01-01" },
        { step: 25, game: 1, annotation: "end boss", created: "2026-01-01" },
    ];

    it("renders bookmark indicators", () => {
        const { container } = renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={1}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        // Should render marker dots for each bookmark
        const markers = container.querySelectorAll("[data-testid='bookmark-marker']");
        expect(markers).toHaveLength(3);
    });

    it("positions markers proportionally within the step range", () => {
        const { container } = renderWithProviders(
            <BookmarkBar
                bookmarks={[{ step: 50, game: 1, annotation: "", created: "" }]}
                currentStep={1}
                stepMin={0}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        const marker = container.querySelector("[data-testid='bookmark-marker']") as HTMLElement;
        expect(marker).toBeTruthy();
        // step=50 in range 0-100 -> 50%
        expect(marker.style.left).toBe("50%");
    });

    it("shows toggle button with filled state when step is bookmarked", () => {
        renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={5}
                stepMin={1}
                stepMax={100}
                isBookmarked={true}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        const toggleBtn = screen.getByLabelText("Remove bookmark");
        expect(toggleBtn).toBeInTheDocument();
    });

    it("shows toggle button with empty state when step is not bookmarked", () => {
        renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={3}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        const toggleBtn = screen.getByLabelText("Add bookmark");
        expect(toggleBtn).toBeInTheDocument();
    });

    it("calls onToggle when bookmark button is clicked", () => {
        const onToggle = vi.fn();
        renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={3}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={onToggle}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        fireEvent.click(screen.getByLabelText("Add bookmark"));
        expect(onToggle).toHaveBeenCalledOnce();
    });

    it("calls onNavigate when a marker is clicked", () => {
        const onNavigate = vi.fn();
        const { container } = renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={1}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={onNavigate}
                onAnnotate={() => {}}
            />,
        );
        const markers = container.querySelectorAll("[data-testid='bookmark-marker']");
        fireEvent.click(markers[1]!); // click step=15 marker
        expect(onNavigate).toHaveBeenCalledWith(bookmarks[1]);
    });

    it("shows annotation tooltip on marker hover", () => {
        const { container } = renderWithProviders(
            <BookmarkBar
                bookmarks={bookmarks}
                currentStep={1}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        const markers = container.querySelectorAll("[data-testid='bookmark-marker']");
        // First marker has annotation "start" -- shown as title attribute
        expect((markers[0] as HTMLElement).title).toContain("start");
    });

    it("renders nothing when no bookmarks and step not bookmarked", () => {
        const { container } = renderWithProviders(
            <BookmarkBar
                bookmarks={[]}
                currentStep={1}
                stepMin={1}
                stepMax={100}
                isBookmarked={false}
                onToggle={() => {}}
                onNavigate={() => {}}
                onAnnotate={() => {}}
            />,
        );
        // Should still render the toggle button even with no bookmarks
        expect(screen.getByLabelText("Add bookmark")).toBeInTheDocument();
        // But no markers
        const markers = container.querySelectorAll("[data-testid='bookmark-marker']");
        expect(markers).toHaveLength(0);
    });
});
