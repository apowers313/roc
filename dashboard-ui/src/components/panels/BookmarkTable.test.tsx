import { screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { BookmarkTable } from "./BookmarkTable";

const bookmarks = [
    { step: 10, game: 1, annotation: "interesting", created: "2026-01-01T00:00:00Z" },
    { step: 50, game: 1, annotation: "", created: "2026-01-01T00:00:00Z" },
    { step: 200, game: 2, annotation: "bug here", created: "2026-01-01T00:00:00Z" },
];

describe("BookmarkTable", () => {
    it("shows empty state with no bookmarks", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={[]}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        expect(screen.getByText(/No bookmarks/)).toBeInTheDocument();
    });

    it("renders all bookmarks sorted by step", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        expect(screen.getByText("10")).toBeInTheDocument();
        expect(screen.getByText("50")).toBeInTheDocument();
        expect(screen.getByText("200")).toBeInTheDocument();
        expect(screen.getByText("interesting")).toBeInTheDocument();
        expect(screen.getByText("bug here")).toBeInTheDocument();
    });

    it("calls onNavigate with full bookmark when a row is clicked", () => {
        const onNavigate = vi.fn();
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={onNavigate}
                onUpdateBookmark={vi.fn()}
            />,
        );
        // Click the step number -- single click navigates
        fireEvent.click(screen.getByText("50"));
        expect(onNavigate).toHaveBeenCalledWith(bookmarks[1]);
    });

    it("navigates to correct game when clicking bookmark from different game", () => {
        const onNavigate = vi.fn();
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={onNavigate}
                onUpdateBookmark={vi.fn()}
            />,
        );
        // Click the annotation text -- single click navigates via row handler
        fireEvent.click(screen.getByText("bug here"));
        expect(onNavigate).toHaveBeenCalledWith(
            expect.objectContaining({ step: 200, game: 2 }),
        );
    });

    it("highlights current step row", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={10}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        const row = screen.getByText("10").closest("tr");
        expect(row?.style.fontWeight).toBe("600");
    });

    it("shows column headers", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        expect(screen.getByText("Step")).toBeInTheDocument();
        expect(screen.getByText("Game")).toBeInTheDocument();
        expect(screen.getByText("Annotation")).toBeInTheDocument();
    });

    it("enters edit mode when double-clicking annotation cell", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        // Double-click the annotation placeholder for step 50
        const placeholders = screen.getAllByText("double-click to edit");
        fireEvent.doubleClick(placeholders[0]!);
        expect(screen.getByRole("textbox")).toBeInTheDocument();
    });

    it("calls onUpdateBookmark when annotation is edited", () => {
        const onUpdateBookmark = vi.fn();
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={onUpdateBookmark}
            />,
        );
        // Double-click annotation for step 10
        fireEvent.doubleClick(screen.getByText("interesting"));
        const input = screen.getByRole("textbox");
        fireEvent.change(input, { target: { value: "very interesting" } });
        fireEvent.keyDown(input, { key: "Enter" });
        expect(onUpdateBookmark).toHaveBeenCalledWith(10, { annotation: "very interesting" });
    });

    it("calls onUpdateBookmark when step is edited", () => {
        const onUpdateBookmark = vi.fn();
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={onUpdateBookmark}
            />,
        );
        // Double-click the step number "10"
        fireEvent.doubleClick(screen.getByText("10"));
        const input = screen.getByRole("textbox");
        fireEvent.change(input, { target: { value: "15" } });
        fireEvent.keyDown(input, { key: "Enter" });
        expect(onUpdateBookmark).toHaveBeenCalledWith(10, { step: 15 });
    });

    it("calls onUpdateBookmark when game is edited", () => {
        const onUpdateBookmark = vi.fn();
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={onUpdateBookmark}
            />,
        );
        // The game column for step 200 shows "2"
        fireEvent.doubleClick(screen.getByText("2"));
        const input = screen.getByRole("textbox");
        fireEvent.change(input, { target: { value: "3" } });
        fireEvent.keyDown(input, { key: "Enter" });
        expect(onUpdateBookmark).toHaveBeenCalledWith(200, { game: 3 });
    });

    it("cancels edit on Escape", () => {
        renderWithProviders(
            <BookmarkTable
                bookmarks={bookmarks}
                currentStep={1}
                onNavigate={vi.fn()}
                onUpdateBookmark={vi.fn()}
            />,
        );
        fireEvent.doubleClick(screen.getByText("interesting"));
        const input = screen.getByRole("textbox");
        fireEvent.keyDown(input, { key: "Escape" });
        expect(screen.queryByRole("textbox")).toBeNull();
        expect(screen.getByText("interesting")).toBeInTheDocument();
    });
});
