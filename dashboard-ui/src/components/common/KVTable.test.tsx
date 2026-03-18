import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { KVTable } from "./KVTable";

describe("KVTable", () => {
    it("shows empty text when data is null", () => {
        renderWithProviders(<KVTable data={null} />);
        expect(screen.getByText("No data")).toBeInTheDocument();
    });

    it("shows empty text when data is undefined", () => {
        renderWithProviders(<KVTable data={undefined} />);
        expect(screen.getByText("No data")).toBeInTheDocument();
    });

    it("shows custom empty text", () => {
        renderWithProviders(
            <KVTable data={null} emptyText="Nothing here" />,
        );
        expect(screen.getByText("Nothing here")).toBeInTheDocument();
    });

    it("shows empty text for empty object", () => {
        renderWithProviders(<KVTable data={{}} />);
        expect(screen.getByText("No data")).toBeInTheDocument();
    });

    it("renders key-value pairs", () => {
        renderWithProviders(<KVTable data={{ score: 42, depth: 3 }} />);
        expect(screen.getByText("score")).toBeInTheDocument();
        expect(screen.getByText("42")).toBeInTheDocument();
        expect(screen.getByText("depth")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("formats floating point numbers to 4 decimal places", () => {
        renderWithProviders(<KVTable data={{ ratio: 0.123456789 }} />);
        expect(screen.getByText("0.1235")).toBeInTheDocument();
    });

    it("shows -- for null/undefined values", () => {
        renderWithProviders(<KVTable data={{ empty: null }} />);
        expect(screen.getByText("--")).toBeInTheDocument();
    });

    it("truncates long strings", () => {
        const longStr = "a".repeat(100);
        renderWithProviders(<KVTable data={{ long: longStr }} />);
        expect(screen.getByText("a".repeat(77) + "...")).toBeInTheDocument();
    });

    it("renders with a title in a card", () => {
        renderWithProviders(
            <KVTable data={{ key: "val" }} title="Test Title" />,
        );
        expect(screen.getByText("Test Title")).toBeInTheDocument();
        expect(screen.getByText("key")).toBeInTheDocument();
    });
});
