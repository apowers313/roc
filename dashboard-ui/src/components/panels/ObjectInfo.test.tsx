import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { ObjectInfo } from "./ObjectInfo";

describe("ObjectInfo", () => {
    it("shows 'No object data' when data is undefined", () => {
        renderWithProviders(<ObjectInfo data={undefined} />);
        expect(screen.getByText("No object data")).toBeInTheDocument();
    });

    it("shows 'No object data' when object_info is empty", () => {
        renderWithProviders(
            <ObjectInfo data={makeStepData({ object_info: [] })} />,
        );
        expect(screen.getByText("No object data")).toBeInTheDocument();
    });

    it("renders objects with raw field as simple text", () => {
        const data = makeStepData({
            object_info: [{ raw: "Object #1 at (3,5)" }],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        expect(screen.getByText("Object #1 at (3,5)")).toBeInTheDocument();
    });

    it("renders structured objects as a table with column headers", () => {
        const data = makeStepData({
            object_info: [
                { id: 1, name: "goblin", type: "monster" },
                { id: 2, name: "sword", type: "weapon" },
            ],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        // Column headers
        expect(screen.getByText("id")).toBeInTheDocument();
        expect(screen.getByText("name")).toBeInTheDocument();
        expect(screen.getByText("type")).toBeInTheDocument();
        // Cell values
        expect(screen.getByText("goblin")).toBeInTheDocument();
        expect(screen.getByText("sword")).toBeInTheDocument();
    });

    it("formats float values with 3 decimal places", () => {
        const data = makeStepData({
            object_info: [{ score: 0.12345 }],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        expect(screen.getByText("0.123")).toBeInTheDocument();
    });

    it("shows -- for null values in structured mode", () => {
        const data = makeStepData({
            object_info: [
                { id: 1, name: "goblin" },
                { id: 2, name: null },
            ],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        expect(screen.getByText("--")).toBeInTheDocument();
    });
});
