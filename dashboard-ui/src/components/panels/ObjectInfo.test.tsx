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

    it("renders objects with raw field", () => {
        const data = makeStepData({
            object_info: [{ raw: "Object #1 at (3,5)" }],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        expect(screen.getByText("Object #1 at (3,5)")).toBeInTheDocument();
    });

    it("renders objects as JSON when no raw field", () => {
        const data = makeStepData({
            object_info: [{ id: 1, name: "player" }],
        });
        renderWithProviders(<ObjectInfo data={data} />);
        expect(
            screen.getByText('{"id":1,"name":"player"}'),
        ).toBeInTheDocument();
    });
});
