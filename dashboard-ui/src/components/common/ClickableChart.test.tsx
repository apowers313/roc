import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ClickableChart } from "./ClickableChart";

const sampleData = [
    { step: 10, value: 1 },
    { step: 20, value: 2 },
    { step: 30, value: 3 },
];

describe("ClickableChart", () => {
    it("renders children", () => {
        render(
            <ClickableChart onStepClick={() => {}} data={sampleData}>
                <div data-testid="child">Chart content</div>
            </ClickableChart>,
        );
        expect(screen.getByTestId("child")).toBeInTheDocument();
    });

    it("calls onStepClick when clicked", () => {
        const onStepClick = vi.fn();
        const { container } = render(
            <ClickableChart onStepClick={onStepClick} data={sampleData}>
                <div style={{ width: 500, height: 200 }}>Chart</div>
            </ClickableChart>,
        );

        const wrapper = container.firstElementChild as HTMLElement;
        // Simulate a click -- the exact step depends on position math,
        // but the callback should be called with a valid step number
        fireEvent.click(wrapper, { clientX: 250, clientY: 100 });
        expect(onStepClick).toHaveBeenCalled();
        const calledWith = onStepClick.mock.calls[0]![0] as number;
        expect([10, 20, 30]).toContain(calledWith);
    });

    it("does not crash with empty data", () => {
        const onStepClick = vi.fn();
        const { container } = render(
            <ClickableChart onStepClick={onStepClick} data={[]}>
                <div>Empty</div>
            </ClickableChart>,
        );

        const wrapper = container.firstElementChild as HTMLElement;
        fireEvent.click(wrapper, { clientX: 100, clientY: 100 });
        expect(onStepClick).not.toHaveBeenCalled();
    });

    it("has crosshair cursor style", () => {
        const { container } = render(
            <ClickableChart onStepClick={() => {}} data={sampleData}>
                <div>Chart</div>
            </ClickableChart>,
        );
        const wrapper = container.firstElementChild as HTMLElement;
        expect(wrapper.style.cursor).toBe("crosshair");
    });
});
