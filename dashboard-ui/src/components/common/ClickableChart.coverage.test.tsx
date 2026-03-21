/**
 * Additional coverage tests for ClickableChart -- covers the fallback path
 * when no recharts-cartesian-grid element is found.
 */

import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ClickableChart } from "./ClickableChart";

const sampleData = [
    { step: 10, value: 1 },
    { step: 20, value: 2 },
    { step: 30, value: 3 },
];

describe("ClickableChart (fallback path)", () => {
    it("uses fallback margin-based calculation when no cartesian grid exists", () => {
        const onStepClick = vi.fn();
        const { container } = render(
            <ClickableChart onStepClick={onStepClick} data={sampleData}>
                <div style={{ width: 500, height: 200 }}>
                    {/* No .recharts-cartesian-grid element here */}
                    <span>Simple content</span>
                </div>
            </ClickableChart>,
        );

        const wrapper = container.firstElementChild as HTMLElement;
        // Click somewhere in the middle -- should use fallback calculation
        fireEvent.click(wrapper, { clientX: 250, clientY: 100 });
        expect(onStepClick).toHaveBeenCalled();
        const calledWith = onStepClick.mock.calls[0]![0] as number;
        expect([10, 20, 30]).toContain(calledWith);
    });

    it("uses cartesian grid bounds when the element exists", () => {
        const onStepClick = vi.fn();
        const { container } = render(
            <ClickableChart onStepClick={onStepClick} data={sampleData}>
                <div style={{ width: 500, height: 200 }}>
                    {/* Simulate a recharts cartesian grid element */}
                    <div className="recharts-cartesian-grid" style={{ width: 400, height: 150 }} />
                </div>
            </ClickableChart>,
        );

        const wrapper = container.firstElementChild as HTMLElement;
        // getBoundingClientRect returns all zeros in jsdom, but the code path is exercised
        fireEvent.click(wrapper, { clientX: 200, clientY: 100 });
        expect(onStepClick).toHaveBeenCalled();
    });

    it("handles single data point", () => {
        const onStepClick = vi.fn();
        const { container } = render(
            <ClickableChart onStepClick={onStepClick} data={[{ step: 42 }]}>
                <div>Single point</div>
            </ClickableChart>,
        );

        const wrapper = container.firstElementChild as HTMLElement;
        fireEvent.click(wrapper, { clientX: 100, clientY: 50 });
        expect(onStepClick).toHaveBeenCalledWith(42);
    });
});
