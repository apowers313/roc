import { describe, expect, it } from "vitest";

import { makeGridData, renderWithProviders } from "../../test-utils";
import { CharGrid } from "./CharGrid";

describe("CharGrid", () => {
    it("renders a pre element with grid HTML", () => {
        const data = makeGridData();
        const { container } = renderWithProviders(<CharGrid data={data} />);

        const pre = container.querySelector("pre");
        expect(pre).toBeTruthy();
        // Check that spans with characters are rendered
        const spans = pre!.querySelectorAll("span");
        expect(spans.length).toBe(4); // 2x2 grid = 4 cells
    });

    it("applies foreground and background colors", () => {
        const data = makeGridData();
        const { container } = renderWithProviders(<CharGrid data={data} />);

        const pre = container.querySelector("pre");
        const html = pre!.innerHTML;
        // First cell: fg=ffffff, bg=000000
        expect(html).toContain('color:#ffffff');
        expect(html).toContain('background:#000000');
    });

    it("handles colors with # prefix", () => {
        const data = makeGridData({
            chars: [[65]],
            fg: [["#aabbcc"]],
            bg: [["#112233"]],
        });
        const { container } = renderWithProviders(<CharGrid data={data} />);

        const pre = container.querySelector("pre");
        const html = pre!.innerHTML;
        expect(html).toContain('color:#aabbcc');
        expect(html).toContain('background:#112233');
    });

    it("escapes HTML characters", () => {
        const data = makeGridData({
            chars: [[60, 62, 38]], // < > &
            fg: [["ffffff", "ffffff", "ffffff"]],
            bg: [["000000", "000000", "000000"]],
        });
        const { container } = renderWithProviders(<CharGrid data={data} />);

        const pre = container.querySelector("pre");
        const html = pre!.innerHTML;
        expect(html).toContain("&lt;");
        expect(html).toContain("&gt;");
        expect(html).toContain("&amp;");
    });

    it("separates rows with newlines", () => {
        const data = makeGridData();
        const { container } = renderWithProviders(<CharGrid data={data} />);

        const pre = container.querySelector("pre");
        const html = pre!.innerHTML;
        expect(html).toContain("\n");
    });
});
