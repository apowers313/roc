import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeLog, makeStepData, renderWithProviders } from "../../test-utils";
import { LogMessages } from "./LogMessages";

describe("LogMessages", () => {
    it("shows 'No log messages' when data is undefined", () => {
        renderWithProviders(<LogMessages data={undefined} />);
        expect(screen.getByText("No log messages")).toBeInTheDocument();
    });

    it("shows 'No log messages' when logs is empty", () => {
        renderWithProviders(
            <LogMessages data={makeStepData({ logs: [] })} />,
        );
        expect(screen.getByText("No log messages")).toBeInTheDocument();
    });

    it("renders log entries with severity and body", () => {
        const data = makeStepData({
            logs: [
                makeLog({ body: "hello world", severity_text: "INFO" }),
                makeLog({ body: "uh oh", severity_text: "ERROR" }),
            ],
        });
        const { container } = renderWithProviders(<LogMessages data={data} />);
        expect(screen.getByText("hello world")).toBeInTheDocument();
        expect(screen.getByText("uh oh")).toBeInTheDocument();
        // Check severity in table cells (avoid matching Select dropdown options)
        const cells = container.querySelectorAll("td");
        const cellTexts = Array.from(cells).map((c) => c.textContent);
        expect(cellTexts).toContain("ERROR");
        expect(cellTexts).toContain("INFO");
    });

    it("defaults severity to INFO when missing", () => {
        const data = makeStepData({
            logs: [makeLog({ severity_text: undefined, body: "no level" })],
        });
        renderWithProviders(<LogMessages data={data} />);
        expect(screen.getByText("no level")).toBeInTheDocument();
        // The severity column shows INFO (default), but "INFO" also appears
        // in the Select dropdown, so check via table cells.
        const { container } = renderWithProviders(<LogMessages data={data} />);
        const cells = container.querySelectorAll("td");
        const severityTexts = Array.from(cells).map((c) => c.textContent);
        expect(severityTexts).toContain("INFO");
    });

    it("renders without error when body is undefined", () => {
        const data = makeStepData({
            logs: [makeLog({ body: undefined, severity_text: "WARN" })],
        });
        const { container } = renderWithProviders(<LogMessages data={data} />);
        // WARN appears in both Select options and the table -- check table cells
        const cells = container.querySelectorAll("td");
        const cellTexts = Array.from(cells).map((c) => c.textContent);
        expect(cellTexts).toContain("WARN");
    });

    it("renders duplicate log entries without React key warnings", () => {
        // Regression: during a live game, the attenuation DEBUG records
        // from the saliency pipeline are emitted many times per step with
        // identical timestamp and body. The pre-fix key (level + timestamp
        // + body-prefix) collided, producing "Encountered two children
        // with the same key" warnings that flooded the console and
        // prompted React to unmount/remount rows on every render. The fix
        // includes the array index in the key so every row is unique.
        const warnings: unknown[][] = [];
        const originalWarn = console.error;
        console.error = (...args: unknown[]) => {
            warnings.push(args);
        };
        try {
            const duplicateBody =
                "attenuation: 5 history entries, max_penalty=1.000, 61 cells attenuated";
            const data = makeStepData({
                logs: [
                    makeLog({
                        body: duplicateBody,
                        severity_text: "DEBUG",
                        timestamp: 1775860974437,
                    }),
                    makeLog({
                        body: duplicateBody,
                        severity_text: "DEBUG",
                        timestamp: 1775860974437,
                    }),
                    makeLog({
                        body: duplicateBody,
                        severity_text: "DEBUG",
                        timestamp: 1775860974437,
                    }),
                ],
            });
            renderWithProviders(<LogMessages data={data} />);
            const duplicateKeyWarnings = warnings.filter((args) =>
                args.some(
                    (a) =>
                        typeof a === "string" &&
                        a.includes("Encountered two children with the same key"),
                ),
            );
            expect(duplicateKeyWarnings).toHaveLength(0);
        } finally {
            console.error = originalWarn;
        }
    });
});
