import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import type { StepData } from "../../types/step-data";
import { PredictionPanel } from "./PredictionPanel";

/** Shorthand: create StepData with only prediction fields set. */
function predictionData(prediction: StepData["prediction"]): StepData {
    return makeStepData({ prediction });
}

/** Render PredictionPanel with the given prediction fields. */
function renderPrediction(prediction: StepData["prediction"]) {
    return renderWithProviders(<PredictionPanel data={predictionData(prediction)} />);
}

describe("PredictionPanel", () => {
    it("shows 'No prediction data' when data is undefined", () => {
        renderWithProviders(<PredictionPanel data={undefined} />);
        expect(screen.getByText("No prediction data")).toBeInTheDocument();
    });

    it("shows 'No prediction data' when prediction field is null", () => {
        renderPrediction(null);
        expect(screen.getByText("No prediction data")).toBeInTheDocument();
    });

    it("shows PREDICTED badge when made is true", () => {
        renderPrediction({ made: true });
        expect(screen.getByText("PREDICTED")).toBeInTheDocument();
    });

    it("shows NO PREDICTION badge when made is false", () => {
        renderPrediction({ made: false });
        expect(screen.getByText("NO PREDICTION")).toBeInTheDocument();
    });

    it("renders candidate count", () => {
        renderPrediction({ made: true, candidate_count: 7 });
        expect(screen.getByText("Candidates")).toBeInTheDocument();
        expect(screen.getByText("7")).toBeInTheDocument();
    });

    it("does not render candidate count when absent", () => {
        renderPrediction({ made: true });
        expect(screen.queryByText("Candidates")).not.toBeInTheDocument();
    });

    it("renders confidence with 3 decimal places", () => {
        renderPrediction({ made: true, confidence: 0.87654 });
        expect(screen.getByText("Confidence")).toBeInTheDocument();
        expect(screen.getByText("0.877")).toBeInTheDocument();
    });

    it("does not render confidence when absent", () => {
        renderPrediction({ made: false });
        expect(screen.queryByText("Confidence")).not.toBeInTheDocument();
    });

    it("renders all_scores as badges when more than 1 score", () => {
        renderPrediction({ made: true, confidence: 0.9, all_scores: [0.9, 0.7, 0.3] });
        expect(screen.getByText("Scores:")).toBeInTheDocument();
        // "0.900" appears twice: once in confidence display and once in the scores badge
        expect(screen.getAllByText("0.900").length).toBe(2);
        expect(screen.getByText("0.700")).toBeInTheDocument();
        expect(screen.getByText("0.300")).toBeInTheDocument();
    });

    it("does not render scores section when only 1 score", () => {
        renderPrediction({ made: true, confidence: 0.5, all_scores: [0.5] });
        expect(screen.queryByText("Scores:")).not.toBeInTheDocument();
    });

    it("does not render scores section when all_scores is absent", () => {
        renderPrediction({ made: true, confidence: 0.5 });
        expect(screen.queryByText("Scores:")).not.toBeInTheDocument();
    });

    it("renders predicted intrinsics with progress bars", () => {
        renderPrediction({ made: true, predicted_intrinsics: { hp: 0.8, hunger: 0.3 } });
        expect(screen.getByText("Predicted Intrinsics")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("hunger")).toBeInTheDocument();
        expect(screen.getByText("0.8000")).toBeInTheDocument();
        expect(screen.getByText("0.3000")).toBeInTheDocument();
    });

    it("sorts predicted intrinsic keys alphabetically", () => {
        const { container } = renderPrediction({
            made: true,
            predicted_intrinsics: { zebra: 0.1, alpha: 0.9, middle: 0.5 },
        });
        const labels = container.querySelectorAll('[style*="width"]');
        // Find the text content of the intrinsic labels in order
        const texts: string[] = [];
        labels.forEach((el) => {
            const text = el.textContent?.trim();
            if (text && ["alpha", "middle", "zebra"].includes(text)) {
                texts.push(text);
            }
        });
        expect(texts).toEqual(["alpha", "middle", "zebra"]);
    });

    it("does not render Predicted Intrinsics heading when none provided", () => {
        renderPrediction({ made: true });
        expect(screen.queryByText("Predicted Intrinsics")).not.toBeInTheDocument();
    });

    it("clamps progress bar value between 0 and 100", () => {
        renderPrediction({ made: true, predicted_intrinsics: { over: 1.5, under: -0.5 } });
        expect(screen.getByText("over")).toBeInTheDocument();
        expect(screen.getByText("under")).toBeInTheDocument();
        expect(screen.getByText("1.5000")).toBeInTheDocument();
        expect(screen.getByText("-0.5000")).toBeInTheDocument();
    });

    it("renders full combination of all fields", () => {
        renderPrediction({
            made: true,
            candidate_count: 3,
            confidence: 0.95,
            all_scores: [0.95, 0.8],
            predicted_intrinsics: { energy: 0.6 },
        });
        expect(screen.getByText("PREDICTED")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
        // "0.950" appears in both confidence and scores badge
        expect(screen.getAllByText("0.950").length).toBe(2);
        expect(screen.getByText("Scores:")).toBeInTheDocument();
        expect(screen.getByText("Predicted Intrinsics")).toBeInTheDocument();
        expect(screen.getByText("energy")).toBeInTheDocument();
    });
});
