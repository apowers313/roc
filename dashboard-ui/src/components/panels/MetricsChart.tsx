/** Metrics trend chart -- HP and score over time using recharts. */

import { Text } from "@mantine/core";
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import { useMetricsHistory } from "../../api/queries";
import { ClickableChart } from "../common/ClickableChart";

const CHART_FIELDS = ["hp", "hp_max", "score", "energy", "energy_max"];
const LINE_COLORS: Record<string, string> = {
    hp: "#3fb950",
    hp_max: "#238636",
    score: "#58a6ff",
    energy: "#d29922",
    energy_max: "#8b6914",
};

interface MetricsChartProps {
    run: string;
    game?: number;
    currentStep: number;
    onStepClick?: (step: number) => void;
}

export function MetricsChart({ run, game, currentStep, onStepClick }: MetricsChartProps) {
    const { data: history } = useMetricsHistory(run, game, CHART_FIELDS);

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No metrics history
            </Text>
        );
    }

    const chart = (
        <ResponsiveContainer width="100%" height={180}>
            <LineChart data={history} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: "#888" }}
                    tickLine={false}
                />
                <YAxis tick={{ fontSize: 10, fill: "#888" }} tickLine={false} width={40} />
                <Tooltip
                    contentStyle={{
                        background: "#25262b",
                        border: "1px solid #333",
                        fontSize: 11,
                    }}
                    labelStyle={{ color: "#888" }}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <ReferenceLine x={currentStep} stroke="#666" strokeDasharray="3 3" />
                {CHART_FIELDS.map((field) => (
                    <Line
                        key={field}
                        type="monotone"
                        dataKey={field}
                        stroke={LINE_COLORS[field] ?? "#888"}
                        dot={false}
                        strokeWidth={field.endsWith("_max") ? 1 : 1.5}
                        strokeDasharray={field.endsWith("_max") ? "4 2" : undefined}
                        connectNulls
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );

    if (onStepClick) {
        return (
            <ClickableChart onStepClick={onStepClick} data={history}>
                {chart}
            </ClickableChart>
        );
    }
    return chart;
}
