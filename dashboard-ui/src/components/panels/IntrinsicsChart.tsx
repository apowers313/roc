/** Intrinsics trend chart -- normalized intrinsic values over time. */

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

import { useIntrinsicsHistory } from "../../api/queries";
import { ClickableChart } from "../common/ClickableChart";

const LINE_COLORS = [
    "#3fb950", "#58a6ff", "#d29922", "#f47067", "#bc8cff",
    "#39d353", "#79c0ff", "#e3b341", "#ff7b72", "#d2a8ff",
];

interface IntrinsicsChartProps {
    run: string;
    game?: number;
    currentStep: number;
    onStepClick?: (step: number) => void;
}

export function IntrinsicsChart({ run, game, currentStep, onStepClick }: IntrinsicsChartProps) {
    const { data: history } = useIntrinsicsHistory(run, game);

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No intrinsics history
            </Text>
        );
    }

    // Flatten: each entry has {step, normalized: {hp: 0.5, ...}} -> {step, hp: 0.5, ...}
    const chartData = history.map((pt) => {
        const flat: { step: number; [key: string]: number } = { step: pt.step };
        if (pt.normalized) {
            for (const [k, v] of Object.entries(pt.normalized)) {
                flat[k] = v;
            }
        }
        return flat;
    });

    // Discover keys from first entry
    const allKeys = chartData.length > 0
        ? Object.keys(chartData[0]!).filter((k) => k !== "step")
        : [];

    const chart = (
        <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: "#888" }}
                    tickLine={false}
                />
                <YAxis
                    tick={{ fontSize: 10, fill: "#888" }}
                    tickLine={false}
                    width={40}
                    domain={[0, 1]}
                />
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
                {allKeys.map((field, i) => (
                    <Line
                        key={field}
                        type="monotone"
                        dataKey={field}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        dot={false}
                        strokeWidth={1.5}
                        connectNulls
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );

    if (onStepClick) {
        return (
            <ClickableChart onStepClick={onStepClick} data={chartData}>
                {chart}
            </ClickableChart>
        );
    }
    return chart;
}
