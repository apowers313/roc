/** Graph DB cache history -- node/edge counts over time as an area chart. */

import { Text } from "@mantine/core";
import { memo } from "react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    Legend,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import { useGraphHistory } from "../../api/queries";
import { ClickableChart } from "../common/ClickableChart";

interface GraphHistoryProps {
    run: string;
    game?: number;
    currentStep: number;
    onStepClick?: (step: number) => void;
}

function GraphHistoryInner({ run, game, currentStep, onStepClick }: Readonly<GraphHistoryProps>) {
    const { data: history } = useGraphHistory(run, game);

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No graph history
            </Text>
        );
    }

    const chart = (
        <div>
            <Text size="xs" fw={500} mb={4}>
                Graph Cache
            </Text>
            <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={history} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
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
                    <Area
                        type="monotone"
                        dataKey="node_count"
                        name="nodes"
                        stroke="#3fb950"
                        fill="#3fb950"
                        fillOpacity={0.15}
                        dot={false}
                        strokeWidth={1.5}
                    />
                    <Area
                        type="monotone"
                        dataKey="edge_count"
                        name="edges"
                        stroke="#58a6ff"
                        fill="#58a6ff"
                        fillOpacity={0.15}
                        dot={false}
                        strokeWidth={1.5}
                    />
                    <Line
                        type="monotone"
                        dataKey="node_max"
                        name="node limit"
                        stroke="#238636"
                        dot={false}
                        strokeWidth={1}
                        strokeDasharray="4 2"
                    />
                    <Line
                        type="monotone"
                        dataKey="edge_max"
                        name="edge limit"
                        stroke="#1f6feb"
                        dot={false}
                        strokeWidth={1}
                        strokeDasharray="4 2"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
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

export const GraphHistory = memo(GraphHistoryInner);
