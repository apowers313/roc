/** Resolution accuracy chart -- correct matches, incorrect matches, new objects over time. */

import { Text } from "@mantine/core";
import { memo } from "react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import { useResolutionHistory } from "../../api/queries";
import { ClickableChart } from "../common/ClickableChart";

interface ResolutionChartProps {
    run: string;
    game?: number;
    currentStep: number;
    onStepClick?: (step: number) => void;
}

interface BucketedPoint {
    step: number;
    correct: number;
    incorrect: number;
    new_object: number;
    unknown: number;
}

function ResolutionChartInner({ run, game, currentStep, onStepClick }: Readonly<ResolutionChartProps>) {
    const { data: history } = useResolutionHistory(run, game);

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No resolution history
            </Text>
        );
    }

    const title = "Object Resolution Error Rate";

    // Compute cumulative counts
    const cumulative: BucketedPoint[] = [];
    let correct = 0;
    let incorrect = 0;
    let newObj = 0;
    let unknown = 0;

    for (const pt of history) {
        if (pt.outcome === "new_object") {
            newObj++;
        } else if (pt.outcome === "match") {
            if (pt.correct === true) correct++;
            else if (pt.correct === false) incorrect++;
            else unknown++;
        }
        cumulative.push({
            step: pt.step,
            correct,
            incorrect,
            new_object: newObj,
            unknown,
        });
    }

    return (
        <div>
        <Text size="xs" fw={600} mb={4}>{title}</Text>
        <ClickableChart onStepClick={onStepClick ?? (() => {})} data={cumulative}>
            <ResponsiveContainer width="100%" height={160}>
                <AreaChart
                    data={cumulative}
                    margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
                >
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
                    />
                    <Tooltip
                        contentStyle={{
                            background: "#25262b",
                            border: "1px solid #333",
                            fontSize: 11,
                        }}
                        labelStyle={{ color: "#888" }}
                    />
                    <ReferenceLine
                        x={currentStep}
                        stroke="#666"
                        strokeDasharray="3 3"
                    />
                    <Area
                        type="stepAfter"
                        dataKey="correct"
                        stackId="1"
                        stroke="#3fb950"
                        fill="#3fb950"
                        fillOpacity={0.4}
                        name="Correct Match"
                    />
                    <Area
                        type="stepAfter"
                        dataKey="incorrect"
                        stackId="1"
                        stroke="#f85149"
                        fill="#f85149"
                        fillOpacity={0.4}
                        name="Incorrect Match"
                    />
                    <Area
                        type="stepAfter"
                        dataKey="unknown"
                        stackId="1"
                        stroke="#d29922"
                        fill="#d29922"
                        fillOpacity={0.3}
                        name="Unknown Match"
                    />
                    <Area
                        type="stepAfter"
                        dataKey="new_object"
                        stackId="1"
                        stroke="#58a6ff"
                        fill="#58a6ff"
                        fillOpacity={0.3}
                        name="New Object"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </ClickableChart>
        </div>
    );
}

export const ResolutionChart = memo(ResolutionChartInner);
