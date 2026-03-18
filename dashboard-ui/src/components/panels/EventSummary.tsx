/** Event summary panel -- per-step event bus activity as a bar chart. */

import { Text } from "@mantine/core";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { StepData } from "../../types/step-data";

interface EventSummaryProps {
    data: StepData | undefined;
}

export function EventSummary({ data }: EventSummaryProps) {
    const summary = data?.event_summary?.[0] ?? null;

    if (!summary || typeof summary !== "object") {
        return (
            <Text size="xs" c="dimmed">
                No event data
            </Text>
        );
    }

    // Convert {bus_name: count} map to array for recharts
    const chartData = Object.entries(summary)
        .filter(([key]) => key !== "raw")
        .map(([name, count]) => ({
            name: name.replace(/^roc\./, ""),
            count: typeof count === "number" ? count : 0,
        }))
        .sort((a, b) => b.count - a.count);

    if (chartData.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No event data
            </Text>
        );
    }

    return (
        <div>
            <Text size="xs" fw={500} mb={4}>
                Events
            </Text>
            <ResponsiveContainer width="100%" height={Math.max(chartData.length * 20, 80)}>
                <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
                >
                    <XAxis
                        type="number"
                        tick={{ fontSize: 10, fill: "#888" }}
                        tickLine={false}
                    />
                    <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fontSize: 9, fill: "#aaa" }}
                        tickLine={false}
                        width={100}
                    />
                    <Tooltip
                        contentStyle={{
                            background: "#25262b",
                            border: "1px solid #333",
                            fontSize: 11,
                        }}
                    />
                    <Bar dataKey="count" fill="#58a6ff" radius={[0, 2, 2, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
