/** Event bus activity history -- stacked area chart of per-bus event counts over steps. */

import { Text } from "@mantine/core";
import { useMemo } from "react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    Legend,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import { useEventHistory } from "../../api/queries";
import { ClickableChart } from "../common/ClickableChart";

// Distinct colors for event bus series
const BUS_COLORS = [
    "#58a6ff",
    "#3fb950",
    "#d29922",
    "#f85149",
    "#bc8cff",
    "#39d2c0",
    "#ff7b72",
    "#79c0ff",
    "#7ee787",
    "#e3b341",
];

interface EventHistoryProps {
    run: string;
    game?: number;
    currentStep: number;
    onStepClick?: (step: number) => void;
}

export function EventHistory({ run, game, currentStep, onStepClick }: Readonly<EventHistoryProps>) {
    const { data: history } = useEventHistory(run, game);

    // Collect all unique bus names across all steps
    const busNames = useMemo(() => {
        if (!history || history.length === 0) return [];
        const names = new Set<string>();
        for (const entry of history) {
            for (const key of Object.keys(entry)) {
                if (key !== "step") names.add(key);
            }
        }
        return Array.from(names).sort((a, b) => a.localeCompare(b));
    }, [history]);

    // Strip "roc." prefix for display
    const displayData = useMemo(() => {
        if (!history) return [];
        return history.map((entry) => {
            const cleaned: Record<string, unknown> = { step: entry.step };
            for (const key of Object.keys(entry)) {
                if (key !== "step") {
                    cleaned[key.replace(/^roc\./, "")] = entry[key];
                }
            }
            return cleaned;
        });
    }, [history]);

    const displayNames = useMemo(
        () => busNames.map((n) => n.replace(/^roc\./, "")),
        [busNames],
    );

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No event history
            </Text>
        );
    }

    const chart = (
        <div>
            <Text size="xs" fw={500} mb={4}>
                Event Activity
            </Text>
            <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={displayData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
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
                    {displayNames.map((name, i) => (
                        <Area
                            key={name}
                            type="monotone"
                            dataKey={name}
                            stackId="events"
                            stroke={BUS_COLORS[i % BUS_COLORS.length]}
                            fill={BUS_COLORS[i % BUS_COLORS.length]}
                            fillOpacity={0.3}
                            dot={false}
                            strokeWidth={1}
                        />
                    ))}
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );

    if (onStepClick) {
        return (
            <ClickableChart onStepClick={onStepClick} data={displayData as { step: number }[]}>
                {chart}
            </ClickableChart>
        );
    }
    return chart;
}
