/** Action histogram -- bar chart of action frequency with log scale. */

import { Text } from "@mantine/core";
import {
    Bar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import type { ActionMapEntry, ActionPoint } from "../../api/client";
import { useActionHistory, useActionMap } from "../../api/queries";

interface ActionHistogramProps {
    run: string;
    game?: number;
}

export interface HistogramBin {
    action_id: number;
    action_name: string;
    action_key: string;
    count: number;
}

/** Build histogram bins from action history + optional action map. Exported for testing. */
export function buildBins(
    history: ActionPoint[],
    actionMap: ActionMapEntry[] | undefined,
): HistogramBin[] {
    // Build lookup from action map (complete), falling back to history data
    const mapLookup = new Map<number, { name: string; key: string }>();
    if (actionMap) {
        for (const entry of actionMap) {
            mapLookup.set(entry.action_id, {
                name: entry.action_name,
                key: entry.action_key ?? "",
            });
        }
    }

    // Aggregate counts per action_id from history
    const counts = new Map<number, number>();
    let maxId = 0;
    for (const pt of history) {
        if (pt.action_id > maxId) maxId = pt.action_id;
        counts.set(pt.action_id, (counts.get(pt.action_id) ?? 0) + 1);
        // Backfill lookup from history if action map is not available
        if (!mapLookup.has(pt.action_id)) {
            mapLookup.set(pt.action_id, {
                name: pt.action_name ?? `Action #${pt.action_id}`,
                key: pt.action_key ?? "",
            });
        }
    }

    // Use action map max if available (covers actions never taken)
    if (actionMap && actionMap.length > 0) {
        const mapMax = actionMap[actionMap.length - 1]!.action_id;
        if (mapMax > maxId) maxId = mapMax;
    }

    // Fill in all IDs from 0 to max so the x axis is always complete
    const bins: HistogramBin[] = [];
    for (let id = 0; id <= maxId; id++) {
        const info = mapLookup.get(id);
        bins.push({
            action_id: id,
            action_name: info?.name ?? `Action #${id}`,
            action_key: info?.key ?? "",
            count: counts.get(id) ?? 0,
        });
    }
    return bins;
}

export function ActionHistogram({ run, game }: Readonly<ActionHistogramProps>) {
    const { data: history } = useActionHistory(run, game);
    const { data: actionMap } = useActionMap(run);

    if (!history || history.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No action history
            </Text>
        );
    }

    const bins = buildBins(history, actionMap);

    return (
        <div>
            <Text size="xs" fw={600} mb={4}>
                Action Frequency
            </Text>
            <ResponsiveContainer width="100%" height={240}>
                <BarChart
                    data={bins}
                    margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis
                        dataKey="action_id"
                        tick={{ fontSize: 10, fill: "#888" }}
                        tickLine={false}
                        label={{
                            value: "Action",
                            position: "insideBottom",
                            offset: -2,
                            fontSize: 10,
                            fill: "#888",
                        }}
                    />
                    <YAxis
                        scale="log"
                        domain={[0.5, "auto"]}
                        allowDataOverflow
                        tick={{ fontSize: 10, fill: "#888" }}
                        tickLine={false}
                        width={45}
                        tickFormatter={(v: number) => (v < 1 ? "0" : String(Math.round(v)))}
                        label={{
                            value: "Count",
                            angle: -90,
                            position: "insideLeft",
                            offset: 10,
                            fontSize: 10,
                            fill: "#888",
                        }}
                    />
                    <Tooltip
                        contentStyle={{
                            background: "#25262b",
                            border: "1px solid #333",
                            fontSize: 11,
                        }}
                        labelStyle={{ color: "#888" }}
                        labelFormatter={(_label, payload) => {
                            if (payload && payload.length > 0) {
                                const bin = payload[0]?.payload as HistogramBin | undefined;
                                if (bin) {
                                    const keyPart = bin.action_key ? `key: ${bin.action_key}, ` : "";
                                    return `${bin.action_name} (${keyPart}id: ${bin.action_id})`;
                                }
                            }
                            return `Action ${String(_label)}`;
                        }}
                        formatter={(value: number) => [value, "Count"]}
                    />
                    <Bar
                        dataKey="count"
                        fill="#58a6ff"
                        fillOpacity={0.7}
                        stroke="#58a6ff"
                        name="Count"
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
