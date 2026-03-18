/** Graph DB summary panel with cache utilization gauges. */

import { Group, Progress, Stack, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface GraphSummaryProps {
    data: StepData | undefined;
}

function CacheGauge({ label, count, max }: { label: string; count: number; max: number }) {
    const pct = max > 0 ? (count / max) * 100 : 0;
    const color = pct > 80 ? "red" : pct > 60 ? "yellow" : "teal";
    return (
        <div>
            <Group justify="space-between" mb={2}>
                <Text size="xs" c="dimmed">
                    {label}
                </Text>
                <Text size="xs" ff="monospace">
                    {count} / {max}
                </Text>
            </Group>
            <Progress value={pct} color={color} size="sm" />
        </div>
    );
}

export function GraphSummary({ data }: GraphSummaryProps) {
    const gs = data?.graph_summary;
    if (!gs) {
        return (
            <Text size="xs" c="dimmed">
                No graph data
            </Text>
        );
    }

    const nodeCount = typeof gs.node_count === "number" ? gs.node_count : 0;
    const nodeMax = typeof gs.node_max === "number" ? gs.node_max : 0;
    const edgeCount = typeof gs.edge_count === "number" ? gs.edge_count : 0;
    const edgeMax = typeof gs.edge_max === "number" ? gs.edge_max : 0;

    return (
        <Stack gap={8}>
            <Text size="xs" fw={500}>
                Graph DB
            </Text>
            <CacheGauge label="Nodes" count={nodeCount} max={nodeMax} />
            <CacheGauge label="Edges" count={edgeCount} max={edgeMax} />
        </Stack>
    );
}
