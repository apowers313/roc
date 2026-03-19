/** Compact pipeline status overview -- one card per processing stage. */

import { Badge, Group, Paper, SimpleGrid, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface PipelineStatusProps {
    data: StepData | undefined;
}

interface StageCard {
    label: string;
    value: string;
    color: string;
}

function getStages(data: StepData | undefined): StageCard[] {
    if (!data) return [];

    const stages: StageCard[] = [];

    // Perception
    const featCount = data.features?.length ?? 0;
    stages.push({
        label: "Perception",
        value: featCount > 0 ? `${featCount} feat` : "--",
        color: featCount > 0 ? "blue" : "gray",
    });

    // Attention
    const fpCount = data.focus_points?.length ?? 0;
    stages.push({
        label: "Attention",
        value: fpCount > 0 ? `${fpCount} focus` : "--",
        color: fpCount > 0 ? "cyan" : "gray",
    });

    // Resolution
    const res = data.resolution_metrics;
    stages.push({
        label: "Resolution",
        value: res ? String(res.outcome ?? "done") : "--",
        color: res ? "teal" : "gray",
    });

    // Graph
    const g = data.graph_summary;
    stages.push({
        label: "Graph",
        value: g ? `${g.node_count}n/${g.edge_count}e` : "--",
        color: g ? "violet" : "gray",
    });

    // Transform
    const t = data.transform_summary;
    stages.push({
        label: "Transform",
        value: t ? `${t.count} changes` : "--",
        color: t && Number(t.count) > 0 ? "yellow" : "gray",
    });

    // Prediction
    const p = data.prediction;
    stages.push({
        label: "Prediction",
        value: p ? (p.made ? "MADE" : "SKIP") : "--",
        color: p ? (p.made ? "green" : "orange") : "gray",
    });

    // Action
    const a = data.action_taken;
    stages.push({
        label: "Action",
        value: a ? (a.action_name ?? `#${a.action_id}`) : "--",
        color: a ? "indigo" : "gray",
    });

    // Significance
    const sig = data.significance;
    stages.push({
        label: "Significance",
        value: sig != null ? sig.toFixed(3) : "--",
        color: sig != null ? (sig > 0.5 ? "red" : sig > 0.1 ? "yellow" : "green") : "gray",
    });

    return stages;
}

export function PipelineStatus({ data }: PipelineStatusProps) {
    const stages = getStages(data);

    if (stages.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No pipeline data
            </Text>
        );
    }

    return (
        <SimpleGrid cols={{ base: 4, md: 8 }} spacing="xs">
            {stages.map((s) => (
                <Paper key={s.label} p="xs" withBorder>
                    <Group justify="space-between" gap={4}>
                        <Text size="xs" c="dimmed" fw={500}>
                            {s.label}
                        </Text>
                        <Badge size="xs" color={s.color} variant="light">
                            {s.value}
                        </Badge>
                    </Group>
                </Paper>
            ))}
        </SimpleGrid>
    );
}
