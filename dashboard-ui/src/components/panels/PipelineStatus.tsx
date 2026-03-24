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

function getPredictionColor(p: StepData["prediction"]): string {
    if (!p) return "gray";
    return p.made ? "green" : "orange";
}

function getSignificanceColor(sig: number | null | undefined): string {
    if (sig == null) return "gray";
    if (sig > 0.5) return "red";
    if (sig > 0.1) return "yellow";
    return "green";
}

function getStages(data: StepData | undefined): StageCard[] {
    if (!data) return [];

    const stages: StageCard[] = [];

    // Resolution
    const res = data.resolution_metrics;
    let resValue = "--";
    if (res) {
        resValue = typeof res.outcome === "string" ? res.outcome : "done";
    }
    stages.push({
        label: "Resolution",
        value: resValue,
        color: res ? "teal" : "gray",
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
    let predValue = "--";
    if (p) {
        predValue = p.made ? "MADE" : "SKIP";
    }
    const predColor = getPredictionColor(p);
    stages.push({
        label: "Prediction",
        value: predValue,
        color: predColor,
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
    const sigColor = getSignificanceColor(sig);
    stages.push({
        label: "Significance",
        value: sig == null ? "--" : sig.toFixed(3),
        color: sigColor,
    });

    return stages;
}

export function PipelineStatus({ data }: Readonly<PipelineStatusProps>) {
    const stages = getStages(data);

    if (stages.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No pipeline data
            </Text>
        );
    }

    return (
        <SimpleGrid cols={{ base: 3, md: 5 }} spacing="xs">
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
