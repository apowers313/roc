/** Prediction panel -- prediction status, confidence, predicted intrinsics. */

import { Badge, Group, Progress, Stack, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface PredictionPanelProps {
    data: StepData | undefined;
}

export function PredictionPanel({ data }: PredictionPanelProps) {
    const p = data?.prediction;

    if (!p) {
        return (
            <Text size="xs" c="dimmed">
                No prediction data
            </Text>
        );
    }

    const predictedKeys = p.predicted_intrinsics
        ? Object.keys(p.predicted_intrinsics).sort()
        : [];

    return (
        <Stack gap="sm">
            {/* Status row */}
            <Group gap="md">
                <Badge
                    color={p.made ? "green" : "orange"}
                    variant="filled"
                    size="sm"
                >
                    {p.made ? "PREDICTED" : "NO PREDICTION"}
                </Badge>

                {p.candidate_count != null && (
                    <Group gap={4}>
                        <Text size="xs" c="dimmed">Candidates</Text>
                        <Text size="xs" ff="monospace" fw={600}>
                            {p.candidate_count}
                        </Text>
                    </Group>
                )}

                {p.confidence != null && (
                    <Group gap={4}>
                        <Text size="xs" c="dimmed">Confidence</Text>
                        <Text size="xs" ff="monospace" fw={600}>
                            {Number(p.confidence).toFixed(3)}
                        </Text>
                    </Group>
                )}
            </Group>

            {/* Score distribution */}
            {p.all_scores != null && p.all_scores.length > 1 && (
                <Group gap="xs">
                    <Text size="xs" c="dimmed">Scores:</Text>
                    {p.all_scores.map((s, i) => (
                        <Badge
                            key={i}
                            variant={s === p.confidence ? "filled" : "light"}
                            color={s === p.confidence ? "green" : "gray"}
                            size="xs"
                        >
                            {s.toFixed(3)}
                        </Badge>
                    ))}
                </Group>
            )}

            {/* Predicted intrinsics */}
            {predictedKeys.length > 0 && p.predicted_intrinsics && (
                <Stack gap="xs">
                    <Text size="xs" fw={600}>Predicted Intrinsics</Text>
                    {predictedKeys.map((k) => {
                        const norm = p.predicted_intrinsics![k] ?? 0;
                        return (
                            <div key={k}>
                                <Group justify="space-between" gap={4}>
                                    <Text size="xs" c="dimmed" style={{ width: 100 }}>
                                        {k}
                                    </Text>
                                    <Text size="xs" ff="monospace">
                                        {norm.toFixed(4)}
                                    </Text>
                                </Group>
                                <Progress
                                    value={Math.min(Math.max(norm * 100, 0), 100)}
                                    size="sm"
                                    color="cyan"
                                />
                            </div>
                        );
                    })}
                </Stack>
            )}
        </Stack>
    );
}
