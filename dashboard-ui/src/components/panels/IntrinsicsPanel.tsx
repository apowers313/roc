/** Intrinsics panel -- normalized gauges + significance badge. */

import { Badge, Group, Progress, Stack, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface IntrinsicsPanelProps {
    data: StepData | undefined;
}

function sigColor(v: number): string {
    if (v > 0.5) return "red";
    if (v > 0.1) return "yellow";
    return "green";
}

export function IntrinsicsPanel({ data }: IntrinsicsPanelProps) {
    const intr = data?.intrinsics;
    const sig = data?.significance;

    if (!intr) {
        return (
            <Text size="xs" c="dimmed">
                No intrinsics data
            </Text>
        );
    }

    const normalized = intr.normalized ?? {};
    const raw = intr.raw ?? {};
    const keys = Object.keys(normalized).sort();

    return (
        <Stack gap="xs">
            {sig != null && (
                <Group gap="xs">
                    <Text size="xs" fw={500}>
                        Significance
                    </Text>
                    <Badge color={sigColor(sig)} variant="filled" size="sm">
                        {sig.toFixed(4)}
                    </Badge>
                </Group>
            )}
            {keys.map((k) => {
                const norm = normalized[k] ?? 0;
                const rawVal = raw[k];
                return (
                    <div key={k}>
                        <Group justify="space-between" gap={4}>
                            <Text size="xs" c="dimmed" style={{ width: 100 }}>
                                {k}
                            </Text>
                            <Text size="xs" ff="monospace">
                                {rawVal != null ? String(rawVal) : "--"}
                            </Text>
                        </Group>
                        <Progress
                            value={Math.min(Math.max(norm * 100, 0), 100)}
                            size="sm"
                            color={norm > 0.7 ? "red" : norm > 0.3 ? "yellow" : "blue"}
                        />
                    </div>
                );
            })}
        </Stack>
    );
}
