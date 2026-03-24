/** Sequence panel -- frame composition: objects, intrinsics, significance. */

import { Badge, Group, Progress, Stack, Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface SequencePanelProps {
    data: StepData | undefined;
}

export function SequencePanel({ data }: SequencePanelProps) {
    const seq = data?.sequence_summary;

    if (!seq) {
        return (
            <Text size="xs" c="dimmed">
                No sequence data
            </Text>
        );
    }

    const intrinsicKeys = Object.keys(seq.intrinsics).sort();

    return (
        <Stack gap="sm">
            {/* Header stats */}
            <Group gap="md">
                <Group gap={4}>
                    <Text size="xs" c="dimmed">Tick</Text>
                    <Badge variant="light" color="blue" size="sm">
                        {seq.tick}
                    </Badge>
                </Group>
                <Group gap={4}>
                    <Text size="xs" c="dimmed">Objects</Text>
                    <Badge variant="light" color="violet" size="sm">
                        {seq.object_count}
                    </Badge>
                </Group>
                <Group gap={4}>
                    <Text size="xs" c="dimmed">Intrinsics</Text>
                    <Badge variant="light" color="teal" size="sm">
                        {seq.intrinsic_count}
                    </Badge>
                </Group>
                {seq.significance != null && (
                    <Group gap={4}>
                        <Text size="xs" c="dimmed">Significance</Text>
                        <Badge
                            variant="filled"
                            color={seq.significance > 0.5 ? "red" : seq.significance > 0.1 ? "yellow" : "green"}
                            size="sm"
                        >
                            {seq.significance.toFixed(4)}
                        </Badge>
                    </Group>
                )}
            </Group>

            {/* Objects table */}
            {seq.objects.length > 0 && (
                <Table
                    striped
                    highlightOnHover
                    withTableBorder
                    withColumnBorders
                    fz="xs"
                >
                    <Table.Thead>
                        <Table.Tr>
                            <Table.Th>ID</Table.Th>
                            <Table.Th>Position</Table.Th>
                            <Table.Th>Resolves</Table.Th>
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {seq.objects.map((obj) => (
                            <Table.Tr key={obj.id}>
                                <Table.Td>
                                    <Text size="xs" ff="monospace">{obj.id}</Text>
                                </Table.Td>
                                <Table.Td>
                                    <Text size="xs" ff="monospace">
                                        {obj.x != null && obj.y != null
                                            ? `(${obj.x}, ${obj.y})`
                                            : "--"}
                                    </Text>
                                </Table.Td>
                                <Table.Td>
                                    <Text size="xs" ff="monospace">
                                        {obj.resolve_count ?? "--"}
                                    </Text>
                                </Table.Td>
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            )}

            {/* Intrinsic bars */}
            {intrinsicKeys.length > 0 && (
                <Stack gap="xs">
                    <Text size="xs" fw={600}>Frame Intrinsics</Text>
                    {intrinsicKeys.map((k) => {
                        const norm = seq.intrinsics[k] ?? 0;
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
                                    color={norm > 0.7 ? "red" : norm > 0.3 ? "yellow" : "blue"}
                                />
                            </div>
                        );
                    })}
                </Stack>
            )}
        </Stack>
    );
}
