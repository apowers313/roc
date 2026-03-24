/** Transition panel -- what changed between consecutive frames. */

import { Badge, Group, Stack, Table, Text } from "@mantine/core";

import type { StepData, TransformChangeData } from "../../types/step-data";

interface TransitionPanelProps {
    data: StepData | undefined;
}

/** Normalize a change entry -- old format is a plain string, new format is an object. */
function normalizeChange(ch: TransformChangeData | string): TransformChangeData {
    if (typeof ch === "string") {
        return { description: ch };
    }
    return ch;
}

export function TransitionPanel({ data }: TransitionPanelProps) {
    const t = data?.transform_summary;

    if (!t) {
        return (
            <Text size="xs" c="dimmed">
                No transform data
            </Text>
        );
    }

    if (t.count === 0) {
        return (
            <Text size="xs" c="dimmed">
                No changes this step
            </Text>
        );
    }

    const changes = t.changes.map(normalizeChange);

    // Check if we have structured data (type/name fields) or just descriptions
    const hasStructured = changes.some((ch) => ch.type != null);

    return (
        <Stack gap="sm">
            <Group gap="xs">
                <Text size="sm" fw={600}>Changes</Text>
                <Badge variant="filled" color="orange" size="sm">
                    {t.count}
                </Badge>
            </Group>

            <Table
                striped
                highlightOnHover
                withTableBorder
                withColumnBorders
                fz="xs"
            >
                <Table.Thead>
                    <Table.Tr>
                        {hasStructured ? (
                            <>
                                <Table.Th>Type</Table.Th>
                                <Table.Th>Name</Table.Th>
                                <Table.Th>Delta</Table.Th>
                            </>
                        ) : (
                            <>
                                <Table.Th>#</Table.Th>
                                <Table.Th>Change</Table.Th>
                            </>
                        )}
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {changes.map((ch, i) => (
                        <Table.Tr key={i}>
                            {hasStructured ? (
                                <>
                                    <Table.Td>
                                        <Badge
                                            variant="light"
                                            color="grape"
                                            size="xs"
                                        >
                                            {ch.type ?? "unknown"}
                                        </Badge>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {ch.name ?? ch.description}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text
                                            size="xs"
                                            ff="monospace"
                                            c={ch.normalized_change != null
                                                ? ch.normalized_change > 0 ? "green" : "red"
                                                : undefined}
                                        >
                                            {ch.normalized_change != null
                                                ? (ch.normalized_change > 0 ? "+" : "") +
                                                  ch.normalized_change.toFixed(4)
                                                : "--"}
                                        </Text>
                                    </Table.Td>
                                </>
                            ) : (
                                <>
                                    <Table.Td>{i + 1}</Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace" lineClamp={2}>
                                            {ch.description}
                                        </Text>
                                    </Table.Td>
                                </>
                            )}
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </Stack>
    );
}
