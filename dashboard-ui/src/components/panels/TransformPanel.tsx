/** Transform & Prediction panel -- what changed + prediction status. */

import { Badge, Group, Stack, Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface TransformPanelProps {
    data: StepData | undefined;
}

export function TransformPanel({ data }: TransformPanelProps) {
    const t = data?.transform_summary;
    const p = data?.prediction;

    return (
        <Stack gap="sm">
            <Group gap="xs">
                <Text size="sm" fw={600}>
                    Prediction
                </Text>
                {p ? (
                    <Badge
                        color={p.made ? "green" : "orange"}
                        variant="filled"
                        size="sm"
                    >
                        {p.made ? "PREDICTED" : "NO PREDICTION"}
                    </Badge>
                ) : (
                    <Text size="xs" c="dimmed">
                        --
                    </Text>
                )}
                {p?.candidates != null && (
                    <Text size="xs" c="dimmed">
                        {p.candidates} candidates
                    </Text>
                )}
                {p?.confidence != null && (
                    <Text size="xs" c="dimmed">
                        conf: {Number(p.confidence).toFixed(3)}
                    </Text>
                )}
                {p?.candidate_expmod && (
                    <Badge size="xs" variant="light" color="grape">
                        {p.candidate_expmod}
                    </Badge>
                )}
                {p?.confidence_expmod && (
                    <Badge size="xs" variant="light" color="grape">
                        {p.confidence_expmod}
                    </Badge>
                )}
            </Group>

            {t && t.count > 0 ? (
                <Table
                    striped
                    highlightOnHover
                    withTableBorder
                    withColumnBorders
                    fz="xs"
                >
                    <Table.Thead>
                        <Table.Tr>
                            <Table.Th>#</Table.Th>
                            <Table.Th>Change</Table.Th>
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {t.changes.map((ch, i) => (
                            <Table.Tr key={i}>
                                <Table.Td>{i + 1}</Table.Td>
                                <Table.Td>
                                    <Text size="xs" ff="monospace" lineClamp={2}>
                                        {ch}
                                    </Text>
                                </Table.Td>
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            ) : (
                <Text size="xs" c="dimmed">
                    {t ? "No changes this step" : "No transform data"}
                </Text>
            )}
        </Stack>
    );
}
