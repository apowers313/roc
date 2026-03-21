/** Aural Perception panel -- displays the message and its phoneme decomposition. */

import { Paper, Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface AuralPerceptionProps {
    data: StepData | undefined;
}

export function AuralPerception({ data }: AuralPerceptionProps) {
    const msg = data?.message;
    const phonemes = data?.phonemes;

    if (!msg && !phonemes) {
        return (
            <Text size="xs" c="dimmed">
                No auditory data this step
            </Text>
        );
    }

    return (
        <>
            {msg && (
                <Paper p="xs" withBorder mb="xs">
                    <Text size="xs" fw={600} c="dimmed" mb={4}>
                        Message
                    </Text>
                    <Text size="sm" ff="monospace">
                        {msg}
                    </Text>
                </Paper>
            )}
            {phonemes && phonemes.length > 0 && (
                <Paper p="xs" withBorder>
                    <Text size="xs" fw={600} c="dimmed" mb={4}>
                        Phonemes
                    </Text>
                    <Table
                        striped
                        highlightOnHover
                        withTableBorder
                        withColumnBorders
                        fz="xs"
                    >
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th>Word</Table.Th>
                                <Table.Th>IPA</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {phonemes.map((entry, i) => (
                                <Table.Tr
                                    key={i}
                                    style={
                                        entry.is_break
                                            ? { opacity: 0.5, fontStyle: "italic" }
                                            : undefined
                                    }
                                >
                                    <Table.Td ff="monospace">{entry.word}</Table.Td>
                                    <Table.Td ff="monospace">
                                        {entry.phonemes.join(" ")}
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </Paper>
            )}
        </>
    );
}
