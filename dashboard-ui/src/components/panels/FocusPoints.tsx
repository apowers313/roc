/** Focus points panel -- attention focus point coordinates. */

import { Card, Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface FocusPointsProps {
    data: StepData | undefined;
}

/** Parse a pandas DataFrame string into rows of {x, y, strength, label}. */
function parseFocusRaw(raw: string): Array<Record<string, string>> {
    const lines = raw.trim().split("\n");
    if (lines.length < 2) return [];

    // Header line: "    x   y  strength  label"
    const header = lines[0]!.trim().split(/\s+/);
    const rows: Array<Record<string, string>> = [];
    for (let i = 1; i < lines.length; i++) {
        // Data lines: "0  28  18  0.933333      3"
        const parts = lines[i]!.trim().split(/\s+/);
        // First element is the index, skip it
        const row: Record<string, string> = {};
        for (let j = 0; j < header.length; j++) {
            row[header[j]!] = parts[j + 1] ?? "";
        }
        rows.push(row);
    }
    return rows;
}

export function FocusPoints({ data }: FocusPointsProps) {
    if (!data?.focus_points || data.focus_points.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No focus data
            </Text>
        );
    }

    const fp = data.focus_points[0]!;
    const rows = fp.raw ? parseFocusRaw(String(fp.raw)) : [];

    if (rows.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No focus data
            </Text>
        );
    }

    return (
        <Card padding={6} radius="sm" withBorder>
            <Text size="xs" fw={600} c="dimmed" mb={2}>
                Focus Points
            </Text>
            <div style={{ maxHeight: 120, overflowY: "auto" }}>
                <Table
                    layout="fixed"
                    horizontalSpacing={4}
                    verticalSpacing={0}
                    withRowBorders={false}
                    style={{ fontVariantNumeric: "tabular-nums" }}
                >
                    <Table.Thead>
                        <Table.Tr>
                            <Table.Th style={{ width: 30, padding: 0 }}>
                                <Text size="xs" c="dimmed">x</Text>
                            </Table.Th>
                            <Table.Th style={{ width: 30, padding: 0 }}>
                                <Text size="xs" c="dimmed">y</Text>
                            </Table.Th>
                            <Table.Th style={{ width: 60, padding: 0, textAlign: "right" }}>
                                <Text size="xs" c="dimmed">str</Text>
                            </Table.Th>
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {rows.map((row, i) => (
                            <Table.Tr key={i}>
                                <Table.Td style={{ padding: 0 }}>
                                    <Text size="xs">{row.x}</Text>
                                </Table.Td>
                                <Table.Td style={{ padding: 0 }}>
                                    <Text size="xs">{row.y}</Text>
                                </Table.Td>
                                <Table.Td style={{ padding: 0, textAlign: "right" }}>
                                    <Text size="xs">
                                        {Number(row.strength).toFixed(2)}
                                    </Text>
                                </Table.Td>
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            </div>
        </Card>
    );
}
