/** Attention cycle summary table with optional stepper for multi-cycle attention. */

import { SegmentedControl, Stack, Table, Text, UnstyledButton } from "@mantine/core";

import { useHighlight, findHighlightColor } from "../../state/highlight";

export interface CycleSummaryEntry {
    preIorPeak: { x: number; y: number; strength: number };
    postIorPeak: { x: number; y: number; strength: number };
    focusedPoint: { x: number; y: number; strength: number };
}

interface AttentionCycleSummaryProps {
    cycles: CycleSummaryEntry[];
    selectedCycle?: number;
    onCycleChange?: (cycle: number) => void;
}

function ClickablePoint({
    pt,
    label,
    hlColor,
}: Readonly<{
    pt: { x: number; y: number; strength: number };
    label: string;
    hlColor: string | undefined;
}>) {
    const { togglePoint } = useHighlight();
    return (
        <UnstyledButton
            onClick={() => togglePoint({ x: pt.x, y: pt.y, label })}
            style={{
                fontSize: "inherit",
                fontFamily: "monospace",
                textDecoration: "underline dotted",
                background: hlColor ? `${hlColor}26` : undefined,
                borderRadius: 2,
                padding: "0 2px",
            }}
        >
            ({pt.x}, {pt.y}) s={pt.strength.toFixed(2)}
        </UnstyledButton>
    );
}

export function AttentionCycleSummary({
    cycles,
    selectedCycle,
    onCycleChange,
}: Readonly<AttentionCycleSummaryProps>) {
    const { points } = useHighlight();

    if (cycles.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No cycle data
            </Text>
        );
    }

    return (
        <Stack gap="xs">
            <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
                <Table.Thead>
                    <Table.Tr>
                        <Table.Th>#</Table.Th>
                        <Table.Th>Pre-IOR Peak</Table.Th>
                        <Table.Th>Post-IOR Peak</Table.Th>
                        <Table.Th>Focused Point</Table.Th>
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {cycles.map((c, idx) => (
                        <Table.Tr key={`${c.focusedPoint.x}-${c.focusedPoint.y}-${idx}`}>
                            <Table.Td>{idx + 1}</Table.Td>
                            <Table.Td>
                                <ClickablePoint
                                    pt={c.preIorPeak}
                                    label={`pre-IOR #${idx + 1}`}
                                    hlColor={findHighlightColor(points, c.preIorPeak.x, c.preIorPeak.y)}
                                />
                            </Table.Td>
                            <Table.Td>
                                <ClickablePoint
                                    pt={c.postIorPeak}
                                    label={`post-IOR #${idx + 1}`}
                                    hlColor={findHighlightColor(points, c.postIorPeak.x, c.postIorPeak.y)}
                                />
                            </Table.Td>
                            <Table.Td>
                                <ClickablePoint
                                    pt={c.focusedPoint}
                                    label={`focus #${idx + 1}`}
                                    hlColor={findHighlightColor(points, c.focusedPoint.x, c.focusedPoint.y)}
                                />
                            </Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>

            {cycles.length > 1 && onCycleChange && (
                <SegmentedControl
                    size="xs"
                    data={cycles.map((_, i) => ({
                        label: `Cycle ${i + 1}`,
                        value: String(i),
                    }))}
                    value={String(selectedCycle ?? 0)}
                    onChange={(v) => onCycleChange(Number(v))}
                />
            )}
        </Stack>
    );
}
