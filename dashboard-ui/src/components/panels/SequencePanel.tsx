/** Sequence panel -- frame composition: objects, intrinsics, significance. */

import { Stack, Table, Text } from "@mantine/core";

import type { StepData, TransformChangeData } from "../../types/step-data";
import { ObjectLink } from "../common/ObjectLink";

interface SequencePanelProps {
    data: StepData | undefined;
}

/** Check if an intrinsic name appears in transform_summary.changes. */
function isIntrinsicMatched(
    name: string,
    changes: (TransformChangeData | string)[] | undefined,
): boolean {
    if (!changes) return false;
    return changes.some((ch) => {
        if (typeof ch === "string") return false;
        return ch.name === name;
    });
}

export function SequencePanel({ data }: Readonly<SequencePanelProps>) {
    const seq = data?.sequence_summary;

    if (!seq) {
        return (
            <Text size="xs" c="dimmed">
                No sequence data
            </Text>
        );
    }

    const intrinsicKeys = Object.keys(seq.intrinsics).sort((a, b) => a.localeCompare(b));
    const rawIntrinsics = data?.intrinsics?.raw;
    const transformChanges = data?.transform_summary?.changes;

    return (
        <Stack gap="sm">
            {/* Header -- single-line format per design Section 11.4 */}
            <Text size="xs" c="dimmed" ff="monospace">
                {"Frame: tick=" + seq.tick + " | " + seq.object_count + " objects | " + seq.intrinsic_count + " intrinsics"}
                {seq.significance != null && (" | significance=" + seq.significance.toFixed(4))}
            </Text>

            {/* Objects table -- columns: #, Glyph, Pos, Color, Matched, Resolves (no Shape) */}
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
                            <Table.Th>#</Table.Th>
                            <Table.Th>Glyph</Table.Th>
                            <Table.Th>Pos</Table.Th>
                            <Table.Th>Color</Table.Th>
                            <Table.Th>Shape</Table.Th>
                            <Table.Th>Matched</Table.Th>
                            <Table.Th>Resolves</Table.Th>
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {seq.objects.map((obj) => {
                            const nodeId = Number(obj.id);
                            const hasValidId = Number.isFinite(nodeId) && nodeId !== 0;
                            let matchedLabel = "--";
                            if (obj.matched_previous === true) matchedLabel = "Yes";
                            else if (obj.matched_previous === false) matchedLabel = "No";
                            return (
                                <Table.Tr key={obj.id}>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {obj.cycle_number ?? "--"}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        {obj.glyph && hasValidId ? (
                                            <ObjectLink
                                                objectId={nodeId}
                                                glyph={obj.glyph}
                                            />
                                        ) : (
                                            <Text size="xs" ff="monospace" fw={700}>
                                                {obj.glyph ?? "--"}
                                            </Text>
                                        )}
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
                                            {obj.color == null ? "--" : String(obj.color)}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {obj.shape == null ? "--" : String(obj.shape)}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {matchedLabel}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {obj.resolve_count ?? "--"}
                                        </Text>
                                    </Table.Td>
                                </Table.Tr>
                            );
                        })}
                    </Table.Tbody>
                </Table>
            )}

            {/* Intrinsics table -- columns: Name, Raw, Normalized, Matched */}
            {intrinsicKeys.length > 0 && (
                <Table
                    striped
                    highlightOnHover
                    withTableBorder
                    withColumnBorders
                    fz="xs"
                >
                    <Table.Thead>
                        <Table.Tr>
                            <Table.Th>Name</Table.Th>
                            <Table.Th>Raw</Table.Th>
                            <Table.Th>Normalized</Table.Th>
                            <Table.Th>Matched</Table.Th>
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {intrinsicKeys.map((k) => {
                            const norm = seq.intrinsics[k] ?? 0;
                            const raw = rawIntrinsics?.[k];
                            const matched = isIntrinsicMatched(k, transformChanges);
                            return (
                                <Table.Tr key={k}>
                                    <Table.Td>
                                        <Text size="xs" c="dimmed">{k}</Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {raw == null ? "--" : String(raw)}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {norm.toFixed(4)}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {matched ? "Yes" : "No"}
                                        </Text>
                                    </Table.Td>
                                </Table.Tr>
                            );
                        })}
                    </Table.Tbody>
                </Table>
            )}
        </Stack>
    );
}
