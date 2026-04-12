/** Transition panel -- three-column prev / current / delta view of frame changes. */

import { Badge, Group, Stack, Table, Text, UnstyledButton } from "@mantine/core";

import type {
    ObjectTransformChangeData,
    ObjectTransformData,
    SequenceObjectData,
    StepData,
    TransformChangeData,
} from "../../types/step-data";
import { ObjectLink } from "../common/ObjectLink";

interface TransitionPanelProps {
    data: StepData | undefined;
    onStepClick?: (step: number) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function deltaColor(value: number | undefined | null): string | undefined {
    if (value == null) return undefined;
    return value > 0 ? "green" : "red";
}

function fmtDelta(value: number | undefined | null): string {
    if (value == null) return "--";
    const prefix = value > 0 ? "+" : "";
    return prefix + (Number.isInteger(value) ? String(value) : value.toFixed(4));
}

/** Normalize a change entry -- old format is a plain string, new format is an object. */
function normalizeChange(ch: TransformChangeData | string): TransformChangeData {
    if (typeof ch === "string") return { description: ch };
    return ch;
}

// ---------------------------------------------------------------------------
// Row types for the three-column table
// ---------------------------------------------------------------------------

interface ObjectRow {
    /** Object UUID -- string because ROC UUIDs are 63-bit ints. */
    uuid: string;
    glyph?: string;
    color?: string;
    /** Memgraph internal node id (small int, safe as number). */
    nodeId?: number;
    prevLabel: string;
    currentLabel: string;
    deltaLabel: string;
    status: "matched" | "new" | "gone";
}

interface IntrinsicRow {
    name: string;
    prev: string;
    current: string;
    delta: string;
    deltaNum: number | null;
}

// ---------------------------------------------------------------------------
// Build rows from available data
// ---------------------------------------------------------------------------

function positionLabel(x?: number | null, y?: number | null): string {
    if (x == null && y == null) return "--";
    return `(${x ?? "?"}, ${y ?? "?"})`;
}

function extractPosition(changes: ObjectTransformChangeData[]): { oldX?: number; oldY?: number; newX?: number; newY?: number } {
    const result: Record<string, number | undefined> = {};
    for (const ch of changes) {
        if (ch.property === "x" || ch.property === "y") {
            result[`old${ch.property.toUpperCase()}`] = ch.old_value == null ? undefined : Number(ch.old_value);
            result[`new${ch.property.toUpperCase()}`] = ch.new_value == null ? undefined : Number(ch.new_value);
        }
    }
    return { oldX: result.oldX, oldY: result.oldY, newX: result.newX, newY: result.newY };
}

function fmtChangeVal(v: unknown): string {
    if (v == null) return "?";
    if (typeof v === "number") return String(v);
    if (typeof v === "string") return v;
    return JSON.stringify(v);
}

function summarizeDelta(changes: ObjectTransformChangeData[]): string {
    if (changes.length === 0) return "(no change)";
    return changes.map((ch) => {
        if (ch.type === "discrete") {
            return `${ch.property}: ${fmtChangeVal(ch.old_value)} -> ${fmtChangeVal(ch.new_value)}`;
        }
        return `${ch.property}: ${fmtDelta(ch.delta)}`;
    }).join(", ");
}

function buildObjectRows(
    transforms: ObjectTransformData[],
    currentObjects: SequenceObjectData[],
): ObjectRow[] {
    const rows: ObjectRow[] = [];
    const transformUuids = new Set(transforms.map((t) => t.uuid));

    // Matched / gone objects from transforms
    for (const t of transforms) {
        const pos = extractPosition(t.changes);

        if (t.status === "gone") {
            rows.push({
                uuid: t.uuid,
                glyph: t.glyph,
                color: t.color,
                nodeId: t.node_id,
                prevLabel: positionLabel(pos.oldX, pos.oldY),
                currentLabel: "--",
                deltaLabel: "(gone)",
                status: "gone",
            });
            continue;
        }

        if (t.status === "new") {
            rows.push({
                uuid: t.uuid,
                glyph: t.glyph,
                color: t.color,
                nodeId: t.node_id,
                prevLabel: "--",
                currentLabel: positionLabel(pos.newX, pos.newY),
                deltaLabel: "(new)",
                status: "new",
            });
            continue;
        }

        // Default: matched (has changes in both frames)
        rows.push({
            uuid: t.uuid,
            glyph: t.glyph,
            color: t.color,
            nodeId: t.node_id,
            prevLabel: positionLabel(pos.oldX, pos.oldY),
            currentLabel: positionLabel(pos.newX, pos.newY),
            deltaLabel: summarizeDelta(t.changes),
            status: "matched",
        });
    }

    // New objects from sequence_summary that aren't already in transforms.
    // obj.id from sequence_summary is the Memgraph node id (small int as
    // string); transform.uuid is the 63-bit Object UUID (also a string
    // now). They aren't directly comparable, so we match by node_id.
    for (const obj of currentObjects) {
        if (obj.matched_previous === false) {
            // Skip if already covered by a transform with the same node id.
            const objNodeId = Number(obj.id);
            if (Number.isFinite(objNodeId) && transformUuids.has(String(objNodeId))) {
                continue;
            }

            rows.push({
                // No UUID available from sequence_summary, fall back to node id.
                uuid: obj.id,
                glyph: obj.glyph,
                nodeId: Number.isFinite(objNodeId) ? objNodeId : undefined,
                prevLabel: "--",
                currentLabel: positionLabel(obj.x, obj.y),
                deltaLabel: "(new)",
                status: "new",
            });
        }
    }

    return rows;
}

function buildIntrinsicRows(
    changes: TransformChangeData[],
    intrinsics: Record<string, number> | undefined,
): IntrinsicRow[] {
    return changes.map((ch) => {
        const norm = normalizeChange(ch);
        const delta = norm.normalized_change;
        const name = norm.name ?? norm.description;
        // Use current intrinsic values from sequence_summary when available
        const currentVal = intrinsics?.[name];
        const prevVal = currentVal != null && delta != null ? currentVal - delta : undefined;
        return {
            name,
            prev: prevVal == null ? "--" : prevVal.toFixed(4),
            current: currentVal == null ? "--" : currentVal.toFixed(4),
            delta: delta == null ? "--" : fmtDelta(delta),
            deltaNum: delta ?? null,
        };
    });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<string, string | undefined> = { new: "blue", gone: "red" };

export function TransitionPanel({ data, onStepClick }: Readonly<TransitionPanelProps>) {
    const t = data?.transform_summary;
    const seq = data?.sequence_summary;

    if (!t) {
        return (
            <Text size="xs" c="dimmed">
                No transform data
            </Text>
        );
    }

    const objectTransforms = t.object_transforms ?? [];
    const currentObjects = seq?.objects ?? [];
    const objectRows = buildObjectRows(objectTransforms, currentObjects);
    const intrinsicRows = buildIntrinsicRows(t.changes.map(normalizeChange), seq?.intrinsics);

    const hasContent = objectRows.length > 0 || intrinsicRows.length > 0;

    if (!hasContent) {
        return (
            <Text size="xs" c="dimmed">
                No changes this step
            </Text>
        );
    }

    const prevTick = seq ? seq.tick - 1 : null;
    const currentTick = seq?.tick ?? null;

    return (
        <Stack gap="sm">
            {/* Object Changes */}
            {objectRows.length > 0 && (
                <>
                    <Group gap="xs">
                        <Text size="sm" fw={600}>Object Changes</Text>
                        <Badge variant="filled" color="violet" size="sm">
                            {objectRows.length}
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
                                <Table.Th>Object</Table.Th>
                                <Table.Th>
                                    Previous
                                    {prevTick != null && onStepClick && (
                                        <>
                                            {" ("}
                                            <UnstyledButton
                                                component="span"
                                                style={{ fontSize: 10, textDecoration: "underline dotted", fontFamily: "monospace" }}
                                                onClick={() => onStepClick(prevTick)}
                                            >
                                                tick={prevTick}
                                            </UnstyledButton>
                                            {")"}
                                        </>
                                    )}
                                </Table.Th>
                                <Table.Th>
                                    Current
                                    {currentTick != null && onStepClick && (
                                        <>
                                            {" ("}
                                            <UnstyledButton
                                                component="span"
                                                style={{ fontSize: 10, textDecoration: "underline dotted", fontFamily: "monospace" }}
                                                onClick={() => onStepClick(currentTick)}
                                            >
                                                tick={currentTick}
                                            </UnstyledButton>
                                            {")"}
                                        </>
                                    )}
                                </Table.Th>
                                <Table.Th>Delta</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {objectRows.map((row) => (
                                <Table.Tr key={row.uuid}>
                                    <Table.Td>
                                        {row.glyph && row.nodeId ? (
                                            <ObjectLink
                                                objectId={row.nodeId}
                                                glyph={row.glyph}
                                                color={row.color}
                                            />
                                        ) : (
                                            <Text size="xs" ff="monospace" fw={700}>
                                                {row.glyph ?? String(row.uuid)}
                                            </Text>
                                        )}
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {row.prevLabel}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">
                                            {row.currentLabel}
                                        </Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text
                                            size="xs"
                                            ff="monospace"
                                            c={STATUS_COLORS[row.status]}
                                        >
                                            {row.deltaLabel}
                                        </Text>
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </>
            )}

            {/* Intrinsic Changes */}
            {intrinsicRows.length > 0 && (
                <>
                    <Group gap="xs">
                        <Text size="sm" fw={600}>Intrinsic Changes</Text>
                        <Badge variant="filled" color="orange" size="sm">
                            {intrinsicRows.length}
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
                                <Table.Th>Intrinsic</Table.Th>
                                <Table.Th>Previous</Table.Th>
                                <Table.Th>Current</Table.Th>
                                <Table.Th>Delta</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {intrinsicRows.map((row) => (
                                <Table.Tr key={row.name}>
                                    <Table.Td>
                                        <Text size="xs" c="dimmed">{row.name}</Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">{row.prev}</Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text size="xs" ff="monospace">{row.current}</Text>
                                    </Table.Td>
                                    <Table.Td>
                                        <Text
                                            size="xs"
                                            ff="monospace"
                                            c={deltaColor(row.deltaNum)}
                                        >
                                            {row.delta}
                                        </Text>
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </>
            )}
        </Stack>
    );
}
