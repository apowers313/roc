/** Object detail modal -- full history of a resolved object. */

import { Badge, Group, Modal, Pagination, Stack, Table, Text, UnstyledButton } from "@mantine/core";
import { useState } from "react";

import type { ObjectHistoryState } from "../../api/client";
import { useObjectHistory } from "../../api/queries";
import { useDashboard } from "../../state/context";

interface ObjectModalProps {
    objectId: number;
    opened: boolean;
    onClose: () => void;
    glyph?: string;
    color?: string;
}

/** Check whether a field changed between consecutive observations. */
function changed(
    curr: ObjectHistoryState,
    prev: ObjectHistoryState | undefined,
    field: keyof ObjectHistoryState,
): boolean {
    if (!prev) return false;
    return curr[field] !== prev[field];
}

function cellBg(isChanged: boolean): string | undefined {
    return isChanged ? "rgba(248, 81, 73, 0.15)" : undefined;
}

function fmtVal(v: unknown): string {
    if (v == null) return "--";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
    if (typeof v === "string") return v;
    if (typeof v === "boolean") return String(v);
    return JSON.stringify(v);
}

function deltaColor(delta: number | null | undefined): string | undefined {
    if (delta == null) return undefined;
    if (delta > 0) return "green";
    if (delta < 0) return "red";
    return undefined;
}

function fmtDelta(delta: number | null | undefined): string {
    if (delta == null) return "--";
    return delta > 0 ? `+${delta}` : String(delta);
}

/** Number of observations to show per page in the stepper. */
const OBS_PAGE_SIZE = 10;

export function ObjectModal({ objectId, opened, onClose, glyph, color }: Readonly<ObjectModalProps>) {
    const { run, setStep } = useDashboard();
    const { data } = useObjectHistory(run, opened ? objectId : null);
    const [obsPage, setObsPage] = useState(1);

    const info = data?.info;
    const states = data?.states ?? [];
    const transforms = data?.transforms ?? [];
    const firstSeen = states.length > 0 ? states[0]!.tick : null;

    // Observation paging
    const totalObsPages = Math.ceil(states.length / OBS_PAGE_SIZE);
    const pagedStates = states.slice((obsPage - 1) * OBS_PAGE_SIZE, obsPage * OBS_PAGE_SIZE);
    // Offset into full states array for diff highlighting
    const pageOffset = (obsPage - 1) * OBS_PAGE_SIZE;

    const handleTickClick = (tick: number) => {
        setStep(tick);
        onClose();
    };

    const stateFields: { key: keyof ObjectHistoryState; label: string }[] = [
        { key: "x", label: "x" },
        { key: "y", label: "y" },
        { key: "glyph_type", label: "glyph" },
        { key: "color_type", label: "color" },
        { key: "shape_type", label: "shape" },
        { key: "distance", label: "distance" },
    ];

    return (
        <Modal
            opened={opened}
            onClose={onClose}
            title={`Object: ${glyph ?? objectId}`}
            size="lg"
            transitionProps={{ duration: 0 }}
        >
            <Stack gap="sm">
                {/* Header */}
                <Group gap="xs">
                    {glyph && (
                        <Text
                            component="span"
                            ff="monospace"
                            fw={700}
                            style={{
                                color: color ?? "#fff",
                                background: "#000",
                                padding: "0 3px",
                                borderRadius: 2,
                                fontSize: 16,
                            }}
                        >
                            {glyph}
                        </Text>
                    )}
                    <Text size="sm" c="dimmed">UUID: {info?.uuid ?? objectId}</Text>
                    <Text size="sm" c="dimmed">Node: {objectId}</Text>
                    <Badge variant="light" color="blue" size="sm">
                        Matches: {info?.resolve_count ?? 0}
                    </Badge>
                    {firstSeen != null && (
                        <Text size="sm" c="dimmed">
                            First seen:{" "}
                            <UnstyledButton
                                component="span"
                                style={{ fontSize: "inherit", fontFamily: "inherit", textDecoration: "underline dotted" }}
                                onClick={() => handleTickClick(firstSeen)}
                            >
                                step {firstSeen}
                            </UnstyledButton>
                        </Text>
                    )}
                </Group>

                {/* State History */}
                {states.length > 0 && (
                    <>
                        <Group gap="xs" justify="space-between">
                            <Text size="sm" fw={600}>Observations ({states.length})</Text>
                            {totalObsPages > 1 && (
                                <Pagination
                                    size="xs"
                                    total={totalObsPages}
                                    value={obsPage}
                                    onChange={setObsPage}
                                />
                            )}
                        </Group>
                        <div style={{ maxHeight: 300, overflowY: "auto" }}>
                            <Table striped withTableBorder withColumnBorders fz="xs">
                                <Table.Thead>
                                    <Table.Tr>
                                        <Table.Th>tick</Table.Th>
                                        {stateFields.map((f) => (
                                            <Table.Th key={f.key}>{f.label}</Table.Th>
                                        ))}
                                    </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                    {pagedStates.map((s, i) => {
                                        const absIdx = pageOffset + i;
                                        const prev = absIdx > 0 ? states[absIdx - 1] : undefined;
                                        return (
                                            <Table.Tr key={s.tick}>
                                                <Table.Td>
                                                    <UnstyledButton
                                                        style={{
                                                            fontSize: 10,
                                                            fontFamily: "monospace",
                                                            textDecoration: "underline dotted",
                                                        }}
                                                        onClick={() => handleTickClick(s.tick)}
                                                    >
                                                        {s.tick}
                                                    </UnstyledButton>
                                                </Table.Td>
                                                {stateFields.map((f) => (
                                                    <Table.Td
                                                        key={f.key}
                                                        style={{
                                                            background: cellBg(changed(s, prev, f.key)),
                                                            fontSize: 10,
                                                            fontFamily: "monospace",
                                                        }}
                                                    >
                                                        {fmtVal(s[f.key])}
                                                    </Table.Td>
                                                ))}
                                            </Table.Tr>
                                        );
                                    })}
                                </Table.Tbody>
                            </Table>
                        </div>
                    </>
                )}

                {/* Transform History */}
                {transforms.length > 0 && (
                    <>
                        <Text size="sm" fw={600}>Transforms ({transforms.length})</Text>
                        <div style={{ maxHeight: 200, overflowY: "auto" }}>
                            <Table striped withTableBorder withColumnBorders fz="xs">
                                <Table.Thead>
                                    <Table.Tr>
                                        <Table.Th>#</Table.Th>
                                        <Table.Th>ticks</Table.Th>
                                        <Table.Th>property</Table.Th>
                                        <Table.Th>old</Table.Th>
                                        <Table.Th>new</Table.Th>
                                        <Table.Th>delta</Table.Th>
                                    </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                    {transforms.map((t, ti) => {
                                        const fromTick = states[ti]?.tick;
                                        const toTick = states[ti + 1]?.tick;
                                        const tickLabel =
                                            fromTick != null && toTick != null
                                                ? `${fromTick} -> ${toTick}`
                                                : `#${ti + 1}`;
                                        return t.changes.map((ch, ci) => (
                                            <Table.Tr key={`${ti}-${ci}`}>
                                                {ci === 0 && (
                                                    <Table.Td rowSpan={t.changes.length}>
                                                        <Text size="xs" ff="monospace">
                                                            {ti + 1}
                                                        </Text>
                                                    </Table.Td>
                                                )}
                                                {ci === 0 && (
                                                    <Table.Td rowSpan={t.changes.length}>
                                                        <Text size="xs" ff="monospace">
                                                            {tickLabel}
                                                        </Text>
                                                    </Table.Td>
                                                )}
                                                <Table.Td>
                                                    <Text size="xs" ff="monospace">
                                                        {ch.property}
                                                    </Text>
                                                </Table.Td>
                                                <Table.Td>
                                                    <Text size="xs" ff="monospace">
                                                        {fmtVal(ch.old_value)}
                                                    </Text>
                                                </Table.Td>
                                                <Table.Td>
                                                    <Text size="xs" ff="monospace">
                                                        {fmtVal(ch.new_value)}
                                                    </Text>
                                                </Table.Td>
                                                <Table.Td>
                                                    <Text
                                                        size="xs"
                                                        ff="monospace"
                                                        c={deltaColor(ch.delta)}
                                                    >
                                                        {fmtDelta(ch.delta)}
                                                    </Text>
                                                </Table.Td>
                                            </Table.Tr>
                                        ));
                                    })}
                                </Table.Tbody>
                            </Table>
                        </div>
                    </>
                )}

                {states.length === 0 && transforms.length === 0 && (
                    <Text size="xs" c="dimmed">No history data</Text>
                )}
            </Stack>
        </Modal>
    );
}
