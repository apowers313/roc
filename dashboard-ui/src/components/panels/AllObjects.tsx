/** All Objects panel -- sortable table of every resolved object. */

import { Table, Text } from "@mantine/core";
import { useCallback, useState } from "react";

import type { ResolvedObject } from "../../api/client";
import { useAllObjects } from "../../api/queries";

/** Map NetHack color names to CSS colors. */
const NH_COLOR_MAP: Record<string, string> = {
    RED: "#f44", GREEN: "#4f4", BROWN: "#a80", BLUE: "#44f",
    MAGENTA: "#f4f", CYAN: "#4ff", GREY: "#aaa", ORANGE: "#fa0",
    "BRIGHT GREEN": "#0f0", YELLOW: "#ff0", "BRIGHT BLUE": "#88f",
    "BRIGHT MAGENTA": "#f8f", "BRIGHT CYAN": "#8ff", WHITE: "#fff",
    BLACK: "#444", "NO COLOR": "#888",
};

interface AllObjectsProps {
    run: string;
    game?: number;
    onStepClick?: (step: number) => void;
}

type SortKey = "shape" | "glyph" | "color" | "type" | "node_id" | "step_added" | "match_count";

function compareValues(a: unknown, b: unknown): number {
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a).localeCompare(String(b));
}

export function AllObjects({ run, game, onStepClick }: AllObjectsProps) {
    const { data: objects } = useAllObjects(run, game);
    const [sortKey, setSortKey] = useState<SortKey>("shape");
    const [sortAsc, setSortAsc] = useState(true);

    const handleSort = useCallback((key: SortKey) => {
        if (key === sortKey) {
            setSortAsc((prev) => !prev);
        } else {
            setSortKey(key);
            setSortAsc(true);
        }
    }, [sortKey]);

    if (!objects || objects.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No objects
            </Text>
        );
    }

    const sorted = [...objects].sort((a, b) => {
        const cmp = compareValues(a[sortKey], b[sortKey]);
        return sortAsc ? cmp : -cmp;
    });

    const columns: { key: SortKey; label: string; width?: string | number }[] = [
        { key: "shape", label: "shape", width: 40 },
        { key: "glyph", label: "glyph", width: 60 },
        { key: "color", label: "color" },
        { key: "type", label: "type", width: 50 },
        { key: "node_id", label: "node id" },
        { key: "step_added", label: "step added", width: 80 },
        { key: "match_count", label: "matches", width: 70 },
    ];

    const arrow = (key: SortKey) =>
        sortKey === key ? (sortAsc ? " \u25B2" : " \u25BC") : "";

    return (
        <div style={{ maxHeight: "calc(100vh - 120px)", overflowY: "auto" }}>
            <Table
                striped
                highlightOnHover
                withTableBorder
                withColumnBorders
                fz="xs"
                layout="fixed"
            >
                <Table.Thead>
                    <Table.Tr>
                        <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: 30 }}>
                            #
                        </Table.Th>
                        {columns.map((col) => (
                            <Table.Th
                                key={col.key}
                                onClick={() => handleSort(col.key)}
                                style={{
                                    fontSize: 10,
                                    fontWeight: 600,
                                    padding: "2px 4px",
                                    cursor: "pointer",
                                    userSelect: "none",
                                    width: col.width,
                                }}
                            >
                                {col.label}{arrow(col.key)}
                            </Table.Th>
                        ))}
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {sorted.map((obj: ResolvedObject, i: number) => {
                        const fg = obj.color ? NH_COLOR_MAP[obj.color] ?? "#fff" : "#fff";
                        return (
                            <Table.Tr
                                key={obj.node_id ?? `new-${obj.step_added}`}
                                onClick={obj.step_added != null && onStepClick ? () => onStepClick(obj.step_added!) : undefined}
                                style={{
                                    cursor: obj.step_added != null && onStepClick ? "pointer" : undefined,
                                }}
                            >
                                <Table.Td style={{ fontSize: 10, padding: "1px 4px", color: "var(--mantine-color-dimmed)" }}>
                                    {i + 1}
                                </Table.Td>
                                <Table.Td style={{ padding: "1px 4px", textAlign: "center" }}>
                                    {obj.shape ? (
                                        <Text
                                            component="span"
                                            ff="monospace"
                                            fw={700}
                                            style={{ color: fg, background: "#000", padding: "0 3px", borderRadius: 2, fontSize: 13 }}
                                        >
                                            {obj.shape}
                                        </Text>
                                    ) : "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.glyph ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.color ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.type ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.node_id ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.step_added ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.match_count}
                                </Table.Td>
                            </Table.Tr>
                        );
                    })}
                </Table.Tbody>
            </Table>
        </div>
    );
}
