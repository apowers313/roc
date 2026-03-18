/** Object info panel -- structured display of detected objects. */

import { Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface ObjectInfoProps {
    data: StepData | undefined;
}

/** Extract display columns from a list of object records. */
function getColumns(objects: Record<string, unknown>[]): string[] {
    const keys = new Set<string>();
    for (const obj of objects) {
        for (const k of Object.keys(obj)) {
            if (k !== "raw") keys.add(k);
        }
    }
    // Prefer a consistent ordering: id-like fields first, then alphabetical
    const priority = ["id", "name", "type", "label", "class", "glyph", "char"];
    const sorted = [...keys].sort((a, b) => {
        const ai = priority.indexOf(a);
        const bi = priority.indexOf(b);
        if (ai !== -1 && bi !== -1) return ai - bi;
        if (ai !== -1) return -1;
        if (bi !== -1) return 1;
        return a.localeCompare(b);
    });
    return sorted;
}

function formatValue(v: unknown): string {
    if (v == null) return "--";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
    if (typeof v === "boolean") return v ? "Y" : "N";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
}

export function ObjectInfo({ data }: ObjectInfoProps) {
    const objects = data?.object_info;

    if (!objects || objects.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No object data
            </Text>
        );
    }

    // If all objects only have a "raw" field, fall back to simple display
    const hasStructured = objects.some(
        (obj) => Object.keys(obj).some((k) => k !== "raw"),
    );

    if (!hasStructured) {
        return (
            <div style={{ maxHeight: 200, overflowY: "auto" }}>
                <Table
                    horizontalSpacing={4}
                    verticalSpacing={1}
                    withRowBorders={false}
                    layout="fixed"
                >
                    <Table.Tbody>
                        {objects.map((obj, i) => (
                            <Table.Tr key={i}>
                                <Table.Td>
                                    <Text size="xs" style={{ fontFamily: "monospace" }}>
                                        {String(obj.raw ?? JSON.stringify(obj))}
                                    </Text>
                                </Table.Td>
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            </div>
        );
    }

    const columns = getColumns(objects);

    return (
        <div style={{ maxHeight: 200, overflowY: "auto" }}>
            <Table
                horizontalSpacing={4}
                verticalSpacing={1}
                withRowBorders
                layout="fixed"
                striped
            >
                <Table.Thead>
                    <Table.Tr>
                        {columns.map((col) => (
                            <Table.Th
                                key={col}
                                style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}
                            >
                                {col}
                            </Table.Th>
                        ))}
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {objects.map((obj, i) => (
                        <Table.Tr key={i}>
                            {columns.map((col) => (
                                <Table.Td
                                    key={col}
                                    style={{
                                        fontSize: 10,
                                        fontFamily: "monospace",
                                        padding: "1px 4px",
                                    }}
                                >
                                    {formatValue(obj[col])}
                                </Table.Td>
                            ))}
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </div>
    );
}
