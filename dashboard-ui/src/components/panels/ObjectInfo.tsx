/** Object info panel -- structured display of detected objects. */

import { Table, Text } from "@mantine/core";

import { InfoCard } from "../common/InfoCard";
import type { StepData } from "../../types/step-data";

interface ObjectInfoProps {
    data: StepData | undefined;
}

/** Derive a stable React key from an object record. */
function objectRowKey(obj: Record<string, unknown>): string {
    if (typeof obj.id === "string" || typeof obj.id === "number") return String(obj.id);
    if (typeof obj.name === "string" || typeof obj.name === "number") return String(obj.name);
    return JSON.stringify(obj);
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
    if (typeof v === "string") return v;
    return JSON.stringify(v);
}

export function ObjectInfo({ data }: Readonly<ObjectInfoProps>) {
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
            <InfoCard title="Objects">
                <div style={{ maxHeight: 200, overflowY: "auto" }}>
                    <Table
                        horizontalSpacing={4}
                        verticalSpacing={1}
                        withRowBorders={false}
                        layout="fixed"
                    >
                        <Table.Tbody>
                            {objects.map((obj) => (
                                <Table.Tr key={typeof obj.raw === "string" ? obj.raw : JSON.stringify(obj)}>
                                    <Table.Td>
                                        <Text size="xs" style={{ fontFamily: "monospace" }}>
                                            {typeof obj.raw === "string" ? obj.raw : JSON.stringify(obj.raw ?? obj)}
                                        </Text>
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </div>
            </InfoCard>
        );
    }

    const columns = getColumns(objects);

    return (
        <InfoCard title="Objects">
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
                        {objects.map((obj) => (
                            <Table.Tr key={objectRowKey(obj)}>
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
        </InfoCard>
    );
}
