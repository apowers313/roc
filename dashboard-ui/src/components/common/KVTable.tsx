/** Reusable compact key-value table using Mantine Table. */

import { Table, Text } from "@mantine/core";

import { InfoCard } from "./InfoCard";

interface KVTableProps {
    data: Record<string, unknown> | null | undefined;
    emptyText?: string;
    title?: string;
}

/** Formats a value for display, truncating long strings. */
function formatValue(value: unknown): string {
    if (value === null || value === undefined) return "--";
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }
    const s = String(value);
    return s.length > 80 ? s.slice(0, 77) + "..." : s;
}

export function KVTable({ data, emptyText = "No data", title }: KVTableProps) {
    if (!data || Object.keys(data).length === 0) {
        return (
            <Text size="xs" c="dimmed">
                {emptyText}
            </Text>
        );
    }

    const table = (
        <Table
            horizontalSpacing={0}
            verticalSpacing={0}
            withRowBorders={false}
            layout="fixed"
            style={{ maxWidth: 150 }}
        >
            <Table.Tbody>
                {Object.entries(data).map(([key, value]) => (
                    <Table.Tr key={key}>
                        <Table.Td
                            style={{
                                color: "var(--mantine-color-dimmed)",
                                width: "65%",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                                padding: 0,
                                fontSize: "var(--mantine-font-size-xs)",
                            }}
                        >
                            {key}
                        </Table.Td>
                        <Table.Td
                            style={{
                                fontWeight: 500,
                                textAlign: "right",
                                fontVariantNumeric: "tabular-nums",
                                whiteSpace: "nowrap",
                                padding: 0,
                                fontSize: "var(--mantine-font-size-xs)",
                            }}
                        >
                            {formatValue(value)}
                        </Table.Td>
                    </Table.Tr>
                ))}
            </Table.Tbody>
        </Table>
    );

    if (title) {
        return (
            <InfoCard title={title}>
                {table}
            </InfoCard>
        );
    }

    return table;
}
