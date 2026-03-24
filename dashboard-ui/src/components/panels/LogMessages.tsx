/** Log messages panel -- filterable log table with severity coloring. */

import { Select, Table, Text } from "@mantine/core";
import { useState } from "react";

import type { StepData } from "../../types/step-data";

interface LogMessagesProps {
    data: StepData | undefined;
}

const SEVERITY_ORDER: Record<string, number> = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    WARNING: 2,
    ERROR: 3,
};

const SEVERITY_COLORS: Record<string, string> = {
    DEBUG: "var(--mantine-color-dimmed)",
    INFO: "var(--mantine-color-text)",
    WARN: "#d29922",
    WARNING: "#d29922",
    ERROR: "#f85149",
};

const LEVEL_OPTIONS = ["DEBUG", "INFO", "WARN", "ERROR"];

export function LogMessages({ data }: Readonly<LogMessagesProps>) {
    const [minLevel, setMinLevel] = useState("DEBUG");

    const logs = data?.logs ?? [];
    const minOrder = SEVERITY_ORDER[minLevel] ?? 0;
    const filtered = logs.filter((log) => {
        const level = (log.severity_text ?? "INFO").toUpperCase();
        return (SEVERITY_ORDER[level] ?? 1) >= minOrder;
    });

    return (
        <div>
            <Select
                size="xs"
                value={minLevel}
                onChange={(v) => setMinLevel(v ?? "DEBUG")}
                data={LEVEL_OPTIONS}
                style={{ width: 100, marginBottom: 4 }}
            />
            {filtered.length === 0 ? (
                <Text size="xs" c="dimmed">
                    No log messages
                </Text>
            ) : (
                <div style={{ maxHeight: 200, overflowY: "auto" }}>
                    <Table
                        horizontalSpacing={4}
                        verticalSpacing={1}
                        withRowBorders={false}
                    >
                        <Table.Tbody>
                            {filtered.map((log, i) => {
                                const level = (
                                    log.severity_text ?? "INFO"
                                ).toUpperCase();
                                const color =
                                    SEVERITY_COLORS[level] ??
                                    "var(--mantine-color-text)";
                                const logKey = `${level}-${log.timestamp ?? i}-${String(log.body ?? "").slice(0, 40)}`;
                                return (
                                    <Table.Tr key={logKey}>
                                        <Table.Td
                                            style={{
                                                color,
                                                width: 50,
                                                fontWeight: 500,
                                            }}
                                        >
                                            {level}
                                        </Table.Td>
                                        <Table.Td style={{ color }}>
                                            {String(log.body ?? "")}
                                        </Table.Td>
                                    </Table.Tr>
                                );
                            })}
                        </Table.Tbody>
                    </Table>
                </div>
            )}
        </div>
    );
}
