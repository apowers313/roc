/** Attenuation deep dive -- structured display of attenuation data. */

import { Grid, Group, Stack, Table, Text } from "@mantine/core";

import { InfoCard } from "../common/InfoCard";
import { useHighlight } from "../../state/highlight";
import type { StepData } from "../../types/step-data";

interface AttenuationPanelProps {
    data: StepData | undefined;
}

function renderValue(v: unknown): string {
    if (v == null) return "--";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v === "string") return v;
    // 2-element arrays (points) -- handle both number and string-number values
    if (Array.isArray(v) && v.length === 2 && !Number.isNaN(Number(v[0]))) {
        return `(${v[0]}, ${v[1]})`;
    }
    return JSON.stringify(v);
}

function KVSection({
    title,
    entries,
    onPointClick,
    highlightMap,
}: Readonly<{
    title: string;
    entries: [string, unknown][];
    onPointClick?: (x: number, y: number, label: string) => void;
    highlightMap?: Map<string, string>;
}>) {
    if (entries.length === 0) return null;
    return (
        <InfoCard title={title}>
            {entries.map(([k, v]) => {
                // Point values arrive as [x, y] (col, row).
                // Values may be numbers or numeric strings depending on JSON serialization.
                const isPoint = Array.isArray(v) && v.length === 2 &&
                    !Number.isNaN(Number(v[0])) && !Number.isNaN(Number(v[1])) &&
                    v[0] !== null && v[1] !== null && v[0] !== "" && v[1] !== "";
                const ptX = isPoint ? Number(v[0]) : 0;
                const ptY = isPoint ? Number(v[1]) : 0;
                const hlColor = isPoint ? highlightMap?.get(`${ptX},${ptY}`) : undefined;
                return (
                    <Group
                        key={k}
                        justify="space-between"
                        gap={4}
                        onClick={isPoint && onPointClick ? () => onPointClick(ptX, ptY, k) : undefined}
                        style={{
                            cursor: isPoint && onPointClick ? "pointer" : undefined,
                            background: hlColor ? `${hlColor}26` : undefined,
                            borderRadius: 2,
                        }}
                    >
                        <Text size="xs" c="dimmed">
                            {k}
                        </Text>
                        <Text size="xs" ff="monospace" td={isPoint && onPointClick ? "underline dotted" : undefined}>
                            {renderValue(v)}
                        </Text>
                    </Group>
                );
            })}
        </InfoCard>
    );
}

interface HistoryEntry {
    x: number;
    y: number;
    tick: number;
}

export function AttenuationPanel({ data }: Readonly<AttenuationPanelProps>) {
    const { togglePoint, points } = useHighlight();
    const highlightMap = new Map(points.map((p) => [`${p.x},${p.y}`, p.color]));
    const att = data?.attenuation;

    if (!att) {
        return (
            <Text size="xs" c="dimmed">
                No attenuation data
            </Text>
        );
    }

    // Categorize fields -- match actual Python emission keys
    const peakKeys = [
        "peak_count", "top_peak_strength", "top_peak_shifted",
        "pre_peak", "post_peak",
    ];
    // Active inference keys
    const entropyKeys = ["entropy_at_focus", "entropy_max", "entropy_min"];
    const beliefKeys = ["beliefs_tracked", "vocab_size", "omega"];
    // Linear decline keys
    const linearKeys = ["history_size"];
    const flavorKey = "flavor";
    const historyKey = "history";
    const focusKey = "focus_points";
    const eventKey = "event";

    // Derive the winning focus point from focus_points[0]
    const fpList = att[focusKey] as Array<{ x: number; y: number }> | undefined;
    const firstFp = fpList && fpList.length > 0 ? fpList[0] : undefined;
    const focusPoint = firstFp ? [firstFp.x, firstFp.y] : null;

    const peakEntries: [string, unknown][] = peakKeys
        .filter((k) => k in att)
        .map((k) => [k, att[k]] as [string, unknown]);
    if (focusPoint) {
        peakEntries.push(["focus_point", focusPoint]);
    }
    const entropyEntries = entropyKeys
        .filter((k) => k in att)
        .map((k) => [k, att[k]] as [string, unknown]);
    const beliefEntries = [...beliefKeys, ...linearKeys]
        .filter((k) => k in att)
        .map((k) => [k, att[k]] as [string, unknown]);

    const spreadKeys = ["spread_attended", "spread_total", "spread_pct"];
    const knownKeys = new Set([
        ...peakKeys, ...entropyKeys, ...beliefKeys, ...linearKeys,
        ...spreadKeys,
        flavorKey, historyKey, focusKey, eventKey,
    ]);
    const otherEntries = Object.entries(att)
        .filter(([k]) => !knownKeys.has(k));

    const history = att[historyKey] as HistoryEntry[] | undefined;

    return (
        <Stack gap="xs">
            <Grid gutter="xs">
                <Grid.Col span={4}>
                    <KVSection
                        title="Peaks"
                        entries={peakEntries}
                        onPointClick={(x, y, label) => togglePoint({ x, y, label: `peak: ${label}` })}
                        highlightMap={highlightMap}
                    />
                </Grid.Col>
                <Grid.Col span={4}>
                    <KVSection title="Entropy" entries={entropyEntries} />
                </Grid.Col>
                <Grid.Col span={4}>
                    <KVSection title="Beliefs / History" entries={beliefEntries} />
                </Grid.Col>
            </Grid>
            {history && history.length > 0 && (
                <InfoCard title="Attended Locations (attenuated)">
                    <Table striped fz="xs" withTableBorder>
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th>X</Table.Th>
                                <Table.Th>Y</Table.Th>
                                <Table.Th>Tick</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {history.slice(-10).map((h) => {
                                const hlX = h.x;
                                const hlY = h.y;
                                const hlC = highlightMap.get(`${hlX},${hlY}`);
                                return (
                                <Table.Tr
                                    key={`${h.x}-${h.y}-${h.tick}`}
                                    onClick={() => togglePoint({ x: hlX, y: hlY, label: `attn tick ${h.tick}` })}
                                    style={{
                                        cursor: "pointer",
                                        background: hlC ? `${hlC}26` : undefined,
                                    }}
                                >
                                    <Table.Td>{h.x}</Table.Td>
                                    <Table.Td>{h.y}</Table.Td>
                                    <Table.Td>{h.tick}</Table.Td>
                                </Table.Tr>
                                );
                            })}
                        </Table.Tbody>
                    </Table>
                </InfoCard>
            )}
            {otherEntries.length > 0 && (
                <KVSection title="Other" entries={otherEntries} />
            )}
        </Stack>
    );
}
