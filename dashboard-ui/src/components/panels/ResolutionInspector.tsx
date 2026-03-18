/** Resolution Inspector -- structured display of object resolution decisions. */

import { Badge, Group, Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface ResolutionInspectorProps {
    data: StepData | undefined;
}

const OUTCOME_COLORS: Record<string, string> = {
    match: "green",
    new_object: "blue",
    low_confidence: "yellow",
};

const OUTCOME_LABELS: Record<string, string> = {
    match: "MATCHED",
    new_object: "NEW OBJECT",
    low_confidence: "LOW CONFIDENCE",
};

const SUMMARY_KEYS: readonly { key: string; label: string }[] = [
    { key: "algorithm", label: "Algorithm" },
    { key: "num_candidates", label: "Candidates" },
    { key: "matched_object_id", label: "Matched" },
    { key: "vocab_size", label: "Vocab Size" },
    { key: "total_objects_tracked", label: "Objects Tracked" },
];

function formatValue(v: unknown): string {
    if (v == null) return "--";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return String(v);
}

interface CandidateRow {
    object: string;
    value: number;
    label: string;
}

function buildCandidates(d: Record<string, unknown>): CandidateRow[] {
    const dists = d.candidate_distances;
    if (Array.isArray(dists) && dists.length > 0) {
        return dists.map((entry: unknown) => {
            const [objId, dist] = entry as [unknown, number];
            return { object: String(objId), value: dist, label: "distance" };
        });
    }
    const posts = d.posteriors;
    if (Array.isArray(posts) && posts.length > 0) {
        return posts.map((entry: unknown) => {
            const [objId, prob] = entry as [unknown, number];
            return { object: String(objId), value: prob, label: "probability" };
        });
    }
    return [];
}

export function ResolutionInspector({ data }: ResolutionInspectorProps) {
    const d = data?.resolution_metrics;

    if (!d) {
        return (
            <Text size="xs" c="dimmed">
                No resolution data
            </Text>
        );
    }

    const outcome = String(d.outcome ?? "unknown");
    const outcomeLabel = OUTCOME_LABELS[outcome] ?? outcome.toUpperCase();
    const outcomeColor = OUTCOME_COLORS[outcome] ?? "gray";

    // Location
    const hasLocation = d.x != null && d.y != null;
    const location = hasLocation ? `(${d.x}, ${d.y})` : null;

    // Summary rows
    const summaryRows: { label: string; value: string }[] = [];
    if (location) {
        summaryRows.push({ label: "Location", value: location });
    }
    if (d.tick != null) {
        summaryRows.push({ label: "Tick", value: String(d.tick) });
    }
    for (const { key, label } of SUMMARY_KEYS) {
        if (d[key] != null) {
            summaryRows.push({ label, value: formatValue(d[key]) });
        }
    }

    // Candidates
    const candidates = buildCandidates(d);
    const valueHeader = candidates.length > 0 ? candidates[0]!.label : "value";

    // Features
    const features = d.features;
    const featureList = Array.isArray(features) ? features : null;

    return (
        <div>
            <Group gap="xs" mb={4}>
                <Text size="xs" fw={600}>Resolution</Text>
                <Badge size="xs" color={outcomeColor} variant="filled">
                    {outcomeLabel}
                </Badge>
            </Group>

            {summaryRows.length > 0 && (
                <Table
                    horizontalSpacing={4}
                    verticalSpacing={1}
                    withRowBorders={false}
                    layout="fixed"
                    mb={4}
                >
                    <Table.Tbody>
                        {summaryRows.map((row) => (
                            <Table.Tr key={row.label}>
                                <Table.Td
                                    style={{ fontSize: 10, color: "var(--mantine-color-dimmed)", width: "40%" }}
                                >
                                    {row.label}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace" }}>
                                    {row.value}
                                </Table.Td>
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            )}

            {candidates.length > 0 && (
                <>
                    <Text size="xs" fw={600} mt={4} mb={2}>
                        Candidates
                    </Text>
                    <div style={{ maxHeight: 120, overflowY: "auto" }}>
                        <Table
                            horizontalSpacing={4}
                            verticalSpacing={1}
                            withRowBorders
                            layout="fixed"
                            striped
                        >
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                        object
                                    </Table.Th>
                                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                        {valueHeader}
                                    </Table.Th>
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                {candidates.map((c, i) => (
                                    <Table.Tr key={i}>
                                        <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                            {c.object}
                                        </Table.Td>
                                        <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                            {formatValue(c.value)}
                                        </Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>
                    </div>
                </>
            )}

            {featureList && featureList.length > 0 && (
                <>
                    <Text size="xs" fw={600} mt={4} mb={2}>
                        Features
                    </Text>
                    <Text size="xs" c="dimmed" style={{ fontFamily: "monospace", wordBreak: "break-all" }}>
                        {featureList.slice(0, 20).map(String).join(", ")}
                        {featureList.length > 20 && ` ... (+${featureList.length - 20})`}
                    </Text>
                </>
            )}
        </div>
    );
}
