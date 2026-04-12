/** Resolution Inspector -- structured display of object resolution decisions. */

import { Badge, Group, SegmentedControl, Stack, Table, Text, UnstyledButton } from "@mantine/core";
import { memo, useState, type ReactNode } from "react";

import { InfoCard } from "../common/InfoCard";
import { ObjectLink } from "../common/ObjectLink";
import { useHighlight } from "../../state/highlight";
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
    { key: "num_candidates", label: "Candidates" },
    { key: "matched_object_id", label: "Matched" },
    { key: "vocab_size", label: "Vocab Size" },
    { key: "total_objects_tracked", label: "Objects Tracked" },
];

/** Map NetHack color names to approximate CSS colors. */
const NH_COLOR_MAP: Record<string, string> = {
    RED: "#f44", GREEN: "#4f4", BROWN: "#a80", BLUE: "#44f",
    MAGENTA: "#f4f", CYAN: "#4ff", GREY: "#aaa", ORANGE: "#fa0",
    "BRIGHT GREEN": "#0f0", YELLOW: "#ff0", "BRIGHT BLUE": "#88f",
    "BRIGHT MAGENTA": "#f8f", "BRIGHT CYAN": "#8ff", WHITE: "#fff",
    BLACK: "#444", "NO COLOR": "#888",
};

function formatValue(v: unknown): string {
    if (v == null) return "--";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    if (typeof v === "string") return v;
    return JSON.stringify(v);
}

/** Inline glyph character with its color. */
function GlyphBadge({ char, color }: Readonly<{ char?: string; color?: string }>) {
    if (!char) return null;
    const fg = color ? NH_COLOR_MAP[color] ?? "#fff" : "#fff";
    return (
        <Text
            component="span"
            ff="monospace"
            fw={700}
            style={{ color: fg, background: "#000", padding: "0 3px", borderRadius: 2, fontSize: 13, lineHeight: 1.2 }}
        >
            {char}
        </Text>
    );
}

/** Parse non-relational attrs from feature strings like "ShapeNode(.), ColorNode(GREY), SingleNode(2371)". */
function parseAttrsFromFeatures(features: unknown): Record<string, string> {
    if (!Array.isArray(features)) return {};
    const attrs: Record<string, string> = {};
    for (const f of features) {
        const s = String(f);
        const shapeMatch = /^ShapeNode\((.)\)$/.exec(s);
        if (shapeMatch) attrs.shape = shapeMatch[1]!;
        const colorMatch = /^ColorNode\((.+)\)$/.exec(s);
        if (colorMatch) attrs.color = colorMatch[1]!;
        const singleMatch = /^SingleNode\((\d+)\)$/.exec(s);
        if (singleMatch) attrs.glyph = singleMatch[1]!;
    }
    return attrs;
}

function comparisonBackground(same: boolean, different: boolean): string | undefined {
    if (different) return "rgba(248, 81, 73, 0.15)";
    if (same) return "rgba(63, 185, 80, 0.15)";
    return undefined;
}

function renderMatchedValue(
    label: string,
    matchVal: string | undefined,
    matchedAttrs: Record<string, string> | undefined,
): ReactNode {
    if (label === "shape" && matchVal) {
        return <GlyphBadge char={String(matchVal)} color={matchedAttrs?.color} />;
    }
    if (matchVal != null) return String(matchVal);
    return matchedAttrs ? "--" : "?";
}

interface CandidateDetail {
    id: string;
    value: number;
    valueLabel: string;
    char?: string;
    color?: string;
    glyph?: number;
}

function buildCandidateDetails(d: Record<string, unknown>): CandidateDetail[] {
    // Prefer candidate_details (new format with glyph/color)
    const details = d.candidate_details;
    if (Array.isArray(details) && details.length > 0) {
        return details.map((entry: unknown) => {
            const e = entry as Record<string, unknown>;
            const dist = e.distance as number | undefined;
            const prob = e.probability as number | undefined;
            return {
                id: String(e.id),
                value: dist ?? prob ?? 0,
                valueLabel: dist == null ? "probability" : "distance",
                char: e.char as string | undefined,
                color: e.color as string | undefined,
                glyph: e.glyph as number | undefined,
            };
        });
    }

    // Fall back to old format
    const dists = d.candidate_distances;
    if (Array.isArray(dists) && dists.length > 0) {
        return dists.map((entry: unknown) => {
            const [objId, dist] = entry as [unknown, number];
            return { id: String(objId), value: dist, valueLabel: "distance" };
        });
    }
    const posts = d.posteriors;
    if (Array.isArray(posts) && posts.length > 0) {
        return posts.map((entry: unknown) => {
            const [objId, prob] = entry as [unknown, number];
            return { id: String(objId), value: prob, valueLabel: "probability" };
        });
    }
    return [];
}

interface SummaryRow {
    label: string;
    value: string;
    clickable?: boolean;
}

function buildSummaryRows(d: Record<string, unknown>): SummaryRow[] {
    const rows: SummaryRow[] = [];
    const hasLocation = d.x != null && d.y != null;
    const location = hasLocation ? `(${String(d.x)}, ${String(d.y)})` : null;
    if (location) {
        rows.push({ label: "Location", value: location, clickable: true });
    }
    if (d.tick != null) {
        rows.push({ label: "Tick", value: formatValue(d.tick) });
    }
    for (const { key, label } of SUMMARY_KEYS) {
        if (d[key] != null) {
            rows.push({ label, value: formatValue(d[key]) });
        }
    }
    return rows;
}

interface ComparisonSectionProps {
    observedAttrs: Record<string, string>;
    matchedAttrs: Record<string, string> | undefined;
}

const COMPARISON_ROW_DEFS = [
    { label: "shape", obsKey: "shape", matchKey: "char" },
    { label: "color", obsKey: "color", matchKey: "color" },
    { label: "glyph", obsKey: "glyph", matchKey: "glyph" },
] as const;

function ComparisonSection({ observedAttrs, matchedAttrs }: Readonly<ComparisonSectionProps>) {
    return (
        <Table
            horizontalSpacing={4}
            verticalSpacing={1}
            withRowBorders
            mb={4}
            fz="xs"
            style={{ width: "auto" }}
        >
            <Table.Thead>
                <Table.Tr>
                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: "30%" }}>
                        attr
                    </Table.Th>
                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: "35%" }}>
                        observed
                    </Table.Th>
                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: "35%" }}>
                        matched
                    </Table.Th>
                </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
                {COMPARISON_ROW_DEFS.map(({ label, obsKey, matchKey }) => {
                    const obsVal = observedAttrs[obsKey];
                    const matchVal = matchedAttrs?.[matchKey];
                    if (!obsVal && !matchVal) return null;
                    const same = obsVal != null && matchVal != null && String(obsVal) === String(matchVal);
                    const different = obsVal != null && matchVal != null && String(obsVal) !== String(matchVal);
                    return (
                        <Table.Tr key={label}>
                            <Table.Td style={{ fontSize: 10, color: "var(--mantine-color-dimmed)", padding: "1px 4px" }}>
                                {label}
                            </Table.Td>
                            <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                {label === "shape" && obsVal ? (
                                    <GlyphBadge char={obsVal} color={observedAttrs.color} />
                                ) : (obsVal ?? "--")}
                            </Table.Td>
                            <Table.Td style={{
                                fontSize: 10,
                                fontFamily: "monospace",
                                padding: "1px 4px",
                                background: comparisonBackground(same, different),
                            }}>
                                {renderMatchedValue(label, matchVal, matchedAttrs)}
                            </Table.Td>
                        </Table.Tr>
                    );
                })}
            </Table.Tbody>
        </Table>
    );
}

interface CandidatesSectionProps {
    candidates: CandidateDetail[];
}

function CandidatesSection({ candidates }: Readonly<CandidatesSectionProps>) {
    const valueHeader = candidates[0]!.valueLabel;
    const hasGlyphs = candidates.some((c) => c.char);
    return (
        <InfoCard title="Candidates">
            <div style={{ maxHeight: 120, overflowY: "auto" }}>
                <Table
                    horizontalSpacing={4}
                    verticalSpacing={1}
                    withRowBorders
                    striped
                    style={{ width: "auto" }}
                >
                    <Table.Thead>
                        <Table.Tr>
                            {hasGlyphs && (
                                <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: 30 }}>
                                </Table.Th>
                            )}
                            <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                node id
                            </Table.Th>
                            <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                {valueHeader}
                            </Table.Th>
                            {hasGlyphs && (
                                <>
                                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                        glyph
                                    </Table.Th>
                                    <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                                        color
                                    </Table.Th>
                                </>
                            )}
                        </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        {candidates.map((c) => (
                            <Table.Tr key={c.id}>
                                {hasGlyphs && (
                                    <Table.Td style={{ padding: "1px 4px", textAlign: "center" }}>
                                        <GlyphBadge char={c.char} color={c.color} />
                                    </Table.Td>
                                )}
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {c.id}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {formatValue(c.value)}
                                </Table.Td>
                                {hasGlyphs && (
                                    <>
                                        <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                            {c.glyph ?? ""}
                                        </Table.Td>
                                        <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                            {c.color ?? ""}
                                        </Table.Td>
                                    </>
                                )}
                            </Table.Tr>
                        ))}
                    </Table.Tbody>
                </Table>
            </div>
        </InfoCard>
    );
}

function CycleSummaryTable({ cycles }: Readonly<{ cycles: Record<string, unknown>[] }>) {
    return (
        <InfoCard title="Resolution Summary">
            <Table striped withTableBorder withColumnBorders fz="xs">
                <Table.Thead>
                    <Table.Tr>
                        <Table.Th>#</Table.Th>
                        <Table.Th>Outcome</Table.Th>
                        <Table.Th>Object</Table.Th>
                        <Table.Th>Location</Table.Th>
                        <Table.Th>Candidates</Table.Th>
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {cycles.map((c, idx) => {
                        const outcome = typeof c.outcome === "string" ? c.outcome : "unknown";
                        const color = OUTCOME_COLORS[outcome] ?? "gray";
                        const attrs = parseAttrsFromFeatures(c.features);
                        const matchedAttrs = c.matched_attrs as Record<string, string> | undefined;
                        // Prefer matched glyph for MATCHED outcomes, fall back to observed shape
                        const glyphChar = matchedAttrs?.char ?? attrs.shape;
                        const glyphColor = matchedAttrs?.color ?? attrs.color;
                        const loc =
                            c.x != null && c.y != null ? `(${Number(c.x)}, ${Number(c.y)})` : "--";
                        const numCandidates = c.num_candidates == null ? "--" : Number(c.num_candidates);
                        return (
                            <Table.Tr key={`cycle-${String(c.x)}-${String(c.y)}-${idx}`}>
                                <Table.Td>{idx + 1}</Table.Td>
                                <Table.Td>
                                    <Badge size="xs" color={color} variant="filled">
                                        {OUTCOME_LABELS[outcome] ?? outcome}
                                    </Badge>
                                </Table.Td>
                                <Table.Td>
                                    <GlyphBadge char={glyphChar} color={glyphColor} />
                                </Table.Td>
                                <Table.Td>
                                    <Text size="xs" ff="monospace">{loc}</Text>
                                </Table.Td>
                                <Table.Td>
                                    <Text size="xs" ff="monospace">{numCandidates}</Text>
                                </Table.Td>
                            </Table.Tr>
                        );
                    })}
                </Table.Tbody>
            </Table>
        </InfoCard>
    );
}

function ResolutionInspectorInner({ data }: Readonly<ResolutionInspectorProps>) {
    const { togglePoint, points } = useHighlight();
    const [selectedCycle, setSelectedCycle] = useState(0);
    const cycles = data?.resolution_cycles;
    const hasCycles = cycles != null && cycles.length > 0;

    // Use selected cycle data if available, otherwise fall back to resolution_metrics
    const d = hasCycles
        ? (cycles[selectedCycle] as Record<string, unknown> | undefined) ?? null
        : data?.resolution_metrics;

    if (!d) {
        return (
            <Text size="xs" c="dimmed">
                No resolution data
            </Text>
        );
    }

    const outcome = typeof d.outcome === "string" ? d.outcome : "unknown";
    const outcomeLabel = OUTCOME_LABELS[outcome] ?? outcome.toUpperCase();
    const outcomeColor = OUTCOME_COLORS[outcome] ?? "gray";
    const isMatch = outcome === "match";

    // Observed non-relational attributes (from current features)
    const observedAttrs = parseAttrsFromFeatures(d.features);

    // Matched object's stored attributes (from backend matched_attrs)
    const matchedAttrs = d.matched_attrs as Record<string, string> | undefined;

    const hasLocation = d.x != null && d.y != null;
    const locX = hasLocation ? Number(d.x) : 0;
    const locY = hasLocation ? Number(d.y) : 0;
    const locHlColor = hasLocation ? points.find((p) => p.x === locX && p.y === locY)?.color : undefined;

    const summaryRows = buildSummaryRows(d);
    const candidates = buildCandidateDetails(d);
    const featureList = Array.isArray(d.features) ? d.features : null;
    const hasComparison = isMatch && (matchedAttrs != null || Object.keys(observedAttrs).length > 0);

    return (
        <Stack gap="xs">
            {hasCycles && cycles.length > 1 && (
                <>
                    <CycleSummaryTable cycles={cycles as Record<string, unknown>[]} />
                    <SegmentedControl
                        size="xs"
                        data={cycles.map((_, i) => ({
                            label: `Cycle ${i + 1}`,
                            value: String(i),
                        }))}
                        value={String(selectedCycle)}
                        onChange={(v) => setSelectedCycle(Number(v))}
                    />
                </>
            )}
            <InfoCard title="Resolution">
                <Group gap="xs" mb={4}>
                    <Badge size="xs" color={outcomeColor} variant="filled">
                        {outcomeLabel}
                    </Badge>
                    {observedAttrs.shape && (
                        <GlyphBadge char={observedAttrs.shape} color={observedAttrs.color} />
                    )}
                    {isMatch && matchedAttrs?.char && (
                        <>
                            <Text size="xs" c="dimmed">{"\u2192"}</Text>
                            {d.matched_object_id == null ? (
                                <GlyphBadge char={matchedAttrs.char} color={matchedAttrs.color} />
                            ) : (
                                <ObjectLink
                                    objectId={Number(d.matched_object_id)}
                                    glyph={matchedAttrs.char}
                                    color={matchedAttrs.color}
                                />
                            )}
                        </>
                    )}
                    {isMatch && !matchedAttrs?.char && observedAttrs.shape && (
                        <Text size="xs" c="dimmed">{"\u2192 ?"}</Text>
                    )}
                </Group>

                {/* Side-by-side attribute comparison for matches */}
                {hasComparison && (
                    <ComparisonSection observedAttrs={observedAttrs} matchedAttrs={matchedAttrs} />
                )}

                {summaryRows.length > 0 && (
                    <Table
                        horizontalSpacing={4}
                        verticalSpacing={1}
                        withRowBorders={false}
                        style={{ width: "auto" }}
                    >
                        <Table.Tbody>
                            {summaryRows.map((row) => (
                                <Table.Tr
                                    key={row.label}
                                    style={{
                                        cursor: row.clickable ? "pointer" : undefined,
                                        background: row.clickable && locHlColor ? `${locHlColor}26` : undefined,
                                    }}
                                    onClick={row.clickable && hasLocation ? () => togglePoint({ x: locX, y: locY, label: `resolved @ (${locX},${locY})` }) : undefined}
                                >
                                    <Table.Td
                                        style={{ fontSize: 10, color: "var(--mantine-color-dimmed)", paddingRight: 16 }}
                                    >
                                        {row.label}
                                    </Table.Td>
                                    <Table.Td style={{ fontSize: 10, fontFamily: "monospace" }}>
                                        {row.clickable ? (
                                            <UnstyledButton style={{ fontSize: 10, fontFamily: "monospace", textDecoration: "underline dotted" }}>
                                                {row.value}
                                            </UnstyledButton>
                                        ) : row.value}
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                )}
            </InfoCard>

            {candidates.length > 0 && (
                <CandidatesSection candidates={candidates} />
            )}

            {featureList && featureList.length > 0 && (
                <InfoCard title="Features">
                    <Text size="xs" c="dimmed" style={{ fontFamily: "monospace", wordBreak: "break-all" }}>
                        {featureList.slice(0, 20).map(String).join(", ")}
                        {featureList.length > 20 && ` ... (+${featureList.length - 20})`}
                    </Text>
                </InfoCard>
            )}
        </Stack>
    );
}

// ResolutionInspector is 500+ lines with nested Tables and InfoCards
// that rebuild on every step change. Wrap it in ``React.memo`` so the
// reconciliation stops at this boundary when the parent re-renders
// without a new ``data`` reference (same step, sibling state change,
// etc.). TanStack Query returns the same object reference for cached
// hits, so this meaningfully cuts work during rapid step navigation.
export const ResolutionInspector = memo(ResolutionInspectorInner);
