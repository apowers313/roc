/** Feature extraction panel -- shows feature counts per type. */

import { Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";
import { KVTable } from "../common/KVTable";

interface FeatureTableProps {
    data: StepData | undefined;
}

/** All known feature types in display order. Missing types show "--". */
const FEATURE_KEYS = [
    "Flood",
    "Line",
    "Single",
    "Distance",
    "Color",
    "Shape",
    "Delta",
    "Motion",
];

/**
 * Parse feature data which may be a dict with counts or a raw string
 * like "\t\tFlood: 4\n\t\tLine: 114\n...".
 */
function parseFeatures(
    entry: Record<string, unknown>,
): Record<string, unknown> {
    // If the entry has a "raw" key with a string value, parse it
    if (typeof entry.raw === "string") {
        const parsed: Record<string, unknown> = {};
        for (const line of entry.raw.split("\n")) {
            const match = /^(\w+):\s*(.+)$/.exec(line.trim());
            if (match?.[1] && match[2]) {
                const val = Number(match[2]);
                parsed[match[1]] = Number.isNaN(val) ? match[2] : val;
            }
        }
        return parsed;
    }
    return entry;
}

export function FeatureTable({ data }: Readonly<FeatureTableProps>) {
    if (!data?.features || data.features.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No feature data
            </Text>
        );
    }

    // Parse raw string if needed, then normalize to fixed set of rows
    const raw = parseFeatures(data.features[0] ?? {});
    const stable: Record<string, unknown> = {};
    for (const key of FEATURE_KEYS) {
        stable[key] = raw[key] ?? null;
    }

    return <KVTable data={stable} title="Feature Counts" />;
}
