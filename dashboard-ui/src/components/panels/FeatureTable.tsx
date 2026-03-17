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

export function FeatureTable({ data }: FeatureTableProps) {
    if (!data?.features || data.features.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No feature data
            </Text>
        );
    }

    // Normalize to a fixed set of rows so the table height never changes.
    const raw = data.features[0] ?? {};
    const stable: Record<string, unknown> = {};
    for (const key of FEATURE_KEYS) {
        stable[key] = raw[key] ?? null;
    }

    return <KVTable data={stable} title="Feature Counts" />;
}
