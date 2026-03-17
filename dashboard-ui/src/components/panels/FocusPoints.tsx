/** Focus points panel. */

import { Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";
import { KVTable } from "../common/KVTable";

interface FocusPointsProps {
    data: StepData | undefined;
}

export function FocusPoints({ data }: FocusPointsProps) {
    if (!data?.focus_points || data.focus_points.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No focus data
            </Text>
        );
    }

    return <KVTable data={data.focus_points[0]} emptyText="No focus data" />;
}
