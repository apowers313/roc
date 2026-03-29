/** Saliency map panel -- heatmap character grid with multi-cycle support. */

import { Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";
import { CharGrid } from "../common/CharGrid";

interface SaliencyMapProps {
    data: StepData | undefined;
    cycleIndex?: number;
}

export function SaliencyMap({ data, cycleIndex }: Readonly<SaliencyMapProps>) {
    // Use cycle-specific saliency if available, otherwise fall back to top-level
    const saliency =
        cycleIndex != null && data?.saliency_cycles?.[cycleIndex]
            ? data.saliency_cycles[cycleIndex].saliency
            : data?.saliency;

    if (!saliency) {
        return (
            <Text size="xs" c="dimmed">
                No saliency data
            </Text>
        );
    }

    return <CharGrid data={saliency} />;
}
