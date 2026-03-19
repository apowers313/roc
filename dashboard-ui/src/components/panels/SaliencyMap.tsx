/** Saliency map panel -- heatmap character grid. */

import { Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";
import { CharGrid } from "../common/CharGrid";

interface SaliencyMapProps {
    data: StepData | undefined;
}

export function SaliencyMap({ data }: SaliencyMapProps) {
    if (!data?.saliency) {
        return (
            <Text size="xs" c="dimmed">
                No saliency data
            </Text>
        );
    }

    return <CharGrid data={data.saliency} />;
}
