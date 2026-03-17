/** Saliency map panel -- heatmap character grid with legend. */

import { Group, Text } from "@mantine/core";

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

    return (
        <div>
            <CharGrid data={data.saliency} />
            <Group gap={4} mt={2} style={{ fontSize: "10px" }}>
                <Text size="xs" c="dimmed">
                    Low
                </Text>
                <div
                    style={{
                        height: "8px",
                        flex: 1,
                        background:
                            "linear-gradient(to right, #0000bb, #00bbbb, #00bb00, #bbbb00, #bb0000)",
                        borderRadius: "2px",
                    }}
                />
                <Text size="xs" c="dimmed">
                    High
                </Text>
            </Group>
        </div>
    );
}
