/** Attention spread -- single-line metric showing focus coverage. */

import { Text, Tooltip } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface AttentionSpreadProps {
    data: StepData | undefined;
}

export function AttentionSpread({ data }: Readonly<AttentionSpreadProps>) {
    const att = data?.attenuation;
    const attended = att?.spread_attended as number | undefined;
    const total = att?.spread_total as number | undefined;
    const pct = att?.spread_pct as number | undefined;

    if (attended == null || total == null || pct == null) {
        return null;
    }

    return (
        <Text size="xs" fw={500}>
            Spread:{" "}
            <Tooltip label="Unique glyphs attended across all steps" withArrow>
                <Text
                    component="span"
                    size="xs"
                    fw={600}
                    ff="monospace"
                    td="underline dotted"
                    style={{ cursor: "help" }}
                >
                    {attended}
                </Text>
            </Tooltip>
            /
            <Tooltip label="Unique glyphs visible on screen across all steps" withArrow>
                <Text
                    component="span"
                    size="xs"
                    fw={600}
                    ff="monospace"
                    td="underline dotted"
                    style={{ cursor: "help" }}
                >
                    {total}
                </Text>
            </Tooltip>
            {" ("}
            <Tooltip label="Attention coverage percentage" withArrow>
                <Text
                    component="span"
                    size="xs"
                    fw={600}
                    ff="monospace"
                    td="underline dotted"
                    style={{ cursor: "help" }}
                >
                    {pct}%
                </Text>
            </Tooltip>
            {")"}
        </Text>
    );
}
