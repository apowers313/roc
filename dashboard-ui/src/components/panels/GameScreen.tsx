/** Game screen panel -- displays the 21x79 colored character grid. */

import { Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";
import { CharGrid } from "../common/CharGrid";

interface GameScreenProps {
    data: StepData | undefined;
}

export function GameScreen({ data }: Readonly<GameScreenProps>) {
    const hasScreen = data?.screen != null;
    const lastRow = hasScreen
        ? String.fromCodePoint(...(data!.screen!.chars[23] ?? []).filter((c) => c >= 32 && c < 127))
        : "";
    // eslint-disable-next-line no-console
    console.log("[GameScreen] render", JSON.stringify({ hasScreen, lastRow: lastRow.trim().slice(-20) }));

    // Fixed-height container prevents layout shifts when data loads/changes.
    // NetHack screen is 24 rows at 9px * 1.15 line-height + padding.
    return (
        <div style={{ minHeight: 260 }}>
            {data?.screen ? (
                <CharGrid data={data.screen} highlightRowOffset={1} />
            ) : (
                <Text size="xs" c="dimmed">
                    No screen data
                </Text>
            )}
        </div>
    );
}
