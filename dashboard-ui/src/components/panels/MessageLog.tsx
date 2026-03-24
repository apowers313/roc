/** Message log -- displays the current step's auditory message. */

import { Paper, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface MessageLogProps {
    data: StepData | undefined;
}

export function MessageLog({ data }: Readonly<MessageLogProps>) {
    const msg = data?.message;

    if (!msg) {
        return (
            <Text size="xs" c="dimmed">
                No message this step
            </Text>
        );
    }

    return (
        <Paper p="xs" withBorder>
            <Text size="sm" ff="monospace">
                {msg}
            </Text>
        </Paper>
    );
}
