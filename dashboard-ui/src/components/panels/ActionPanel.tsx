/** Action panel -- current action taken and expmod. */

import { Badge, Group, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface ActionPanelProps {
    data: StepData | undefined;
}

export function ActionPanel({ data }: ActionPanelProps) {
    const a = data?.action_taken;

    return (
        <Group gap="xs">
            <Text size="sm" fw={600}>
                Action
            </Text>
            {a ? (
                <>
                    <Badge color="indigo" variant="filled" size="sm">
                        {a.action_name ?? `Action #${a.action_id}`}
                    </Badge>
                    <Text size="xs" c="dimmed">
                        (id: {a.action_id})
                    </Text>
                    {a.expmod_name && (
                        <Badge size="xs" variant="light" color="grape">
                            {a.expmod_name}
                        </Badge>
                    )}
                </>
            ) : (
                <Text size="xs" c="dimmed">
                    No action data
                </Text>
            )}
        </Group>
    );
}
