/** Action panel -- current action taken. */

import { Badge, Group, Text } from "@mantine/core";

import { InfoCard } from "../common/InfoCard";
import type { StepData } from "../../types/step-data";

interface ActionPanelProps {
    data: StepData | undefined;
}

export function ActionPanel({ data }: ActionPanelProps) {
    const a = data?.action_taken;

    return (
        <InfoCard title="Action">
            {a ? (
                <Group gap="xs">
                    <Badge color="indigo" variant="filled" size="sm">
                        {a.action_name
                            ? a.action_key
                                ? `${a.action_name} (${a.action_key})`
                                : a.action_name
                            : `Action #${a.action_id}`}
                    </Badge>
                    <Text size="xs" c="dimmed">
                        (id: {a.action_id})
                    </Text>
                </Group>
            ) : (
                <Text size="xs" c="dimmed">
                    No action data
                </Text>
            )}
        </InfoCard>
    );
}
