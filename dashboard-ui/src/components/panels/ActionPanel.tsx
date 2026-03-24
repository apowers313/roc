/** Action panel -- current action taken. */

import { Badge, Group, Text } from "@mantine/core";

import { InfoCard } from "../common/InfoCard";
import type { StepData } from "../../types/step-data";

interface ActionPanelProps {
    data: StepData | undefined;
}

function actionLabel(a: NonNullable<StepData["action_taken"]>): string {
    if (!a.action_name) return `Action #${a.action_id}`;
    if (a.action_key) return `${a.action_name} (${a.action_key})`;
    return a.action_name;
}

export function ActionPanel({ data }: Readonly<ActionPanelProps>) {
    const a = data?.action_taken;

    return (
        <InfoCard title="Action">
            {a ? (
                <Group gap="xs">
                    <Badge color="indigo" variant="filled" size="sm">
                        {actionLabel(a)}
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
