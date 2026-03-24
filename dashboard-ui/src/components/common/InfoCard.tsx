/** Reusable bordered card with a title -- for grouping panel content. */

import { Paper, Text } from "@mantine/core";
import type { ReactNode } from "react";

interface InfoCardProps {
    title: string;
    children: ReactNode;
}

export function InfoCard({ title, children }: Readonly<InfoCardProps>) {
    return (
        <Paper p="xs" withBorder>
            <Text size="xs" fw={600} mb={4}>
                {title}
            </Text>
            {children}
        </Paper>
    );
}
