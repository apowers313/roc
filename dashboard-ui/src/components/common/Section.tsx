/** Reusable accordion section with icon, colored title, and error boundary. */

import { Accordion, Group, Text } from "@mantine/core";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { ErrorBoundary } from "./ErrorBoundary";

interface SectionProps {
    value: string;
    title: string;
    icon: LucideIcon;
    color: string;
    toolbar?: ReactNode;
    children: ReactNode;
}

export function Section({ value, title, icon: Icon, color, toolbar, children }: SectionProps) {
    return (
        <Accordion.Item value={value}>
            <Accordion.Control
                style={{
                    backgroundColor: `var(--mantine-color-${color}-9)`,
                    borderRadius: 4,
                }}
            >
                <Group gap={8} wrap="nowrap">
                    <Icon size={16} color="white" />
                    <Text size="sm" fw={500} c="white">
                        {title}
                    </Text>
                </Group>
            </Accordion.Control>
            <Accordion.Panel>
                {toolbar && (
                    <Group
                        gap={4}
                        mb={8}
                        p={4}
                        style={{
                            backgroundColor: "var(--mantine-color-dark-6)",
                            borderRadius: 4,
                        }}
                    >
                        {toolbar}
                    </Group>
                )}
                <ErrorBoundary>
                    {children}
                </ErrorBoundary>
            </Accordion.Panel>
        </Accordion.Item>
    );
}
