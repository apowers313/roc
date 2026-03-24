/** Reusable accordion section with icon, colored title, and error boundary. */

import { Accordion, Badge, Group, Text } from "@mantine/core";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { useRatchetHeight } from "../../hooks/useRatchetHeight";
import { ErrorBoundary } from "./ErrorBoundary";

interface SectionProps {
    value: string;
    title: string;
    icon: LucideIcon;
    color: string;
    toolbar?: ReactNode;
    /** ExpMod name(s) used by components in this section. */
    expmod?: string | string[];
    children: ReactNode;
}

export function Section({ value, title, icon: Icon, color, toolbar, expmod, children }: SectionProps) {
    const { contentRef, minHeight } = useRatchetHeight();

    // Normalize to array, filter out empty/undefined values
    const expmodList = expmod
        ? (Array.isArray(expmod) ? expmod : [expmod]).filter(Boolean)
        : [];

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
                <div ref={contentRef} style={{ minHeight }}>
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
                    {expmodList.length > 0 && (
                        <Group gap={4} mb={8}>
                            <Text size="xs" c="dimmed" fw={500}>
                                ExpMod:
                            </Text>
                            {expmodList.map((name) => (
                                <Badge key={name} size="xs" variant="light" color="grape">
                                    {name}
                                </Badge>
                            ))}
                        </Group>
                    )}
                    <ErrorBoundary>
                        {children}
                    </ErrorBoundary>
                </div>
            </Accordion.Panel>
        </Accordion.Item>
    );
}
