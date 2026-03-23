/** Toolbar strip with popout panel buttons. Hidden when empty. */

import { ActionIcon, Group, Tooltip } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { PopoutPanel } from "./PopoutPanel";

interface PopoutButtonProps {
    title: string;
    icon: LucideIcon;
    children: ReactNode;
    size?: string;
}

function PopoutButton({ title, icon: Icon, children, size }: PopoutButtonProps) {
    const [opened, { open, close }] = useDisclosure(false);

    return (
        <>
            <Tooltip label={title} position="bottom" withArrow>
                <ActionIcon
                    onClick={open}
                    variant={opened ? "filled" : "subtle"}
                    size="sm"
                    title={`Open ${title} panel`}
                >
                    <Icon size={14} />
                </ActionIcon>
            </Tooltip>
            <PopoutPanel title={title} opened={opened} onClose={close} size={size}>
                {children}
            </PopoutPanel>
        </>
    );
}

interface PopoutToolbarProps {
    children: ReactNode;
}

/** Container for PopoutButton elements. Renders nothing if children is empty. */
export function PopoutToolbar({ children }: PopoutToolbarProps) {
    return (
        <Group
            gap={4}
            mb={4}
            p={4}
            style={{
                backgroundColor: "var(--mantine-color-dark-6)",
                borderRadius: 4,
            }}
        >
            {children}
        </Group>
    );
}

PopoutToolbar.Button = PopoutButton;
