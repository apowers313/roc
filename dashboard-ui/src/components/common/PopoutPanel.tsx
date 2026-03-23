/** Popout panel -- a right-side Drawer controlled by external open/close state. */

import { Drawer } from "@mantine/core";
import type { ReactNode } from "react";

interface PopoutPanelProps {
    title: string;
    opened: boolean;
    onClose: () => void;
    children: ReactNode;
    size?: string;
}

export function PopoutPanel({ title, opened, onClose, children, size = "xl" }: PopoutPanelProps) {
    return (
        <Drawer
            opened={opened}
            onClose={onClose}
            title={title}
            position="right"
            size={size}
            withOverlay={false}
            closeOnClickOutside={false}
            lockScroll={false}
            removeScrollProps={{ enabled: false }}
        >
            {children}
        </Drawer>
    );
}
