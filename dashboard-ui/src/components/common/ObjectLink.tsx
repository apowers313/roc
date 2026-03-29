/** Shared clickable object reference -- glyph badge that opens ObjectModal. */

import { Text } from "@mantine/core";
import { useState } from "react";

import { ObjectModal } from "../panels/ObjectModal";

/** Map NetHack color names to CSS colors. */
const NH_COLOR_MAP: Record<string, string> = {
    RED: "#f44", GREEN: "#4f4", BROWN: "#a80", BLUE: "#44f",
    MAGENTA: "#f4f", CYAN: "#4ff", GREY: "#aaa", ORANGE: "#fa0",
    "BRIGHT GREEN": "#0f0", YELLOW: "#ff0", "BRIGHT BLUE": "#88f",
    "BRIGHT MAGENTA": "#f8f", "BRIGHT CYAN": "#8ff", WHITE: "#fff",
    BLACK: "#444", "NO COLOR": "#888",
};

interface ObjectLinkProps {
    objectId: number;
    glyph: string;
    color?: string;
    label?: string;
}

export function ObjectLink({ objectId, glyph, color, label }: Readonly<ObjectLinkProps>) {
    const [opened, setOpened] = useState(false);
    const fg = color ? NH_COLOR_MAP[color] ?? "#fff" : "#fff";

    return (
        <>
            <Text
                component="span"
                ff="monospace"
                fw={700}
                role="button"
                tabIndex={0}
                onClick={() => setOpened(true)}
                onKeyDown={(e) => { if (e.key === "Enter") setOpened(true); }}
                style={{
                    color: fg,
                    background: "#000",
                    padding: "0 3px",
                    borderRadius: 2,
                    fontSize: 13,
                    lineHeight: 1.2,
                    cursor: "pointer",
                }}
                title={label}
            >
                {glyph}
            </Text>
            <ObjectModal
                objectId={objectId}
                opened={opened}
                onClose={() => setOpened(false)}
                glyph={glyph}
                color={fg}
            />
        </>
    );
}
