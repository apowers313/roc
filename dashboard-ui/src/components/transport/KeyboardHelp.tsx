/** Keyboard shortcuts help modal. */

import { Kbd, Modal, Table, Text } from "@mantine/core";

interface KeyboardHelpProps {
    opened: boolean;
    onClose: () => void;
}

interface Shortcut {
    keys: readonly string[];
    altKeys?: readonly string[];
    description: string;
}

const SHORTCUTS: readonly Shortcut[] = [
    { keys: ["Right"], description: "Next step" },
    { keys: ["Left"], description: "Previous step" },
    { keys: ["Shift", "Right"], description: "+10 steps" },
    { keys: ["Shift", "Left"], description: "-10 steps" },
    { keys: ["Space"], description: "Play / Pause" },
    { keys: ["Home"], altKeys: ["Ctrl", "Left"], description: "First step" },
    { keys: ["End"], altKeys: ["Ctrl", "Right"], description: "Last step" },
    { keys: ["B"], description: "Toggle bookmark" },
    { keys: ["]"], description: "Next bookmark" },
    { keys: ["["], description: "Previous bookmark" },
    { keys: ["+"], altKeys: ["="], description: "Faster playback" },
    { keys: ["-"], description: "Slower playback" },
    { keys: ["G"], description: "Next game" },
    { keys: ["L"], description: "Go live" },
    { keys: ["?"], description: "Show this help" },
];

function renderKeys(keys: readonly string[]) {
    return keys.map((k, i) => (
        <span key={k}>
            {i > 0 && (
                <Text span size="xs" c="dimmed">
                    {" + "}
                </Text>
            )}
            <Kbd size="xs">{k}</Kbd>
        </span>
    ));
}

export function KeyboardHelp({ opened, onClose }: Readonly<KeyboardHelpProps>) {
    return (
        <Modal
            opened={opened}
            onClose={onClose}
            title="Keyboard Shortcuts"
            size="sm"
        >
            <Table>
                <Table.Tbody>
                    {SHORTCUTS.map((s) => (
                        <Table.Tr key={s.description}>
                            <Table.Td>
                                {renderKeys(s.keys)}
                                {s.altKeys && (
                                    <>
                                        <Text span size="xs" c="dimmed">
                                            {" / "}
                                        </Text>
                                        {renderKeys(s.altKeys)}
                                    </>
                                )}
                            </Table.Td>
                            <Table.Td>
                                <Text size="xs">{s.description}</Text>
                            </Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </Modal>
    );
}
