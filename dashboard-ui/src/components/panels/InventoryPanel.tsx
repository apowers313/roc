/** Inventory panel -- what the agent is carrying. */

import { Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface InventoryPanelProps {
    data: StepData | undefined;
}

export function InventoryPanel({ data }: Readonly<InventoryPanelProps>) {
    const inv = data?.inventory;

    if (!inv || inv.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No inventory data
            </Text>
        );
    }

    return (
        <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
            <Table.Thead>
                <Table.Tr>
                    <Table.Th style={{ width: 40 }}>Slot</Table.Th>
                    <Table.Th>Item</Table.Th>
                </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
                {inv.map((item) => (
                    <Table.Tr key={item.letter}>
                        <Table.Td ff="monospace">{item.letter}</Table.Td>
                        <Table.Td>{item.item}</Table.Td>
                    </Table.Tr>
                ))}
            </Table.Tbody>
        </Table>
    );
}
