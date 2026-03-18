/** Object info panel -- object detection results from the resolution pipeline. */

import { Table, Text } from "@mantine/core";

import type { StepData } from "../../types/step-data";

interface ObjectInfoProps {
    data: StepData | undefined;
}

export function ObjectInfo({ data }: ObjectInfoProps) {
    const objects = data?.object_info;

    if (!objects || objects.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No object data
            </Text>
        );
    }

    return (
        <div style={{ maxHeight: 200, overflowY: "auto" }}>
            <Table
                horizontalSpacing={4}
                verticalSpacing={1}
                withRowBorders={false}
                layout="fixed"
            >
                <Table.Tbody>
                    {objects.map((obj, i) => (
                        <Table.Tr key={i}>
                            <Table.Td>
                                <Text size="xs" style={{ fontFamily: "monospace" }}>
                                    {obj.raw != null
                                        ? String(obj.raw)
                                        : JSON.stringify(obj)}
                                </Text>
                            </Table.Td>
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </div>
    );
}
