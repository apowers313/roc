/** Schema panel -- displays graph database node/edge schema for a run. */

import {
    Accordion,
    Badge,
    Code,
    Group,
    Loader,
    ScrollArea,
    Stack,
    Table,
    Text,
    Tooltip,
} from "@mantine/core";

import { useSchema } from "../../api/queries";
import { DiagramViewer } from "../common/DiagramViewer";

interface SchemaPanelProps {
    run: string;
}

export function SchemaPanel({ run }: SchemaPanelProps) {
    const { data: schema, isLoading, error } = useSchema(run);

    if (isLoading) {
        return (
            <Group gap="xs">
                <Loader size="xs" />
                <Text size="xs" c="dimmed">Loading schema...</Text>
            </Group>
        );
    }

    if (error || !schema) {
        return (
            <Text size="xs" c="dimmed">
                No schema available for this run
            </Text>
        );
    }

    return (
        <Stack gap="md">
            {schema.mermaid && (
                <DiagramViewer definition={schema.mermaid} filename="schema" />
            )}

            <div>
                <Text size="sm" fw={600} mb={4}>
                    Summary
                </Text>
                <Group gap="md">
                    <Badge variant="light" color="blue" size="lg">
                        {schema.nodes.length} Nodes
                    </Badge>
                    <Badge variant="light" color="grape" size="lg">
                        {schema.edges.length} Edges
                    </Badge>
                    <Badge variant="light" color="gray" size="lg">
                        {schema.edges.reduce((sum, e) => sum + e.connections.length, 0)} Connections
                    </Badge>
                </Group>
            </div>

            <div>
                <Text size="sm" fw={600} mb={4}>
                    Nodes
                </Text>
                <Accordion variant="separated" multiple>
                    {schema.nodes.map((node) => (
                        <Accordion.Item key={node.name} value={node.name}>
                            <Accordion.Control>
                                <Group gap="xs">
                                    <Text size="xs" fw={600}>{node.name}</Text>
                                    {node.parents.length > 0 && (
                                        <Text size="xs" c="dimmed">
                                            extends {node.parents.join(", ")}
                                        </Text>
                                    )}
                                    <Badge variant="light" color="blue" size="xs">
                                        {node.fields.filter((f) => f.local && !f.exclude).length} fields
                                    </Badge>
                                    {node.methods.filter((m) => m.local).length > 0 && (
                                        <Badge variant="light" color="teal" size="xs">
                                            {node.methods.filter((m) => m.local).length} methods
                                        </Badge>
                                    )}
                                </Group>
                            </Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    {node.fields.filter((f) => !f.exclude).length > 0 && (
                                        <ScrollArea>
                                            <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
                                                <Table.Thead>
                                                    <Table.Tr>
                                                        <Table.Th>Field</Table.Th>
                                                        <Table.Th>Type</Table.Th>
                                                        <Table.Th>Default</Table.Th>
                                                        <Table.Th>Scope</Table.Th>
                                                    </Table.Tr>
                                                </Table.Thead>
                                                <Table.Tbody>
                                                    {node.fields
                                                        .filter((f) => !f.exclude)
                                                        .map((f) => (
                                                            <Table.Tr key={f.name}>
                                                                <Table.Td>
                                                                    <Code>{f.name}</Code>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Text size="xs" c="dimmed">{f.type}</Text>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    {f.default != null ? (
                                                                        <Tooltip label={f.default} withArrow>
                                                                            <Text size="xs" c="dimmed" truncate="end" maw={200}>
                                                                                {f.default}
                                                                            </Text>
                                                                        </Tooltip>
                                                                    ) : (
                                                                        <Text size="xs" c="dimmed">--</Text>
                                                                    )}
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Badge
                                                                        variant="light"
                                                                        color={f.local ? "blue" : "gray"}
                                                                        size="xs"
                                                                    >
                                                                        {f.local ? "local" : "inherited"}
                                                                    </Badge>
                                                                </Table.Td>
                                                            </Table.Tr>
                                                        ))}
                                                </Table.Tbody>
                                            </Table>
                                        </ScrollArea>
                                    )}
                                    {node.methods.length > 0 && (
                                        <>
                                            <Text size="xs" fw={600} mt={4}>Methods</Text>
                                            <ScrollArea>
                                                <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
                                                    <Table.Thead>
                                                        <Table.Tr>
                                                            <Table.Th>Method</Table.Th>
                                                            <Table.Th>Parameters</Table.Th>
                                                            <Table.Th>Returns</Table.Th>
                                                            <Table.Th>Scope</Table.Th>
                                                        </Table.Tr>
                                                    </Table.Thead>
                                                    <Table.Tbody>
                                                        {node.methods.map((m) => (
                                                            <Table.Tr key={m.name}>
                                                                <Table.Td>
                                                                    <Code>{m.name}</Code>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Text size="xs" c="dimmed">
                                                                        {m.params || "--"}
                                                                    </Text>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Text size="xs" c="dimmed">{m.return_type}</Text>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Badge
                                                                        variant="light"
                                                                        color={m.local ? "teal" : "gray"}
                                                                        size="xs"
                                                                    >
                                                                        {m.local ? "local" : "inherited"}
                                                                    </Badge>
                                                                </Table.Td>
                                                            </Table.Tr>
                                                        ))}
                                                    </Table.Tbody>
                                                </Table>
                                            </ScrollArea>
                                        </>
                                    )}
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>
                    ))}
                </Accordion>
            </div>

            <div>
                <Text size="sm" fw={600} mb={4}>
                    Edges
                </Text>
                <Accordion variant="separated" multiple>
                    {schema.edges.map((edge) => (
                        <Accordion.Item key={edge.name} value={edge.name}>
                            <Accordion.Control>
                                <Group gap="xs">
                                    <Text size="xs" fw={600}>{edge.name}</Text>
                                    {edge.type !== edge.name && (
                                        <Text size="xs" c="dimmed">
                                            type: {edge.type}
                                        </Text>
                                    )}
                                    <Badge variant="light" color="grape" size="xs">
                                        {edge.connections.length} connections
                                    </Badge>
                                </Group>
                            </Accordion.Control>
                            <Accordion.Panel>
                                <ScrollArea>
                                    <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
                                        <Table.Thead>
                                            <Table.Tr>
                                                <Table.Th>Source</Table.Th>
                                                <Table.Th></Table.Th>
                                                <Table.Th>Destination</Table.Th>
                                            </Table.Tr>
                                        </Table.Thead>
                                        <Table.Tbody>
                                            {edge.connections.map(([src, dst], i) => (
                                                <Table.Tr key={i}>
                                                    <Table.Td>
                                                        <Code>{src}</Code>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Text size="xs" c="dimmed" ta="center">
                                                            --&gt;
                                                        </Text>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Code>{dst}</Code>
                                                    </Table.Td>
                                                </Table.Tr>
                                            ))}
                                        </Table.Tbody>
                                    </Table>
                                </ScrollArea>
                                {edge.fields.filter((f) => f.local && !f.exclude).length > 0 && (
                                    <>
                                        <Text size="xs" fw={600} mt={8} mb={4}>Fields</Text>
                                        <ScrollArea>
                                            <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
                                                <Table.Thead>
                                                    <Table.Tr>
                                                        <Table.Th>Field</Table.Th>
                                                        <Table.Th>Type</Table.Th>
                                                        <Table.Th>Default</Table.Th>
                                                    </Table.Tr>
                                                </Table.Thead>
                                                <Table.Tbody>
                                                    {edge.fields
                                                        .filter((f) => f.local && !f.exclude)
                                                        .map((f) => (
                                                            <Table.Tr key={f.name}>
                                                                <Table.Td>
                                                                    <Code>{f.name}</Code>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Text size="xs" c="dimmed">{f.type}</Text>
                                                                </Table.Td>
                                                                <Table.Td>
                                                                    <Text size="xs" c="dimmed">
                                                                        {f.default ?? "--"}
                                                                    </Text>
                                                                </Table.Td>
                                                            </Table.Tr>
                                                        ))}
                                                </Table.Tbody>
                                            </Table>
                                        </ScrollArea>
                                    </>
                                )}
                            </Accordion.Panel>
                        </Accordion.Item>
                    ))}
                </Accordion>
            </div>
        </Stack>
    );
}
