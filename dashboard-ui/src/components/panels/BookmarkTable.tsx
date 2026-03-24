/** Bookmark table -- tabular view of all bookmarks with click-to-jump and inline editing. */

import { Table, Text, TextInput } from "@mantine/core";
import { useState } from "react";

import type { Bookmark } from "../../types/api";

type EditableField = "step" | "game" | "annotation";

interface BookmarkTableProps {
    bookmarks: Bookmark[];
    currentStep: number;
    onNavigate: (bookmark: { step: number; game: number }) => void;
    onUpdateBookmark: (oldStep: number, updates: Partial<Pick<Bookmark, EditableField>>) => void;
}
type EditingField = { step: number; field: EditableField };

export function BookmarkTable({
    bookmarks,
    currentStep,
    onNavigate,
    onUpdateBookmark,
}: Readonly<BookmarkTableProps>) {
    const [editing, setEditing] = useState<EditingField | null>(null);
    const [editValue, setEditValue] = useState("");

    if (bookmarks.length === 0) {
        return (
            <Text size="xs" c="dimmed">
                No bookmarks. Press B to bookmark the current step.
            </Text>
        );
    }

    const sorted = [...bookmarks].sort((a, b) => a.step - b.step);

    function startEditing(step: number, field: "step" | "game" | "annotation", currentValue: string) {
        setEditing({ step, field });
        setEditValue(currentValue);
    }

    function commitEdit() {
        if (!editing) return;
        const { step, field } = editing;
        if (field === "step") {
            const newStep = Number.parseInt(editValue, 10);
            if (Number.isFinite(newStep) && newStep > 0) {
                onUpdateBookmark(step, { step: newStep });
            }
        } else if (field === "game") {
            const newGame = Number.parseInt(editValue, 10);
            if (Number.isFinite(newGame) && newGame > 0) {
                onUpdateBookmark(step, { game: newGame });
            }
        } else {
            onUpdateBookmark(step, { annotation: editValue });
        }
        setEditing(null);
    }

    function renderEditableCell(
        bookmark: Bookmark,
        field: "step" | "game" | "annotation",
        displayValue: string,
    ) {
        const isEditing = editing?.step === bookmark.step && editing?.field === field;
        if (isEditing) {
            return (
                <TextInput
                    size="xs"
                    value={editValue}
                    onChange={(e) => setEditValue(e.currentTarget.value)}
                    onBlur={commitEdit}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            commitEdit();
                        } else if (e.key === "Escape") {
                            setEditing(null);
                        }
                        e.stopPropagation();
                    }}
                    autoFocus
                    styles={{ input: { fontSize: 10, height: 20, minHeight: 20, fontFamily: field === "annotation" ? undefined : "monospace" } }}
                />
            );
        }
        const isAnnotation = field === "annotation";
        return (
            <Text
                size="xs"
                c={isAnnotation && !displayValue ? "dimmed" : undefined}
                style={isAnnotation ? undefined : { fontFamily: "monospace" }}
            >
                {isAnnotation ? (displayValue || "double-click to edit") : displayValue}
            </Text>
        );
    }

    return (
        <div style={{ maxHeight: 200, overflowY: "auto" }}>
            <Table
                horizontalSpacing={4}
                verticalSpacing={1}
                withRowBorders
                striped
                highlightOnHover
            >
                <Table.Thead>
                    <Table.Tr>
                        <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: 60 }}>
                            Step
                        </Table.Th>
                        <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: 50 }}>
                            Game
                        </Table.Th>
                        <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px" }}>
                            Annotation
                        </Table.Th>
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {sorted.map((b) => {
                        const isCurrent = b.step === currentStep;
                        return (
                            <Table.Tr
                                key={b.step}
                                style={{
                                    cursor: "pointer",
                                    fontWeight: isCurrent ? 600 : 400,
                                    background: isCurrent ? "var(--mantine-color-yellow-light)" : undefined,
                                }}
                                onClick={() => onNavigate(b)}
                            >
                                <Table.Td
                                    style={{ fontSize: 10, padding: "1px 4px" }}
                                    onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        startEditing(b.step, "step", String(b.step));
                                    }}
                                >
                                    {renderEditableCell(b, "step", String(b.step))}
                                </Table.Td>
                                <Table.Td
                                    style={{ fontSize: 10, padding: "1px 4px" }}
                                    onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        startEditing(b.step, "game", String(b.game));
                                    }}
                                >
                                    {renderEditableCell(b, "game", String(b.game))}
                                </Table.Td>
                                <Table.Td
                                    style={{ fontSize: 10, padding: "1px 4px" }}
                                    onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        startEditing(b.step, "annotation", b.annotation);
                                    }}
                                >
                                    {renderEditableCell(b, "annotation", b.annotation)}
                                </Table.Td>
                            </Table.Tr>
                        );
                    })}
                </Table.Tbody>
            </Table>
        </div>
    );
}
