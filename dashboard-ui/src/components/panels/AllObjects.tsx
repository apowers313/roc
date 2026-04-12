/** All Objects panel -- sortable table of every resolved object. */

import { Group, Switch, Table, Text } from "@mantine/core";
import { type ReactNode, useCallback, useMemo, useState } from "react";

import type { ResolvedObject } from "../../api/client";
import { useAllObjects } from "../../api/queries";
import { ObjectLink } from "../common/ObjectLink";

/** Map NetHack color names to CSS colors. */
const NH_COLOR_MAP: Record<string, string> = {
    RED: "#f44", GREEN: "#4f4", BROWN: "#a80", BLUE: "#44f",
    MAGENTA: "#f4f", CYAN: "#4ff", GREY: "#aaa", ORANGE: "#fa0",
    "BRIGHT GREEN": "#0f0", YELLOW: "#ff0", "BRIGHT BLUE": "#88f",
    "BRIGHT MAGENTA": "#f8f", "BRIGHT CYAN": "#8ff", WHITE: "#fff",
    BLACK: "#444", "NO COLOR": "#888",
};

interface GameStepRange {
    game_number: number;
    min: number;
    max: number;
}

interface AllObjectsProps {
    run: string;
    game?: number;
    /** Per-game step ranges, used to attribute step_added to its origin game
     *  when an object was first observed in an earlier game (BUG-M2). */
    gameStepRanges?: ReadonlyArray<GameStepRange>;
    onStepClick?: (step: number) => void;
}

type SortKey = "shape" | "glyph" | "color" | "type" | "node_id" | "step_added" | "match_count";

function toSortString(v: unknown): string {
    if (v == null) return "";
    switch (typeof v) {
        case "string":
            return v;
        case "number":
        case "boolean":
        case "bigint":
            return String(v);
        case "object":
            return JSON.stringify(v);
        default:
            return JSON.stringify(v);
    }
}

function compareValues(a: unknown, b: unknown): number {
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    if (typeof a === "number" && typeof b === "number") return a - b;
    return toSortString(a).localeCompare(toSortString(b));
}

export function AllObjects({
    run,
    game,
    gameStepRanges,
    onStepClick,
}: Readonly<AllObjectsProps>) {
    // BUG-H4 fix: always fetch the unfiltered dataset alongside the
    // filtered one. The unfiltered query is the source of truth for
    // identity (canonical PHYSICAL features) and totals; the filtered
    // query produces the per-game match counts and visibility set.
    const { data: filteredObjects } = useAllObjects(run, game);
    const { data: allObjects } = useAllObjects(run, undefined);

    const [sortKey, setSortKey] = useState<SortKey>("shape");
    const [sortAsc, setSortAsc] = useState(true);
    const [showAllGames, setShowAllGames] = useState(false);

    const isFiltering = game !== undefined && game > 0;
    const displayed = !isFiltering || showAllGames ? allObjects : filteredObjects;
    const filteredCount = filteredObjects?.length ?? 0;
    const totalCount = allObjects?.length ?? 0;

    const handleSort = useCallback((key: SortKey) => {
        if (key === sortKey) {
            setSortAsc((prev) => !prev);
        } else {
            setSortKey(key);
            setSortAsc(true);
        }
    }, [sortKey]);

    // Map step_added -> game_number, so we can flag rows whose canonical
    // first-observation step belongs to an earlier game (BUG-M2).
    const stepToGame = useMemo(() => {
        const ranges = gameStepRanges ?? [];
        return (step: number | null | undefined): number | null => {
            if (step == null) return null;
            for (const r of ranges) {
                if (step >= r.min && step <= r.max) return r.game_number;
            }
            return null;
        };
    }, [gameStepRanges]);

    if (!displayed || displayed.length === 0) {
        return (
            <>
                {isFiltering && (
                    <FilterHeader
                        game={game}
                        filteredCount={filteredCount}
                        totalCount={totalCount}
                        showAllGames={showAllGames}
                        onToggle={setShowAllGames}
                    />
                )}
                <Text size="xs" c="dimmed">
                    No objects
                </Text>
            </>
        );
    }

    const sorted = [...displayed].sort((a, b) => {
        const cmp = compareValues(a[sortKey], b[sortKey]);
        return sortAsc ? cmp : -cmp;
    });

    const columns: { key: SortKey; label: string; width?: string | number }[] = [
        { key: "shape", label: "shape", width: 40 },
        { key: "glyph", label: "glyph", width: 60 },
        { key: "color", label: "color" },
        { key: "type", label: "type", width: 50 },
        { key: "node_id", label: "node id" },
        { key: "step_added", label: "step added", width: 80 },
        { key: "match_count", label: "matches", width: 70 },
    ];

    const arrow = (key: SortKey) => {
        if (sortKey !== key) return "";
        return sortAsc ? " \u25B2" : " \u25BC";
    };

    return (
        <div style={{ maxHeight: "calc(100vh - 120px)", overflowY: "auto" }}>
            {isFiltering && (
                <FilterHeader
                    game={game}
                    filteredCount={filteredCount}
                    totalCount={totalCount}
                    showAllGames={showAllGames}
                    onToggle={setShowAllGames}
                />
            )}
            <Table
                striped
                highlightOnHover
                withTableBorder
                withColumnBorders
                fz="xs"
                layout="fixed"
            >
                <Table.Thead>
                    <Table.Tr>
                        <Table.Th style={{ fontSize: 10, fontWeight: 600, padding: "2px 4px", width: 30 }}>
                            #
                        </Table.Th>
                        {columns.map((col) => (
                            <Table.Th
                                key={col.key}
                                onClick={() => handleSort(col.key)}
                                style={{
                                    fontSize: 10,
                                    fontWeight: 600,
                                    padding: "2px 4px",
                                    cursor: "pointer",
                                    userSelect: "none",
                                    width: col.width,
                                }}
                            >
                                {col.label}{arrow(col.key)}
                            </Table.Th>
                        ))}
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {sorted.map((obj: ResolvedObject, i: number) => {
                        const fg = obj.color ? NH_COLOR_MAP[obj.color] ?? "#fff" : "#fff";
                        let shapeCell: ReactNode;
                        if (obj.shape && obj.node_id) {
                            shapeCell = (
                                <ObjectLink
                                    objectId={Number(obj.node_id)}
                                    glyph={obj.shape}
                                    color={obj.color ?? undefined}
                                />
                            );
                        } else if (obj.shape) {
                            shapeCell = (
                                <Text
                                    component="span"
                                    ff="monospace"
                                    fw={700}
                                    style={{ color: fg, background: "#000", padding: "0 3px", borderRadius: 2, fontSize: 13 }}
                                >
                                    {obj.shape}
                                </Text>
                            );
                        } else {
                            shapeCell = "--";
                        }
                        return (
                            <Table.Tr
                                key={obj.node_id ?? `new-${obj.step_added}`}
                                onClick={obj.step_added != null && onStepClick ? () => onStepClick(obj.step_added!) : undefined}
                                style={{
                                    cursor: obj.step_added != null && onStepClick ? "pointer" : undefined,
                                }}
                            >
                                <Table.Td style={{ fontSize: 10, padding: "1px 4px", color: "var(--mantine-color-dimmed)" }}>
                                    {i + 1}
                                </Table.Td>
                                <Table.Td style={{ padding: "1px 4px", textAlign: "center" }}>
                                    {shapeCell}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.glyph ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.color ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.type ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.node_id ?? "--"}
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    <StepAddedCell
                                        stepAdded={obj.step_added}
                                        currentGame={game}
                                        originGame={stepToGame(obj.step_added)}
                                    />
                                </Table.Td>
                                <Table.Td style={{ fontSize: 10, fontFamily: "monospace", padding: "1px 4px" }}>
                                    {obj.match_count}
                                </Table.Td>
                            </Table.Tr>
                        );
                    })}
                </Table.Tbody>
            </Table>
        </div>
    );
}

interface FilterHeaderProps {
    game: number;
    filteredCount: number;
    totalCount: number;
    showAllGames: boolean;
    onToggle: (next: boolean) => void;
}

function FilterHeader({
    game,
    filteredCount,
    totalCount,
    showAllGames,
    onToggle,
}: Readonly<FilterHeaderProps>) {
    const title = showAllGames ? "All Objects" : `Objects in Game ${game}`;
    const countText = showAllGames ? `${totalCount}` : `${filteredCount} / ${totalCount}`;
    return (
        <Group justify="space-between" align="center" mb={4} px={4}>
            <Group gap={6}>
                <Text size="xs" fw={600}>
                    {title}
                </Text>
                <Text size="xs" c="dimmed">
                    {countText}
                </Text>
            </Group>
            <Switch
                size="xs"
                label="Show all games"
                labelPosition="left"
                checked={showAllGames}
                onChange={(e) => onToggle(e.currentTarget.checked)}
            />
        </Group>
    );
}

interface StepAddedCellProps {
    stepAdded: number | null;
    currentGame: number | undefined;
    originGame: number | null;
}

function StepAddedCell({ stepAdded, currentGame, originGame }: Readonly<StepAddedCellProps>) {
    if (stepAdded == null) {
        return <span>--</span>;
    }
    if (
        currentGame !== undefined &&
        currentGame > 0 &&
        originGame !== null &&
        originGame !== currentGame
    ) {
        // Native title tooltip + a visible "(game N)" annotation so the
        // origin is discoverable without hovering. The tooltip text is
        // also queryable from tests via the title attribute.
        const tooltip = `created in game ${originGame}`;
        return (
            <span title={tooltip} style={{ cursor: "help" }}>
                {stepAdded}{" "}
                <Text component="span" size="xs" c="dimmed">
                    ({tooltip})
                </Text>
            </span>
        );
    }
    return <span>{stepAdded}</span>;
}
