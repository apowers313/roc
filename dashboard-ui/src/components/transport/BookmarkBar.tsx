/** Bookmark toggle button and visual marker strip on the slider track. */

import { ActionIcon, Tooltip } from "@mantine/core";
import { Bookmark as BookmarkIcon, BookmarkCheck } from "lucide-react";

import type { Bookmark } from "../../types/api";

interface BookmarkBarProps {
    bookmarks: Bookmark[];
    currentStep: number;
    stepMin: number;
    stepMax: number;
    isBookmarked: boolean;
    onToggle: () => void;
    onNavigate: (bookmark: { step: number; game: number }) => void;
}

export function BookmarkBar({
    bookmarks,
    currentStep,
    stepMin,
    stepMax,
    isBookmarked,
    onToggle,
    onNavigate,
}: Readonly<BookmarkBarProps>) {
    const range = stepMax - stepMin;

    return (
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Tooltip label={isBookmarked ? "Remove bookmark" : "Add bookmark"}>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    color={isBookmarked ? "yellow" : "gray"}
                    onClick={onToggle}
                    aria-label={isBookmarked ? "Remove bookmark" : "Add bookmark"}
                >
                    {isBookmarked ? (
                        <BookmarkCheck size={14} />
                    ) : (
                        <BookmarkIcon size={14} />
                    )}
                </ActionIcon>
            </Tooltip>

            {bookmarks.length > 0 && (
                <div
                    style={{
                        position: "relative",
                        flex: 1,
                        height: 8,
                        minWidth: 200,
                    }}
                >
                    {bookmarks.map((b) => {
                        const pct = range > 0
                            ? ((b.step - stepMin) / range) * 100
                            : 0;
                        const label = b.annotation
                            ? `Step ${b.step}: ${b.annotation}`
                            : `Step ${b.step}`;
                        return (
                            <button
                                key={b.step}
                                type="button"
                                data-testid="bookmark-marker"
                                title={label}
                                onClick={() => onNavigate(b)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter" || e.key === " ") onNavigate(b);
                                }}
                                style={{
                                    position: "absolute",
                                    left: `${pct}%`,
                                    top: 0,
                                    width: 6,
                                    height: 8,
                                    marginLeft: -3,
                                    borderRadius: 1,
                                    background:
                                        b.step === currentStep
                                            ? "#ffd43b"
                                            : "#fab005",
                                    cursor: "pointer",
                                    opacity: b.step === currentStep ? 1 : 0.7,
                                    border: "none",
                                    padding: 0,
                                }}
                                aria-label={label}
                            />
                        );
                    })}
                </div>
            )}
        </div>
    );
}
