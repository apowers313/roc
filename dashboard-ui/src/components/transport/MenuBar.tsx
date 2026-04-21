/** Menu bar with Game and utility actions.
 *
 * Game state reads come exclusively from ``useGameState`` (the
 * dashboard's single source of truth: Socket.io event stream plus a
 * one-shot REST fetch on mount). MenuBar does not maintain its own
 * copy of the game status -- it only owns the local UX state for
 * start/stop actions (``numGames``, ``loading``) and the clipboard
 * copy indicator. This is the consolidated replacement for the prior
 * GameMenu.tsx + nested ``GameMenu`` duplication that caused
 * TC-GAME-004 (stale local state diverging from the live hook).
 */

import { Badge, Group, Menu, NumberInput, Text, UnstyledButton } from "@mantine/core";
import { Link, Gamepad2 } from "lucide-react";
import { useCallback, useState } from "react";

import { useGameState } from "../../hooks/useRunSubscription";

function getGameIconColor(isRunning: boolean, isStopping: boolean): string | undefined {
    if (isRunning) return "var(--mantine-color-green-5)";
    if (isStopping) return "var(--mantine-color-yellow-5)";
    return undefined;
}

function GameStatusLabel({ isRunning, isStopping }: Readonly<{ isRunning: boolean; isStopping: boolean }>) {
    if (isRunning) {
        return (
            <Badge color="green" size="xs" variant="filled">
                Game Running
            </Badge>
        );
    }
    if (isStopping) {
        return (
            <Badge color="yellow" size="xs" variant="filled">
                Stopping...
            </Badge>
        );
    }
    return (
        <Text size="xs" c="dimmed">
            No game running
        </Text>
    );
}

function GameMenu() {
    const gameState = useGameState();
    const [numGames, setNumGames] = useState<number>(5);
    const [loading, setLoading] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);

    const startGame = useCallback(async () => {
        setLoading(true);
        setActionError(null);
        try {
            const res = await fetch(`/api/game/start?num_games=${numGames}`, {
                method: "POST",
            });
            if (!res.ok) {
                const msg = `Start game failed: ${res.status} ${res.statusText}`;
                console.error(msg);
                setActionError(msg);
            }
        } catch (err) {
            const msg = `Start game failed: ${err instanceof Error ? err.message : String(err)}`;
            console.error(msg);
            setActionError(msg);
        } finally {
            setLoading(false);
        }
    }, [numGames]);

    const stopGame = useCallback(async () => {
        setLoading(true);
        setActionError(null);
        try {
            const res = await fetch("/api/game/stop", { method: "POST" });
            if (!res.ok) {
                const msg = `Stop game failed: ${res.status} ${res.statusText}`;
                console.error(msg);
                setActionError(msg);
            }
        } catch (err) {
            const msg = `Stop game failed: ${err instanceof Error ? err.message : String(err)}`;
            console.error(msg);
            setActionError(msg);
        } finally {
            setLoading(false);
        }
    }, []);

    // Default to "idle" before the one-shot initial fetch in
    // useGameState resolves. Treat ``null`` the same as ``idle`` so
    // the menu is always interactive.
    const state = gameState?.state ?? "idle";
    const error = gameState?.error ?? null;
    const isRunning = state === "running" || state === "initializing";
    const isStopping = state === "stopping";

    const iconColor = getGameIconColor(isRunning, isStopping);

    return (
        <Menu position="bottom-start" withArrow shadow="md">
            <Menu.Target>
                <UnstyledButton
                    style={{
                        padding: "2px 8px",
                        borderRadius: 4,
                        fontSize: 12,
                        color: "var(--mantine-color-dimmed)",
                    }}
                >
                    <Group gap={4}>
                        <Gamepad2 size={12} color={iconColor} />
                        <span>Game</span>
                    </Group>
                </UnstyledButton>
            </Menu.Target>

            <Menu.Dropdown>
                <Menu.Label>
                    <GameStatusLabel isRunning={isRunning} isStopping={isStopping} />
                </Menu.Label>

                {(error || actionError) && !isRunning && (
                    <Menu.Label>
                        <Text size="xs" c="red">
                            {actionError ?? error}
                        </Text>
                    </Menu.Label>
                )}

                {!isRunning && !isStopping && (
                    <>
                        <Menu.Divider />
                        <div style={{ padding: "4px 12px" }}>
                            <NumberInput
                                size="xs"
                                label="Number of games"
                                value={numGames}
                                onChange={(v) =>
                                    setNumGames(typeof v === "number" ? v : 5)
                                }
                                min={1}
                                max={100}
                                style={{ width: 160 }}
                            />
                        </div>
                        <Menu.Divider />
                        <Menu.Item
                            onClick={() => void startGame()}
                            disabled={loading}
                        >
                            Start Game
                        </Menu.Item>
                    </>
                )}

                {isRunning && (
                    <>
                        <Menu.Divider />
                        <Menu.Item
                            color="red"
                            onClick={() => void stopGame()}
                            disabled={loading}
                        >
                            Stop Game
                        </Menu.Item>
                    </>
                )}
            </Menu.Dropdown>
        </Menu>
    );
}

function CopyLinkButton() {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(() => {
        void navigator.clipboard.writeText(globalThis.location.href).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        });
    }, []);

    return (
        <UnstyledButton
            onClick={handleCopy}
            style={{
                padding: "2px 8px",
                borderRadius: 4,
                fontSize: 12,
                color: copied ? "var(--mantine-color-green-5)" : "var(--mantine-color-dimmed)",
            }}
        >
            <Group gap={4}>
                <Link size={12} />
                <span>{copied ? "Copied!" : "Copy Link"}</span>
            </Group>
        </UnstyledButton>
    );
}

export function MenuBar() {
    return (
        <Group
            gap={0}
            style={{
                backgroundColor: "var(--mantine-color-dark-7)",
                borderBottom: "1px solid var(--mantine-color-dark-4)",
                padding: "0 8px",
            }}
        >
            <GameMenu />
            <CopyLinkButton />
        </Group>
    );
}
