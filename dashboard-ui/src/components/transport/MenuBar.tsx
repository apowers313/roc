/** Menu bar with Game and utility actions. */

import { Badge, Group, Menu, NumberInput, Text, UnstyledButton } from "@mantine/core";
import { Link, Gamepad2 } from "lucide-react";
import { useCallback, useState } from "react";

interface GameState {
    state: string;
    run_name?: string | null;
    exit_code?: number | null;
    error?: string | null;
}

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
    const [gameState, setGameState] = useState<GameState>({ state: "idle" });
    const [numGames, setNumGames] = useState<number>(5);
    const [loading, setLoading] = useState(false);

    const refreshStatus = useCallback(async () => {
        try {
            const res = await fetch("/api/game/status");
            if (res.ok) {
                const data: GameState = await res.json();
                setGameState(data);
            }
        } catch {
            // ignore
        }
    }, []);

    const startGame = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch(`/api/game/start?num_games=${numGames}`, {
                method: "POST",
            });
            if (res.ok) {
                await refreshStatus();
            }
        } catch {
            // ignore
        } finally {
            setLoading(false);
        }
    }, [numGames, refreshStatus]);

    const stopGame = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch("/api/game/stop", { method: "POST" });
            if (res.ok) {
                await refreshStatus();
            }
        } catch {
            // ignore
        } finally {
            setLoading(false);
        }
    }, [refreshStatus]);

    const isRunning =
        gameState.state === "running" || gameState.state === "initializing";
    const isStopping = gameState.state === "stopping";

    const iconColor = getGameIconColor(isRunning, isStopping);

    return (
        <Menu position="bottom-start" withArrow shadow="md" onOpen={() => void refreshStatus()}>
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

                {gameState.error && !isRunning && (
                    <Menu.Label>
                        <Text size="xs" c="red">
                            {gameState.error}
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
