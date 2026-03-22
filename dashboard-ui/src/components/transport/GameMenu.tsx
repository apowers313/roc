/** Game lifecycle menu -- start/stop games from the dashboard. */

import { ActionIcon, Badge, Menu, NumberInput, Text } from "@mantine/core";
import { Gamepad2 } from "lucide-react";
import { useCallback, useState } from "react";

interface GameState {
    state: string;
    run_name?: string | null;
    exit_code?: number | null;
    error?: string | null;
}

export function GameMenu() {
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

    return (
        <Menu
            position="bottom-end"
            withArrow
            shadow="md"
            onOpen={() => void refreshStatus()}
        >
            <Menu.Target>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    color={isRunning ? "green" : isStopping ? "yellow" : "gray"}
                    aria-label="Game menu"
                >
                    <Gamepad2 size={14} />
                </ActionIcon>
            </Menu.Target>

            <Menu.Dropdown>
                <Menu.Label>
                    {isRunning ? (
                        <Badge color="green" size="xs" variant="filled">
                            Game Running
                        </Badge>
                    ) : isStopping ? (
                        <Badge color="yellow" size="xs" variant="filled">
                            Stopping...
                        </Badge>
                    ) : (
                        <Text size="xs" c="dimmed">
                            No game running
                        </Text>
                    )}
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
