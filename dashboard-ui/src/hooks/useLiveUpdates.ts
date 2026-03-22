/** Socket.io hook for receiving live step push updates from the game loop. */

import { useCallback, useEffect, useRef, useState } from "react";
import { io, type Socket } from "socket.io-client";

import type { StepData } from "../types/step-data";

interface LiveStatus {
    active: boolean;
    run_name: string | null;
    step: number;
    game_number: number;
    step_min: number;
    step_max: number;
    game_numbers: number[];
}

interface UseLiveUpdatesOptions {
    /** Called when a new step arrives from the game loop (full StepData). */
    onNewStep?: (data: StepData) => void;
}

export function useLiveUpdates({ onNewStep }: UseLiveUpdatesOptions = {}) {
    const [connected, setConnected] = useState(false);
    const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
    const socketRef = useRef<Socket | null>(null);
    const onNewStepRef = useRef(onNewStep);
    onNewStepRef.current = onNewStep;
    const pollLiveStatusRef = useRef<() => Promise<void>>(() => Promise.resolve());

    useEffect(() => {
        // Connect to the API server's Socket.io endpoint.
        // In dev mode, Vite proxies /socket.io to the API server.
        const socket = io({
            path: "/socket.io",
            transports: ["polling", "websocket"],
        });
        socketRef.current = socket;

        socket.on("connect", () => {
            setConnected(true);
        });

        socket.on("disconnect", () => {
            setConnected(false);
        });

        socket.on("new_step", (data: StepData) => {
            onNewStepRef.current?.(data);
        });

        // React immediately to game state changes instead of waiting for poll
        socket.on("game_state_changed", () => {
            void pollLiveStatusRef.current();
        });

        return () => {
            socket.disconnect();
            socketRef.current = null;
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Poll live status periodically. Ref is used by Socket.io callback above.
    const pollLiveStatus = useCallback(async () => {
        try {
            const res = await fetch("/api/live/status");
            if (res.ok) {
                const status: LiveStatus = await res.json();
                setLiveStatus(status);
            }
        } catch {
            // ignore fetch errors
        }
    }, []);

    pollLiveStatusRef.current = pollLiveStatus;

    useEffect(() => {
        void pollLiveStatus();
        const interval = setInterval(() => void pollLiveStatus(), 3000);
        return () => clearInterval(interval);
    }, [pollLiveStatus]);

    return { connected, liveStatus };
}
