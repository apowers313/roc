/**
 * Debounce a value -- returns the latest value after it stops changing for
 * `delay` ms.
 *
 * Optional `resetKey`: when this value changes (strict equality), the
 * debounced output immediately adopts the current `value` without waiting.
 * Use this to prevent stale cross-context requests (e.g. pass `run` so
 * that switching runs instantly resets the debounced step instead of
 * emitting the old run's step against the new run's API endpoint).
 */

import { useEffect, useState } from "react";

export function useDebouncedValue<T>(value: T, delay: number, resetKey?: unknown): T {
    const [debounced, setDebounced] = useState(value);
    // Track which resetKey produced the current debounced value.
    // When the key changes we bypass debounced until the state catches up.
    const [settledKey, setSettledKey] = useState(resetKey);

    // When resetKey changes, mark the debounced value as stale.
    // The state update ensures future renders also see the new key.
    if (resetKey !== undefined && resetKey !== settledKey) {
        setSettledKey(resetKey);
        setDebounced(value);
    }

    useEffect(() => {
        const timer = setTimeout(() => setDebounced(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);

    // If the key that produced `debounced` differs from the current
    // resetKey, the debounced value is from a previous context (e.g.
    // an old run). Return `value` directly until state catches up.
    // This covers multi-render propagation: even if the resetKey
    // arrived in render N but `value` only updates in render N+1,
    // we keep bypassing until settledKey matches.
    if (resetKey !== undefined && resetKey !== settledKey) {
        return value;
    }
    return debounced;
}
