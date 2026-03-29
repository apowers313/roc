import { vi } from "vitest";

/** Shared mock state -- use getter/setter functions to access. */
const _state = {
    capturedOnNewStep: undefined as ((data: unknown) => void) | undefined,
    mockLiveStatusValue: null as unknown,
};

/** Get the captured onNewStep callback. */
export function getCapturedOnNewStep() {
    return _state.capturedOnNewStep;
}

/** Get the current mock live status value. */
export function getMockLiveStatusValue() {
    return _state.mockLiveStatusValue;
}

/** Reset shared state between tests. */
export function resetLiveUpdatesMock() {
    _state.capturedOnNewStep = undefined;
    _state.mockLiveStatusValue = null;
}

/** Set the mock live status value for the next render. */
export function setMockLiveStatusValue(value: unknown) {
    _state.mockLiveStatusValue = value;
}

export const useLiveUpdates = vi.fn((opts?: { onNewStep?: (data: unknown) => void }) => {
    _state.capturedOnNewStep = opts?.onNewStep;
    return { connected: false, liveStatus: _state.mockLiveStatusValue };
});
