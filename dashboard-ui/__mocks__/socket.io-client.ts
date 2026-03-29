import { vi } from "vitest";

const mockSocket = {
    on: vi.fn(),
    disconnect: vi.fn(),
};

export const io = vi.fn(() => mockSocket);
