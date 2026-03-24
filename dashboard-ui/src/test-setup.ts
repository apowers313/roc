import "@testing-library/jest-dom/vitest";

// Mantine's ScrollArea uses ResizeObserver which jsdom doesn't implement.
class ResizeObserverStub {
    observe() { /* stub for jsdom */ }
    unobserve() { /* stub for jsdom */ }
    disconnect() { /* stub for jsdom */ }
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;

// Mantine requires globalThis.matchMedia which jsdom doesn't implement.
Object.defineProperty(globalThis, "matchMedia", {
    writable: true,
    value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
    }),
});
