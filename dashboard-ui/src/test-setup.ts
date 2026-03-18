import "@testing-library/jest-dom/vitest";

// Mantine's ScrollArea uses ResizeObserver which jsdom doesn't implement.
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
window.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;

// Mantine requires window.matchMedia which jsdom doesn't implement.
Object.defineProperty(window, "matchMedia", {
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
