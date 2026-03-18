/** Keyboard shortcuts for dashboard navigation and bookmarks. */

import { useHotkeys } from "react-hotkeys-hook";

export interface KeyboardHandlers {
    stepForward: () => void;
    stepBack: () => void;
    togglePlay: () => void;
    jumpToStart: () => void;
    jumpToEnd: () => void;
    stepForward10: () => void;
    stepBack10: () => void;
    toggleHelp: () => void;
    toggleBookmark: () => void;
    nextBookmark: () => void;
    prevBookmark: () => void;
    goLive: () => void;
    speedUp: () => void;
    speedDown: () => void;
    cycleGame: () => void;
}

// Prevent default browser behavior (e.g. scrolling) for navigation keys.
const NAV_OPTS = { preventDefault: true } as const;

export function useKeyboardShortcuts(handlers: KeyboardHandlers): void {
    useHotkeys("right", handlers.stepForward, NAV_OPTS);
    useHotkeys("left", handlers.stepBack, NAV_OPTS);
    useHotkeys("space", handlers.togglePlay, NAV_OPTS);
    useHotkeys("home", handlers.jumpToStart, NAV_OPTS);
    useHotkeys("ctrl+left", handlers.jumpToStart, NAV_OPTS);
    useHotkeys("end", handlers.jumpToEnd, NAV_OPTS);
    useHotkeys("ctrl+right", handlers.jumpToEnd, NAV_OPTS);
    useHotkeys("shift+right", handlers.stepForward10, NAV_OPTS);
    useHotkeys("shift+left", handlers.stepBack10, NAV_OPTS);
    useHotkeys("shift+/", handlers.toggleHelp); // ? key
    useHotkeys("b", handlers.toggleBookmark);
    useHotkeys("]", handlers.nextBookmark);
    useHotkeys("[", handlers.prevBookmark);
    useHotkeys("l", handlers.goLive);
    useHotkeys("shift+=", handlers.speedUp); // + key
    useHotkeys("=", handlers.speedUp);
    useHotkeys("-", handlers.speedDown);
    useHotkeys("g", handlers.cycleGame);
}
