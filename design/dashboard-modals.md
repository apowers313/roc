# Dashboard Popout Panels Design

## Problem

Aggregate data panels (e.g. "All Objects", "Graph & Events") contain cross-game
summary visualizations. These are useful to keep visible while navigating
step-by-step through the main accordion, but currently they compete for vertical
scroll space with per-step panels. Users want to "pop out" these panels so they
remain visible alongside the main content.

The popped-out panels must stay interactive: clicking a data point in a chart or
a row in the All Objects table must still navigate the main dashboard to that
step.

## Design Patterns Considered

### 1. Mantine Drawer (slide-in side panel)

A `Drawer` slides in from a screen edge and overlays the main content.

- **Pros**: Native Mantine component, shares React context (DashboardContext
  works unchanged), zero new deps.
- **Cons**: Covers part of the main view; only one edge at a time; resizing is
  limited to predefined size values.

### 2. Floating / draggable panels

Freely positioned panels using `react-rnd` or similar.

- **Pros**: Maximum flexibility, true side-by-side viewing.
- **Cons**: Needs a third-party library, z-index management, position
  persistence, overlap issues.

### 3. Detached browser windows (`window.open`)

True OS-level windows with `BroadcastChannel` for state sync.

- **Pros**: Multi-monitor support, no viewport competition.
- **Cons**: Significant complexity (separate routes, serialized state sync,
  popup blockers, no shared React context).

### 4. Mantine Modal (dialog overlay)

Centered overlay with optional backdrop.

- **Pros**: Already used in the codebase (KeyboardHelp), zero new deps, shares
  React context.
- **Cons**: Blocks main content by default. Can be made non-blocking with
  `withOverlay={false}` and `closeOnClickOutside={false}`.

### 5. Popover (anchored flyout)

Small floating panel anchored to a trigger element.

- **Pros**: Good for small quick-glance content.
- **Cons**: Too small for charts/tables, dismisses easily, poor fit for rich
  interactive content.

## Decision: Drawer (non-blocking, right-side)

Use Mantine's `Drawer` component configured as a non-blocking side panel:

```tsx
<Drawer
  opened={opened}
  onClose={close}
  position="right"
  size="xl"
  withOverlay={false}
  closeOnClickOutside={false}
/>
```

### Rationale

1. **Zero new dependencies** -- Mantine `Drawer` and `useDisclosure` are
   already available.
2. **Zero state management changes** -- the Drawer renders inside the React
   tree, so `DashboardContext` and `ClickableChart`'s `onStepClick` work
   identically to how they work in the accordion today.
3. **Non-blocking** -- `withOverlay={false}` + `closeOnClickOutside={false}`
   lets the user interact with the main accordion while the Drawer is open.
4. **Simple implementation** -- a reusable `PopoutPanel` wrapper component
   (~30 lines) encapsulates Drawer + trigger button. Each panel just wraps
   its existing content.

### Upgrade Path

If true drag-and-drop positioning is later needed, the panel content and state
wiring stays the same -- only the container wrapper changes from Drawer to a
draggable div (e.g. `react-rnd`).

## Implementation

### New Component: `PopoutPanel`

A reusable wrapper that renders a trigger button in the accordion header and
a right-side Drawer containing the panel content.

```
dashboard-ui/src/components/common/PopoutPanel.tsx
```

Props:
- `title: string` -- Drawer title
- `children: ReactNode` -- panel content
- `size?: string` -- Drawer width (default "xl")

Uses `useDisclosure` from `@mantine/hooks` for open/close state.

### Modified: `App.tsx`

The "All Objects" and "Graph & Events" accordion items are replaced with
buttons that open their respective Drawers. The accordion items are removed
and the panels render inside `PopoutPanel` Drawers instead.

The panels' `onStepClick` handlers remain unchanged -- they call
`handleChartStepClick` which updates DashboardContext, and the Drawer
content re-renders automatically since it shares the same React tree.

### Panels Converted

| Panel | Content | Interactive? |
|-------|---------|-------------|
| All Objects | Sortable object table | Row click navigates to step |
| Graph & Events | GraphHistory + EventHistory charts | Chart click navigates to step |

### What Does NOT Change

- Panel component code (AllObjects, GraphHistory, EventHistory) -- unchanged
- State management (DashboardContext) -- unchanged
- Data fetching (TanStack Query hooks) -- unchanged
- ClickableChart wrapper -- unchanged
