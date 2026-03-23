# Dashboard Accordion Sections & Popout Toolbar

## Problem

The accordion in App.tsx has two UX issues:

1. **Sections are hard to scan.** All sections have identical white text titles
   with no visual differentiation. Users must read every title to find the
   section they want.

2. **Popout buttons are buried.** The "All Objects" and "Graph & Events" popout
   buttons are inside their parent accordion sections. Users must open those
   sections to access the popout buttons, defeating the purpose of having
   them be always-accessible aggregate views.

## Design

### 1. PopoutToolbar

A horizontal strip rendered above the accordion. Contains one ActionIcon per
popout panel. Hidden entirely when there are no popout panels.

- Each button shows a Tooltip on hover with the panel name
- Clicking opens the corresponding right-side Drawer (reuses PopoutPanel)
- The toolbar is visually compact -- just icons in a row

### 2. Section Component

A reusable wrapper that replaces the repeated Accordion.Item / Accordion.Control
/ Accordion.Panel / ErrorBoundary boilerplate. Each section gets:

- **Icon**: A lucide-react icon rendered before the title
- **Color**: A Mantine color applied to both the icon and title text
- **ErrorBoundary**: Wraps children automatically

Props:
```typescript
interface SectionProps {
    value: string;        // accordion item value
    title: string;        // display title
    icon: LucideIcon;     // lucide-react icon component
    color: string;        // mantine color (e.g. "blue", "teal.4")
    children: ReactNode;  // panel content
}
```

### 3. Color Groupings

Sections are colored by pipeline stage / functional area. Similar sections
share a color so users can visually scan by category.

| Color | Mantine token | Sections |
|-------|--------------|----------|
| Blue | blue | Pipeline Status, Game State, Log Messages |
| Teal | teal | Visual Perception, Aural Perception |
| Violet | violet | Visual Attention, Object Resolution |
| Orange | orange | Intrinsics & Significance, Transforms & Prediction, Actions |
| Yellow | yellow | Bookmarks, Inventory |

### 4. Icon Assignments

| Section | Icon | Rationale |
|---------|------|-----------|
| Pipeline Status | Activity | pipeline activity monitor |
| Bookmarks | Bookmark | standard bookmark icon |
| Game State | Gamepad2 | game context |
| Log Messages | MessageSquare | log/message output |
| Intrinsics & Significance | HeartPulse | agent vitals |
| Inventory | Backpack | items carried |
| Visual Perception | Eye | visual input |
| Aural Perception | Ear | audio input |
| Visual Attention | ScanEye | focused visual processing |
| Object Resolution | Shapes | object identification |
| Transforms & Prediction | GitCompare | change detection / diff |
| Actions | Zap | agent output action |

### 5. Popout Toolbar Buttons

| Button | Icon | Tooltip | Drawer Content |
|--------|------|---------|---------------|
| All Objects | Table | "All Objects" | AllObjects table |
| Graph & Events | BarChart3 | "Graph & Events" | GraphHistory + EventHistory |

## Implementation

### New Files

- `dashboard-ui/src/components/common/Section.tsx` -- reusable accordion section
- `dashboard-ui/src/components/common/PopoutToolbar.tsx` -- toolbar strip

### Modified Files

- `dashboard-ui/src/App.tsx` -- replace all Accordion.Item blocks with Section,
  move PopoutPanel instances to PopoutToolbar, remove ErrorBoundary imports
  (handled by Section)
- `dashboard-ui/src/components/common/PopoutPanel.tsx` -- remove the inline
  label+button Group; just expose the Drawer and a trigger API for the toolbar

### What Does NOT Change

- Panel components (AllObjects, GraphHistory, etc.) -- unchanged
- State management (DashboardContext) -- unchanged
- Accordion persistence (sessionStorage) -- unchanged
- PopoutPanel Drawer behavior (lockScroll, removeScrollProps) -- unchanged
