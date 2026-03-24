/** Renders a Mermaid diagram definition to SVG in the browser.
 *
 * Client-side rendering ensures text is sized using the same font metrics
 * that will display it, avoiding the text clipping that occurs when SVGs
 * are pre-rendered on a different device.
 */

import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

// Initialize mermaid once -- no auto-start, we render on demand
mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    securityLevel: "loose",
});

let renderCounter = 0;

/**
 * Hidden container for mermaid's temporary render step.
 *
 * mermaid.render() briefly appends a full-size SVG to the DOM for text
 * measurement. On iOS Safari / PWA, this can expand the viewport width
 * (e.g. a 5900px-wide diagram) and iOS persists the zoom level across
 * refreshes. Rendering into a zero-size, overflow-hidden container
 * prevents the temporary SVG from affecting layout.
 */
function getOffscreenContainer(): HTMLDivElement {
    let el = document.getElementById("mermaid-offscreen") as HTMLDivElement | null;
    if (!el) {
        el = document.createElement("div");
        el.id = "mermaid-offscreen";
        el.style.cssText =
            "position:fixed;width:0;height:0;overflow:hidden;pointer-events:none;opacity:0;top:0;left:0";
        document.body.appendChild(el);
    }
    return el;
}

interface MermaidDiagramProps {
    /** Mermaid diagram definition text (e.g. classDiagram source). */
    definition: string;
    /** Called with the raw SVG string after rendering completes. */
    onSvgReady?: (svg: string) => void;
}

export function MermaidDiagram({ definition, onSvgReady }: Readonly<MermaidDiagramProps>) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!definition || !containerRef.current) return;

        let cancelled = false;
        const id = `mermaid-schema-${++renderCounter}`;

        (async () => {
            try {
                const offscreen = getOffscreenContainer();
                const { svg } = await mermaid.render(id, definition, offscreen);
                if (cancelled) return;
                if (containerRef.current) {
                    containerRef.current.innerHTML = svg;
                    onSvgReady?.(svg);
                }
                setError(null);
            } catch (e) {
                if (!cancelled) {
                    setError(e instanceof Error ? e.message : "Mermaid render failed");
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [definition, onSvgReady]);

    if (error) {
        return (
            <div style={{ color: "var(--mantine-color-red-6)", fontSize: 12 }}>
                Diagram render error: {error}
            </div>
        );
    }

    return <div ref={containerRef} />;
}
