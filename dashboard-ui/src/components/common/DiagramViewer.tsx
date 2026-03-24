/** Zoomable, pannable diagram viewer with download and open-in-new-window. */

import { ActionIcon, Group, Text, Tooltip } from "@mantine/core";
import { Download, ExternalLink, Minus, Plus, RotateCcw } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import {
    TransformComponent,
    TransformWrapper,
    type ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch";

import { MermaidDiagram } from "./MermaidDiagram";

interface DiagramViewerProps {
    /** Mermaid diagram definition text. */
    definition: string;
    /** Filename stem used for downloads (without extension). */
    filename?: string;
}

/**
 * Button zoom step for smooth mode.
 * In smooth mode, zoomIn(step) applies: scale * e^step
 * 0.3 -> e^0.3 = 1.35x per click (moderate, ~5 clicks to 4.5x)
 */
const BUTTON_STEP = 0.3;

const MIN_SCALE = 0.05;
const MAX_SCALE = 20;

function svgToBlob(svgString: string): Blob {
    return new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
}

export function DiagramViewer({
    definition,
    filename = "schema",
}: DiagramViewerProps) {
    const transformRef = useRef<ReactZoomPanPinchRef | null>(null);
    const [svgString, setSvgString] = useState<string | null>(null);

    const handleSvgReady = useCallback((svg: string) => {
        setSvgString(svg);
    }, []);

    const handleDownload = useCallback(() => {
        if (!svgString) return;
        const blob = svgToBlob(svgString);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${filename}.svg`;
        a.click();
        URL.revokeObjectURL(url);
    }, [svgString, filename]);

    const handleOpenNewWindow = useCallback(() => {
        if (!svgString) return;
        const blob = svgToBlob(svgString);
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank");
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
    }, [svgString]);

    return (
        <div>
            <Group gap={4} mb={4}>
                <Text size="xs" fw={600}>
                    Diagram
                </Text>
                <div style={{ flex: 1 }} />
                <Tooltip label="Zoom in" withArrow>
                    <ActionIcon
                        size="xs"
                        variant="subtle"
                        onClick={() => transformRef.current?.zoomIn(BUTTON_STEP)}
                    >
                        <Plus size={12} />
                    </ActionIcon>
                </Tooltip>
                <Tooltip label="Zoom out" withArrow>
                    <ActionIcon
                        size="xs"
                        variant="subtle"
                        onClick={() => transformRef.current?.zoomOut(BUTTON_STEP)}
                    >
                        <Minus size={12} />
                    </ActionIcon>
                </Tooltip>
                <Tooltip label="Reset zoom" withArrow>
                    <ActionIcon
                        size="xs"
                        variant="subtle"
                        onClick={() => transformRef.current?.resetTransform()}
                    >
                        <RotateCcw size={12} />
                    </ActionIcon>
                </Tooltip>
                <Tooltip label="Download SVG" withArrow>
                    <ActionIcon
                        size="xs"
                        variant="subtle"
                        onClick={handleDownload}
                        disabled={!svgString}
                    >
                        <Download size={12} />
                    </ActionIcon>
                </Tooltip>
                <Tooltip label="Open in new window" withArrow>
                    <ActionIcon
                        size="xs"
                        variant="subtle"
                        onClick={handleOpenNewWindow}
                        disabled={!svgString}
                    >
                        <ExternalLink size={12} />
                    </ActionIcon>
                </Tooltip>
            </Group>
            <div
                style={{
                    border: "1px solid var(--mantine-color-dark-4)",
                    borderRadius: 4,
                    overflow: "hidden",
                    background: "white",
                    minHeight: 200,
                }}
            >
                <TransformWrapper
                    ref={transformRef}
                    initialScale={1}
                    minScale={MIN_SCALE}
                    maxScale={MAX_SCALE}
                    centerOnInit
                    limitToBounds={false}
                    smooth
                    wheel={{ smoothStep: 0.001 }}
                >
                    <TransformComponent
                        wrapperStyle={{ width: "100%", height: 400 }}
                    >
                        <MermaidDiagram
                            definition={definition}
                            onSvgReady={handleSvgReady}
                        />
                    </TransformComponent>
                </TransformWrapper>
            </div>
            <Text size="xs" c="dimmed" mt={2}>
                Scroll to zoom, drag to pan
            </Text>
        </div>
    );
}
