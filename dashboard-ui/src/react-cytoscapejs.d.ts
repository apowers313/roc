declare module "react-cytoscapejs" {
    import type cytoscape from "cytoscape";
    import type { CSSProperties, Component } from "react";

    interface CytoscapeComponentProps {
        elements: cytoscape.ElementDefinition[];
        stylesheet?: cytoscape.StylesheetStyle[] | cytoscape.StylesheetCSS[];
        layout?: cytoscape.LayoutOptions;
        cy?: (cy: cytoscape.Core) => void;
        style?: CSSProperties;
        wheelSensitivity?: number;
        userPanningEnabled?: boolean;
        userZoomingEnabled?: boolean;
        boxSelectionEnabled?: boolean;
        autoungrabify?: boolean;
        autounselectify?: boolean;
        pan?: cytoscape.Position;
        zoom?: number;
        minZoom?: number;
        maxZoom?: number;
    }

    export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {}
}
