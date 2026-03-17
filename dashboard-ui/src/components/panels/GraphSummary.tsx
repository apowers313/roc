/** Graph DB summary panel. */

import type { StepData } from "../../types/step-data";
import { KVTable } from "../common/KVTable";

interface GraphSummaryProps {
    data: StepData | undefined;
}

export function GraphSummary({ data }: GraphSummaryProps) {
    return <KVTable data={data?.graph_summary} emptyText="No graph data" />;
}
