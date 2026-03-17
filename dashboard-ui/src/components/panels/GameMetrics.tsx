/** Game metrics panel -- HP, Score, Depth, etc. */

import type { StepData } from "../../types/step-data";
import { KVTable } from "../common/KVTable";

interface GameMetricsProps {
    data: StepData | undefined;
}

export function GameMetrics({ data }: GameMetricsProps) {
    return (
        <KVTable
            data={data?.game_metrics}
            emptyText="No metrics data"
            title="Metrics"
        />
    );
}
