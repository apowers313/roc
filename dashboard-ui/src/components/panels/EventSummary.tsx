/** Event summary panel -- per-step event bus activity counts. */

import type { StepData } from "../../types/step-data";
import { KVTable } from "../common/KVTable";

interface EventSummaryProps {
    data: StepData | undefined;
}

export function EventSummary({ data }: EventSummaryProps) {
    // event_summary is a list of dicts; the first entry is typically the
    // bus-name -> count map emitted by roc.event.summary.
    const summary = data?.event_summary?.[0] ?? null;
    return <KVTable data={summary} emptyText="No event data" title="Events" />;
}
