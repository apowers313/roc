import { vi } from "vitest";

export const useStepData = vi.fn(() => ({
    data: undefined,
    isLoading: false,
    isPlaceholderData: false,
}));
export const useRuns = vi.fn(() => ({ data: undefined }));
export const useGames = vi.fn(() => ({ data: undefined }));
export const useStepRange = vi.fn(() => ({ data: undefined }));
export const useResolutionHistory = vi.fn(() => ({ data: undefined }));
export const useAllObjects = vi.fn(() => ({ data: undefined }));
export const useObjectHistory = vi.fn(() => ({ data: undefined, isLoading: false }));
export const useIntrinsicsHistory = vi.fn(() => ({ data: undefined }));
export const useMetricsHistory = vi.fn(() => ({ data: undefined }));
export const useGraphHistory = vi.fn(() => ({ data: undefined }));
export const useEventHistory = vi.fn(() => ({ data: undefined }));
export const useActionHistory = vi.fn(() => ({ data: undefined }));
