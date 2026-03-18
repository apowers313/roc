/** Panel-level error boundary -- catches render errors and shows a fallback. */

import { Text } from "@mantine/core";
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, info: ErrorInfo): void {
        console.error("Panel render error:", error, info.componentStack);
    }

    render(): ReactNode {
        if (this.state.hasError) {
            if (this.props.fallback) return this.props.fallback;
            return (
                <Text size="xs" c="red">
                    Render error: {this.state.error?.message ?? "unknown"}
                </Text>
            );
        }
        return this.props.children;
    }
}
