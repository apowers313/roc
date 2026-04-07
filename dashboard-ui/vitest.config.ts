import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: ["./src/test-setup.ts"],
        exclude: ["e2e/**", "node_modules/**"],
        pool: "threads",
        poolOptions: {
            threads: {
                maxThreads: 8,
            },
        },
        coverage: {
            provider: "v8",
            reporter: ["text", "lcov"],
            reportsDirectory: "./coverage",
            reportOnFailure: true,
            include: ["src/**/*.{ts,tsx}"],
            exclude: [
                "src/main.tsx",
                "src/test-setup.ts",
                "src/test-utils.tsx",
                "src/**/*.test.*",
                "src/types/**",
                "src/prototype/**",
            ],
            thresholds: {
                statements: 80,
                branches: 80,
                functions: 80,
                lines: 80,
            },
        },
    },
});
