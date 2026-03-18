import { defineConfig } from "@playwright/test";

export default defineConfig({
    testDir: "./e2e",
    timeout: 60_000,
    retries: 0,
    use: {
        headless: true,
        viewport: { width: 1280, height: 800 },
        ignoreHTTPSErrors: true,
    },
    reporter: [["list"]],
});
