import { remoteLoggerPlugin } from "@graphty/remote-logger/vite";
import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "fs";
import { resolve } from "path";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

const SSL_CERT = "/home/apowers/ssl/atoms.crt";
const SSL_KEY = "/home/apowers/ssl/atoms.key";
const HOST = "dev.ato.ms";
const PORT = 9044;
const API_PORT = 9043;

const hasSSL = existsSync(SSL_KEY) && existsSync(SSL_CERT);
const apiTarget = hasSSL
    ? `https://localhost:${API_PORT}`
    : `http://localhost:${API_PORT}`;

export default defineConfig({
    plugins: [
        react(),
        remoteLoggerPlugin(),
        VitePWA({
            registerType: "autoUpdate",
            workbox: {
                // Cache app shell (JS, CSS, HTML) for instant reload
                globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
                // API and Socket.io requests go to network -- never cache data
                navigateFallback: "index.html",
                runtimeCaching: [
                    {
                        urlPattern: /^\/api\//,
                        handler: "NetworkOnly",
                    },
                    {
                        urlPattern: /^\/socket\.io\//,
                        handler: "NetworkOnly",
                    },
                ],
            },
            manifest: {
                name: "ROC Debug Dashboard",
                short_name: "ROC",
                description: "Debug dashboard for ROC reinforcement learning agent",
                theme_color: "#1a1b1e",
                background_color: "#1a1b1e",
                display: "standalone",
                start_url: "/",
                icons: [
                    {
                        src: "roc-icon-192.png",
                        sizes: "192x192",
                        type: "image/png",
                    },
                    {
                        src: "roc-icon-512.png",
                        sizes: "512x512",
                        type: "image/png",
                    },
                ],
            },
        }),
    ],
    resolve: {
        alias: {
            react: resolve(__dirname, "node_modules/react"),
            "react-dom": resolve(__dirname, "node_modules/react-dom"),
        },
    },
    server: {
        host: HOST,
        port: PORT,
        ...(hasSSL && {
            https: {
                key: readFileSync(SSL_KEY),
                cert: readFileSync(SSL_CERT),
            },
        }),
        proxy: {
            "/api": {
                target: apiTarget,
                changeOrigin: true,
                secure: false,
            },
            "/socket.io": {
                target: apiTarget,
                ws: true,
                secure: false,
            },
        },
    },
    build: {
        outDir: "dist",
    },
});
