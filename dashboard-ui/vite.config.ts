import { remoteLoggerPlugin } from "@graphty/remote-logger/vite";
import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "fs";
import { resolve } from "path";
import { defineConfig } from "vite";

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
    plugins: [react(), remoteLoggerPlugin()],
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
