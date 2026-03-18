"""Standalone dashboard viewer for browsing historical runs without starting a game."""

from __future__ import annotations

import os
from pathlib import Path

import click


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Data directory containing runs. Defaults to config data_dir.",
)
@click.option(
    "--port", type=int, default=None, help="Port to serve on. Defaults to config dashboard_port."
)
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to.")
def main(data_dir: Path | None, port: int | None, host: str) -> None:
    """Browse historical game runs in the debug dashboard."""
    # Disable observability before importing anything from roc.reporting
    # to prevent creating empty run directories on startup.
    os.environ["roc_emit_state"] = "false"

    import socketio
    import uvicorn

    from roc.config import Config
    from roc.reporting.api_server import app, sio

    cfg = Config.get()

    if data_dir is None:
        data_dir = Path(cfg.data_dir)
    if port is None:
        port = cfg.dashboard_port

    # Set module-level state for the API server (no StepBuffer needed for
    # historical-only mode).
    import roc.reporting.api_server as srv

    srv._data_dir = data_dir
    srv._live_run_name = None
    srv._step_buffer = None

    # Mount the ASGI app (FastAPI + Socket.io)
    sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

    # Try to mount React static build if it exists
    from fastapi.staticfiles import StaticFiles

    dist_dir = Path(__file__).parent.parent / "dashboard-ui" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    # Read SSL certs: Config.get() reads from .env if available
    ssl_certfile = cfg.ssl_certfile if cfg.ssl_certfile else None
    ssl_keyfile = cfg.ssl_keyfile if cfg.ssl_keyfile else None

    # Fall back to reading .env directly if Config didn't pick up SSL.
    # Check both CWD and the project root (parent of roc/ package).
    if not ssl_certfile:
        project_root = Path(__file__).parent.parent
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            env_path = project_root / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("roc_ssl_certfile="):
                    val = line.split("=", 1)[1].strip().strip('"')
                    if Path(val).exists():
                        ssl_certfile = val
                elif line.startswith("roc_ssl_keyfile="):
                    val = line.split("=", 1)[1].strip().strip('"')
                    if Path(val).exists():
                        ssl_keyfile = val

    proto = "https" if ssl_certfile else "http"
    click.echo(f"Dashboard at {proto}://{host}:{port}")
    click.echo(f"Data directory: {data_dir}")
    click.echo("Press Ctrl+C to stop.")

    config = uvicorn.Config(
        sio_app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
