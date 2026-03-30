"""Standalone dashboard viewer for browsing historical runs without starting a game."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to.")  # nosec B104
def main(data_dir: Path | None, port: int | None, host: str) -> None:
    """Browse historical game runs in the debug dashboard."""
    # No need to set roc_emit_state=false -- Observability.init() no longer
    # creates DuckLake stores at module-level (only roc.init(enable_parquet=True) does).

    import socketio
    import uvicorn

    from roc.framework.config import Config
    from roc.reporting.api_server import app, sio

    cfg = Config.get()

    if data_dir is None:
        data_dir = Path(cfg.data_dir)
    if port is None:
        port = cfg.dashboard_port

    # Set module-level state for the API server (no StepBuffer needed for
    # historical-only mode).
    import roc.reporting.api_server as srv
    from roc.reporting.data_store import DataStore

    srv._data_store = DataStore(data_dir=data_dir)

    # Mount the ASGI app (FastAPI + Socket.io)
    sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

    # Try to mount React static build if it exists
    _mount_static_files(app)

    ssl_certfile, ssl_keyfile = _resolve_ssl_certs(cfg)

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


def _mount_static_files(app: Any) -> None:
    """Mount the React static build directory if it exists."""
    from fastapi.staticfiles import StaticFiles

    dist_dir = Path(__file__).parent.parent / "dashboard-ui" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")


def _resolve_ssl_certs(cfg: Any) -> tuple[str | None, str | None]:
    """Resolve SSL cert/key paths from config or .env fallback."""
    ssl_certfile = cfg.ssl_certfile if cfg.ssl_certfile else None
    ssl_keyfile = cfg.ssl_keyfile if cfg.ssl_keyfile else None
    if ssl_certfile:
        return ssl_certfile, ssl_keyfile
    return _read_ssl_from_env(ssl_keyfile)


def _read_ssl_from_env(
    ssl_keyfile: str | None,
) -> tuple[str | None, str | None]:
    """Fall back to reading SSL paths from .env file."""
    ssl_certfile: str | None = None
    env_path = _find_env_file()
    if env_path is None:
        return None, ssl_keyfile
    for line in env_path.read_text().splitlines():
        cert, key = _parse_ssl_env_line(line.strip(), ssl_keyfile)
        if cert is not None:
            ssl_certfile = cert
        if key is not None:
            ssl_keyfile = key
    return ssl_certfile, ssl_keyfile


def _parse_ssl_env_line(line: str, _current_keyfile: str | None) -> tuple[str | None, str | None]:
    """Parse a single .env line for SSL cert/key paths."""
    if line.startswith("roc_ssl_certfile="):
        val = line.split("=", 1)[1].strip().strip('"')
        if Path(val).exists():
            return val, None
    elif line.startswith("roc_ssl_keyfile="):
        val = line.split("=", 1)[1].strip().strip('"')
        if Path(val).exists():
            return None, val
    return None, None


def _find_env_file() -> Path | None:
    """Find the .env file in CWD or project root."""
    project_root = Path(__file__).parent.parent
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        return env_path
    env_path = project_root / ".env"
    if env_path.exists():
        return env_path
    return None


if __name__ == "__main__":
    main()
