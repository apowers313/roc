"""Unified dashboard server with game lifecycle management.

Serves the dashboard UI and API for browsing historical runs. Games can be
started and stopped via REST endpoints without restarting the server.
"""

from __future__ import annotations

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
    """Unified ROC dashboard server with game lifecycle management."""
    import socketio
    import uvicorn

    from roc.config import Config
    from roc.dashboard_cli import _mount_static_files, _resolve_ssl_certs
    from roc.reporting.api_server import app, sio

    cfg = Config.get()

    if data_dir is None:
        data_dir = Path(cfg.data_dir)
    if port is None:
        port = cfg.dashboard_port

    # Set module-level state for the API server (no StepBuffer -- historical only
    # until a game is started via the /api/game/* endpoints).
    import roc.reporting.api_server as srv
    from roc.reporting.data_store import DataStore

    srv._data_store = DataStore(data_dir=data_dir)

    ssl_certfile, ssl_keyfile = _resolve_ssl_certs(cfg)
    proto = "https" if ssl_certfile else "http"

    # Initialize game manager for /api/game/* endpoints
    from roc.game_manager import GameManager

    game_mgr = GameManager(
        data_dir=data_dir,
        on_state_change=srv._emit_game_state_changed,
        server_url=f"{proto}://localhost:{port}",
    )
    srv._game_manager = game_mgr

    # Mount the ASGI app (FastAPI + Socket.io)
    sio_app = socketio.ASGIApp(sio, other_asgi_app=app)
    _mount_static_files(app)

    click.echo(f"ROC server at {proto}://{host}:{port}")
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
