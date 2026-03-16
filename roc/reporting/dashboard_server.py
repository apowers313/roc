"""Start/stop an in-process Panel debug dashboard with push-based updates."""

from __future__ import annotations

import threading
from typing import Any

import panel as pn

from roc.logger import logger

_stop_event = threading.Event()
_started = False


def start_dashboard() -> None:
    """Start the Panel dashboard in-process using ``pn.serve(threaded=True)``.

    Creates a ``StepBuffer`` and registers it globally so the game loop
    can push ``StepData`` directly.  Each browser session gets its own
    ``PanelDashboard`` instance (factory pattern) so Bokeh document
    ownership is never shared across sessions.

    Push notifications use ``asyncio_loop.call_soon_threadsafe`` to
    schedule UI updates on the Tornado thread -- no polling timers.
    """
    global _started
    if _started:
        return

    from roc.config import Config
    from roc.reporting.observability import Observability
    from roc.reporting.run_store import RunStore
    from roc.reporting.step_buffer import StepBuffer, register_step_buffer

    cfg = Config.get()
    if not cfg.dashboard_enabled:
        return

    store = Observability.get_ducklake_store()
    if store is None:
        logger.warning("Dashboard enabled but no DuckLakeStore available; skipping.")
        return

    # Create the step buffer and register it globally
    step_buffer = StepBuffer(capacity=2000)
    register_step_buffer(step_buffer)

    run_store = RunStore(store.run_dir)
    data_dir = run_store.run_dir.parent

    pn.extension("tabulator", inline=True)

    from roc.reporting.panel_debug import PanelDashboard

    # Factory function: called once per browser session.  Each session gets
    # its own PanelDashboard with its own Bokeh models, avoiding
    # "Models must be owned by only a single document" errors.
    def _create_session() -> pn.template.FastListTemplate:
        dashboard = PanelDashboard(
            RunStore(run_store.run_dir),
            data_dir=data_dir,
            step_buffer=step_buffer,
        )

        # Wire push notification: game thread -> doc.add_next_tick_callback -> Tornado.
        # The Bokeh document's add_next_tick_callback is thread-safe and ensures
        # widget changes are synced to the browser via the document protocol.
        # Deduplication is handled inside _on_new_data via _last_seen_step.
        doc = pn.state.curdoc
        session_id = id(dashboard)

        def _on_push() -> None:
            if doc is None:
                return
            try:
                doc.add_next_tick_callback(lambda: dashboard._on_new_data())
            except Exception:
                pass  # dead session, harmless

        step_buffer.add_listener(_on_push)
        logger.debug("Session {}: doc={}, listener registered", hex(session_id), doc)

        return pn.template.FastListTemplate(
            title="ROC Debug Dashboard",
            theme="dark",
            main=[dashboard],
        )

    serve_kwargs: dict[str, Any] = {
        "port": cfg.dashboard_port,
        "address": "0.0.0.0",
        "websocket_origin": "*",
        "show": False,
        "title": "ROC Debug Dashboard",
        "threaded": True,
    }
    if cfg.ssl_certfile and cfg.ssl_keyfile:
        serve_kwargs["ssl_certfile"] = cfg.ssl_certfile
        serve_kwargs["ssl_keyfile"] = cfg.ssl_keyfile

    _stop_event.clear()
    pn.serve(_create_session, **serve_kwargs)
    _started = True

    proto = "https" if cfg.ssl_certfile else "http"
    logger.info(f"Panel dashboard at {proto}://0.0.0.0:{cfg.dashboard_port}")


def stop_dashboard() -> None:
    """Stop the in-process dashboard and clean up the step buffer."""
    from roc.reporting.step_buffer import clear_step_buffer

    clear_step_buffer()
    _stop_event.set()
    logger.debug("Panel dashboard stopped.")
