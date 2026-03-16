"""Weights & Biases experiment tracking integration for ROC.

Provides a singleton WandbReporter that manages the lifecycle of a W&B run:
init, game boundaries, per-step logging, media logging, and finish with summary.
All methods are no-ops when wandb_enabled=False.

Per-tick data is buffered via ``log_step()`` and ``log_media()``, then flushed
as a single ``wandb.log()`` call by ``flush_step()``. This ensures that all
panels (metrics, screen, saliency map) share one global step counter so the
W&B step slider scrubs them in sync.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Lazy-loaded wandb module reference
_wandb: Any = None


def _ensure_wandb() -> Any:
    """Lazy-import wandb to avoid import overhead when disabled."""
    global _wandb
    if _wandb is None:
        import wandb

        _wandb = wandb
    return _wandb


class WandbReporter:
    """Singleton reporter for Weights & Biases experiment tracking.

    All public methods are no-ops when ``wandb_enabled=False`` in config.
    """

    _instance: WandbReporter | None = None
    _run: Any = None
    _enabled: bool = False
    _game_num: int = 0
    _game_tick: int = 0
    _global_step: int = 0
    _scores: list[int] = []
    _config: Any = None
    _game_table: Any = None
    _game_table_rows: list[dict[str, Any]] = []
    # Buffer for the current tick -- flushed by flush_step()
    _step_buffer: dict[str, Any] = {}

    @classmethod
    def init(cls, config: Any) -> None:
        """Initialize the W&B run.

        Args:
            config: ROC Config instance with wandb_* fields.
        """
        if not config.wandb_enabled:
            cls._enabled = False
            return

        wandb = _ensure_wandb()

        # Store config for toggle/interval checks
        cls._config = config

        # Build config dict from ROC config
        from roc.expmod import expmod_registry
        from roc.reporting.observability import instance_id, roc_version

        config_dict: dict[str, Any] = {
            "roc_version": roc_version,
        }

        # Add all config fields
        for key, value in config.__dict__.items():
            if not key.startswith("_"):
                try:
                    config_dict[key] = value
                except Exception:
                    pass

        # Add ExpMod params
        expmod_params: dict[str, Any] = {}
        for modtype, mods in expmod_registry.items():
            for name, instance in mods.items():
                expmod_params[f"{modtype}/{name}"] = instance.params_dict()
        config_dict["expmods"] = expmod_params

        # Init kwargs
        init_kwargs: dict[str, Any] = {
            "project": config.wandb_project,
            "config": config_dict,
            "tags": config.wandb_tags,
            "mode": config.wandb_mode,
            "name": instance_id,
            "settings": wandb.Settings(silent=True),
        }

        if config.wandb_entity:
            init_kwargs["entity"] = config.wandb_entity

        login_kwargs: dict[str, Any] = {}
        if config.wandb_host:
            login_kwargs["host"] = config.wandb_host
        if config.wandb_api_key:
            login_kwargs["key"] = config.wandb_api_key
        if login_kwargs:
            login_kwargs["relogin"] = True
            wandb.login(**login_kwargs)

        cls._run = wandb.init(**init_kwargs)
        cls._enabled = True
        cls._scores = []
        cls._game_num = 0
        cls._game_tick = 0
        cls._global_step = 0
        cls._game_table = None
        cls._game_table_rows = []
        cls._step_buffer = {}

        # If this is a sweep run, don't override the run name
        if cls._run and cls._run.sweep_id:
            logger.info(f"W&B sweep run detected (sweep_id={cls._run.sweep_id})")

        logger.info(f"W&B run initialized: {instance_id}")

    @classmethod
    def start_game(cls, game_num: int) -> None:
        """Signal the start of a new game.

        Buffers the game-start marker so it is emitted with the first tick's
        ``flush_step()`` call rather than as a separate ``wandb.log()``.

        Args:
            game_num: The game number (1-indexed).
        """
        if not cls._enabled:
            return

        cls._game_num = game_num
        cls._game_tick = 0
        cls._game_table = None
        cls._game_table_rows = []

        # Buffer -- will be emitted with the first tick's flush_step()
        cls._step_buffer["game_start"] = game_num

        logger.debug(f"W&B: game {game_num} started")

    @classmethod
    def end_game(cls, outcome: str = "unknown", final_score: int = 0) -> None:
        """Signal the end of a game.

        Buffers end-of-game data so it is emitted with the current tick's
        ``flush_step()`` call rather than as separate ``wandb.log()`` calls.
        Must be called *before* ``flush_step()`` on the final tick.

        Args:
            outcome: How the game ended (e.g. "died", "escaped").
            final_score: The final score for this game.
        """
        if not cls._enabled:
            return

        cls._scores.append(final_score)

        # Buffer end-of-game markers into the current tick
        cls._step_buffer["game_end"] = cls._game_num
        cls._step_buffer["outcome"] = outcome
        cls._step_buffer["final_score"] = final_score

        # Buffer accumulated game table into the current tick
        if cls._game_table_rows:
            wandb = _ensure_wandb()
            all_columns: set[str] = set()
            for row in cls._game_table_rows:
                all_columns.update(row.keys())
            columns = sorted(all_columns)
            table = wandb.Table(columns=columns)
            for row in cls._game_table_rows:
                table.add_data(*[row.get(c) for c in columns])
            cls._step_buffer[f"game_{cls._game_num}_steps"] = table

        logger.debug(f"W&B: game {cls._game_num} ended, score={final_score}")

    @classmethod
    def log_step(cls, data: dict[str, Any]) -> None:
        """Buffer per-step numeric metrics for the current tick.

        Call ``flush_step()`` after all data for this tick has been collected
        to emit a single ``wandb.log()`` call.

        Args:
            data: Dictionary of metric names to values.
        """
        if not cls._enabled:
            return

        cls._global_step += 1
        cls._game_tick += 1

        cls._step_buffer.update(data)
        cls._step_buffer["game_num"] = cls._game_num
        cls._step_buffer["game_tick"] = cls._game_tick

    @classmethod
    def log_media(cls, key: str, content: str) -> None:
        """Buffer HTML media content for the current tick.

        Respects ``wandb_log_interval``, ``wandb_log_screens``, and
        ``wandb_log_saliency`` config toggles.

        Args:
            key: The media key (e.g. "screen", "saliency_map").
            content: HTML string to log.
        """
        if not cls._enabled:
            return

        config = cls._config

        # Check per-key toggles
        if key == "screen" and not config.wandb_log_screens:
            return
        if key == "saliency_map" and not config.wandb_log_saliency:
            return

        # Check interval: log media when (game_tick - 1) % interval == 0
        interval = config.wandb_log_interval
        if (cls._game_tick - 1) % interval != 0:
            return

        wandb = _ensure_wandb()
        cls._step_buffer[key] = wandb.Html(content)

        # Accumulate row for game table
        table_row: dict[str, Any] = {
            "game_tick": cls._game_tick,
            "game_num": cls._game_num,
            key: content,
        }
        # Merge with existing row for this tick if present
        if cls._game_table_rows and cls._game_table_rows[-1]["game_tick"] == cls._game_tick:
            cls._game_table_rows[-1][key] = content
        else:
            cls._game_table_rows.append(table_row)

        # Mark that we have a table in progress
        cls._game_table = True

    @classmethod
    def flush_step(cls) -> None:
        """Flush the buffered step data as a single ``wandb.log()`` call.

        This should be called once per tick, after all ``log_step()`` and
        ``log_media()`` calls for that tick are complete. Ensures all panels
        share one global step counter.
        """
        if not cls._enabled or not cls._step_buffer:
            return

        wandb = _ensure_wandb()
        wandb.log(cls._step_buffer)
        cls._step_buffer = {}

    @classmethod
    def finish(cls) -> None:
        """Set summary metrics and finish the W&B run."""
        if not cls._enabled or cls._run is None:
            return

        # Set summary metrics
        total_games = len(cls._scores)
        cls._run.summary["total_games"] = total_games
        if total_games > 0:
            cls._run.summary["mean_score"] = sum(cls._scores) / total_games
            cls._run.summary["max_score"] = max(cls._scores)
        else:
            cls._run.summary["mean_score"] = 0
            cls._run.summary["max_score"] = 0

        cls._run.finish()
        logger.info("W&B run finished")
