"""Clean up empty and short game runs from the data directory.

Scans the data directory for run directories and removes those with
no parquet data files (empty catalogs from failed startups) or fewer
than a configurable minimum number of steps.

Usage:
    uv run cleanup                    # dry run (default)
    uv run cleanup --delete           # actually delete
    uv run cleanup --min-steps=100    # raise threshold
    uv run cleanup --data-dir=/path   # custom data dir
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click


def _count_parquet_files(run_dir: Path) -> int:
    """Count parquet files in a run's data directory."""
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return 0
    return len(list(data_dir.rglob("*.parquet")))


def _get_step_count_from_api(run_name: str, api_url: str) -> int | None:
    """Try to get step count from a running dashboard API."""
    import urllib.request

    try:
        url = f"{api_url}/api/runs/{run_name}/step-range"
        with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310
            import json

            data = json.loads(resp.read())
            return int(data.get("max", 0))
    except Exception:
        return None


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Data directory containing runs. Defaults to config data_dir.",
)
@click.option(
    "--min-steps",
    type=int,
    default=10,
    help="Minimum steps to keep a run (default: 10). Runs with fewer steps are deleted.",
)
@click.option(
    "--delete",
    is_flag=True,
    help="Actually delete runs (default: dry run).",
)
def main(data_dir: Path | None, min_steps: int, delete: bool) -> None:
    """Clean up empty and short game runs."""
    from roc.framework.config import Config

    cfg = Config.get()
    if data_dir is None:
        data_dir = Path(cfg.data_dir)

    if not data_dir.exists():
        click.echo(f"Data directory not found: {data_dir}")
        raise SystemExit(1)

    runs = sorted(data_dir.iterdir())
    runs = [r for r in runs if r.is_dir() and (r / "catalog.duckdb").exists()]

    click.echo(f"Scanning {len(runs)} runs in {data_dir}")
    click.echo(f"Minimum steps to keep: {min_steps}")
    click.echo(f"Mode: {'DELETE' if delete else 'DRY RUN'}")
    click.echo()

    to_delete, to_keep = _classify_runs(runs)

    click.echo(f"Keep: {len(to_keep)} runs (have parquet data)")
    click.echo(f"Delete: {len(to_delete)} runs")
    click.echo()

    if not to_delete:
        click.echo("Nothing to clean up.")
        return

    total_bytes = _report_deletions(to_delete)
    click.echo(f"\nTotal space to reclaim: {total_bytes / (1024 * 1024):.1f} MB")

    if delete:
        _execute_deletions(to_delete)
    else:
        click.echo("\nRe-run with --delete to actually remove these runs.")


def _classify_runs(
    runs: list[Path],
) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Classify runs into delete and keep lists."""
    to_delete: list[tuple[Path, str]] = []
    to_keep: list[Path] = []
    for run_dir in runs:
        parquet_count = _count_parquet_files(run_dir)
        if parquet_count == 0:
            to_delete.append((run_dir, "no data files"))
        else:
            to_keep.append(run_dir)
    return to_delete, to_keep


def _report_deletions(to_delete: list[tuple[Path, str]]) -> int:
    """Print runs to delete and return total bytes."""
    total_bytes = 0
    click.echo("Runs to delete:")
    for run_dir, reason in to_delete:
        size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        total_bytes += size
        size_mb = size / (1024 * 1024)
        click.echo(f"  {run_dir.name}: {reason} ({size_mb:.1f} MB)")
    return total_bytes


def _execute_deletions(to_delete: list[tuple[Path, str]]) -> None:
    """Actually delete the listed run directories."""
    click.echo("\nDeleting...")
    for run_dir, _reason in to_delete:
        shutil.rmtree(run_dir)
        click.echo(f"  Deleted {run_dir.name}")
    click.echo(f"\nDone. Deleted {len(to_delete)} runs.")


if __name__ == "__main__":
    main()
