# mypy: disable-error-code="no-untyped-def"

"""Unit tests for graph archive export in roc/game/gymnasium.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np

from roc.game.gymnasium import _export_graph_archive


class TestExportGraphArchive:
    def test_export_graph_archive_writes_json(self, tmp_path: Path):
        """Verify graph.json is written to the run directory."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=1)
        G.add_node(2, labels="TakeAction", action_id=19)
        G.add_edge(1, 2, type="FrameAttribute")

        mock_store = MagicMock()
        mock_store.run_dir = tmp_path

        with (
            patch("roc.game.gymnasium.GraphDB.to_networkx", return_value=G),
            patch("roc.game.gymnasium.Observability.get_ducklake_store", return_value=mock_store),
        ):
            _export_graph_archive()

        assert (tmp_path / "graph.json").exists()

    def test_export_graph_archive_node_link_format(self, tmp_path: Path):
        """Output has 'directed', 'nodes', and 'links' keys."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=1)
        G.add_node(2, labels="Object", uuid=42)
        G.add_edge(1, 2, type="SituatedObjectInstance")

        mock_store = MagicMock()
        mock_store.run_dir = tmp_path

        with (
            patch("roc.game.gymnasium.GraphDB.to_networkx", return_value=G),
            patch("roc.game.gymnasium.Observability.get_ducklake_store", return_value=mock_store),
        ):
            _export_graph_archive()

        data = json.loads((tmp_path / "graph.json").read_text())
        assert data["directed"] is True
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 2
        assert len(data["links"]) == 1

    def test_export_graph_archive_skips_when_no_store(self):
        """Returns without error when Observability store is None."""
        with patch("roc.game.gymnasium.Observability.get_ducklake_store", return_value=None):
            # Should not raise
            _export_graph_archive()

    def test_export_graph_archive_handles_non_serializable_types(self, tmp_path: Path):
        """numpy types, sets, etc. are converted via default=str."""
        G = nx.DiGraph()
        G.add_node(1, labels="Frame", tick=np.int64(1), data=np.array([1, 2, 3]))
        G.add_node(2, labels="Object", tags={"a", "b"})
        G.add_edge(1, 2, type="Test", weight=np.float64(0.5))

        mock_store = MagicMock()
        mock_store.run_dir = tmp_path

        with (
            patch("roc.game.gymnasium.GraphDB.to_networkx", return_value=G),
            patch("roc.game.gymnasium.Observability.get_ducklake_store", return_value=mock_store),
        ):
            _export_graph_archive()

        # Should not raise, and file should be valid JSON
        data = json.loads((tmp_path / "graph.json").read_text())
        assert len(data["nodes"]) == 2

    def test_export_called_from_handle_game_over(self, mocker):
        """_export_graph_archive is called from _handle_game_over."""
        from roc.framework.config import Config
        from roc.game.gymnasium import _handle_game_over

        settings = Config.get()

        obs = {
            "tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
            "blstats": np.zeros(27, dtype=np.int64),
        }

        mock_export = mocker.patch("roc.game.gymnasium._export_graph_archive")
        mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mocker.patch("roc.game.gymnasium.GraphDB.export")
        mocker.patch("roc.game.gymnasium._emit_state_record")

        _handle_game_over(obs, game_num=1, done=True, settings=settings)

        mock_export.assert_called_once()
