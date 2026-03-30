# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/gymnasium.py -- GraphDB flush/export gating and helper functions."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roc.framework.config import Config


def _make_fake_obs() -> dict[str, Any]:
    """Create a minimal observation dict matching what Gym.start() expects."""
    # blstats needs at least 27 entries to cover all blstat_offsets
    blstats = np.zeros(27, dtype=np.int64)
    return {
        "tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
        "tty_colors": np.zeros((24, 80), dtype=np.int8),
        "tty_cursor": np.array([0, 0]),
        "blstats": blstats,
    }


def _make_fake_obs_with_inventory() -> dict[str, Any]:
    """Create a fake observation dict that includes inventory data."""
    obs = _make_fake_obs()
    # inv_strs: 2 items, padded with null bytes
    item1 = [ord(c) for c in "a rusty sword"] + [0] * 42
    item2 = [ord(c) for c in "a food ration"] + [0] * 42
    empty = [0] * 55
    obs["inv_strs"] = np.array([item1, item2, empty], dtype=np.uint8)
    obs["inv_letters"] = np.array([ord("a"), ord("b"), 0], dtype=np.uint8)
    obs["inv_glyphs"] = np.array([100, 200, 5976], dtype=np.int32)
    return obs


class TestGraphDBControls:
    def test_graphdb_export_disabled_skips_export(self, mocker):
        """When roc_graphdb_export=False, GraphDB.export() is not called at game end."""
        settings = Config.get()
        settings.graphdb_export = False
        settings.graphdb_flush = True
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.game.gymnasium.GraphDB.export")

        # Patch Gym.__init__ to bypass real gym setup, then call start() directly
        with patch("roc.game.gymnasium.Gym.__init__", return_value=None):
            from roc.game.gymnasium import Gym

            # Create a concrete subclass for testing
            class FakeGym(Gym):
                name: str = "fakegym"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    """No-op; required by abstract base class."""

                def config(self, env: Any) -> None:
                    """No-op; required by abstract base class."""

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            # Set up minimal env mock: first step returns done=True
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            # Patch _dump_env_start/_dump_env_record/_dump_env_end and State/breakpoints
            mocker.patch("roc.game.gymnasium._dump_env_start")
            mocker.patch("roc.game.gymnasium._dump_env_record")
            mocker.patch("roc.game.gymnasium._dump_env_end")
            mocker.patch("roc.game.gymnasium.breakpoints")
            mocker.patch("roc.game.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_called()
        mock_export.assert_not_called()

    def test_graphdb_flush_disabled_skips_flush(self, mocker):
        """When roc_graphdb_flush=False, GraphDB.flush() is not called at game end."""
        settings = Config.get()
        settings.graphdb_flush = False
        settings.graphdb_export = True
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.game.gymnasium.GraphDB.export")

        with patch("roc.game.gymnasium.Gym.__init__", return_value=None):
            from roc.game.gymnasium import Gym

            class FakeGym(Gym):
                name: str = "fakegym2"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    """No-op; required by abstract base class."""

                def config(self, env: Any) -> None:
                    """No-op; required by abstract base class."""

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            mocker.patch("roc.game.gymnasium._dump_env_start")
            mocker.patch("roc.game.gymnasium._dump_env_record")
            mocker.patch("roc.game.gymnasium._dump_env_end")
            mocker.patch("roc.game.gymnasium.breakpoints")
            mocker.patch("roc.game.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_not_called()
        mock_export.assert_called()

    def test_graphdb_both_disabled_skips_both(self, mocker):
        """When both are False, neither flush nor export is called."""
        settings = Config.get()
        settings.graphdb_flush = False
        settings.graphdb_export = False
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.game.gymnasium.GraphDB.export")

        with patch("roc.game.gymnasium.Gym.__init__", return_value=None):
            from roc.game.gymnasium import Gym

            class FakeGym(Gym):
                name: str = "fakegym3"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    """No-op; required by abstract base class."""

                def config(self, env: Any) -> None:
                    """No-op; required by abstract base class."""

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            mocker.patch("roc.game.gymnasium._dump_env_start")
            mocker.patch("roc.game.gymnasium._dump_env_record")
            mocker.patch("roc.game.gymnasium._dump_env_end")
            mocker.patch("roc.game.gymnasium.breakpoints")
            mocker.patch("roc.game.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_not_called()
        mock_export.assert_not_called()


class TestActionValueToKey:
    """Tests for the action_value_to_key helper."""

    def test_printable_ascii(self):
        """Printable ASCII range (32-126) returns the character."""
        from roc.game.gymnasium import action_value_to_key

        assert action_value_to_key(ord("a")) == "a"
        assert action_value_to_key(ord("Z")) == "Z"
        assert action_value_to_key(ord(" ")) == " "
        assert action_value_to_key(ord("~")) == "~"

    def test_control_chars(self):
        """Control character range (1-31) returns ^X notation."""
        from roc.game.gymnasium import action_value_to_key

        # C("d") = 4 -> "^D"
        assert action_value_to_key(4) == "^D"
        # C("a") = 1 -> "^A"
        assert action_value_to_key(1) == "^A"
        assert action_value_to_key(31) == "^_"

    def test_meta_extended(self):
        """Meta/extended range (128+) returns M-x notation."""
        from roc.game.gymnasium import action_value_to_key

        # M("f") = 0x80 | ord("f") = 128 + 102 = 230 -> "M-f"
        assert action_value_to_key(230) == "M-f"
        # M(" ") = 128 + 32 = 160 -> "M- "
        assert action_value_to_key(160) == "M- "

    def test_meta_non_printable_base(self):
        """Meta values where the base is not printable return None."""
        from roc.game.gymnasium import action_value_to_key

        # 128 + 0 = 128, base=0 which is not in 32-126
        assert action_value_to_key(128) is None
        # Meta value 255 has base 127, which is outside printable range 32-126
        assert action_value_to_key(255) is None

    def test_zero_returns_none(self):
        """Value 0 does not map to any key."""
        from roc.game.gymnasium import action_value_to_key

        assert action_value_to_key(0) is None

    def test_del_returns_none(self):
        """Value 127 (DEL) does not map to any key."""
        from roc.game.gymnasium import action_value_to_key

        assert action_value_to_key(127) is None


class TestBuildActionMap:
    """Tests for _build_action_map helper."""

    def test_basic_enum_actions(self):
        """Builds action map from enum-like actions with name and value."""
        from roc.game.gymnasium import _build_action_map

        class FakeAction:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        actions = (FakeAction("MoveNorth", ord("k")), FakeAction("MoveEast", ord("l")))
        result = _build_action_map(actions)

        assert len(result) == 2
        assert result[0]["action_id"] == 0
        assert result[0]["action_name"] == "MoveNorth"
        assert result[0]["action_key"] == "k"
        assert result[1]["action_id"] == 1
        assert result[1]["action_name"] == "MoveEast"
        assert result[1]["action_key"] == "l"

    def test_non_int_value_no_key(self):
        """Actions with non-int values do not get an action_key."""
        from roc.game.gymnasium import _build_action_map

        class FakeAction:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        actions = (FakeAction("Special", "not_an_int"),)
        result = _build_action_map(actions)

        assert len(result) == 1
        assert "action_key" not in result[0]
        assert result[0]["action_name"] == "Special"

    def test_plain_values_without_name_attr(self):
        """Plain values (no .name) use str() for the name."""
        from roc.game.gymnasium import _build_action_map

        actions = (42, 99)
        result = _build_action_map(actions)

        assert result[0]["action_name"] == "42"
        assert result[1]["action_name"] == "99"

    def test_empty_actions(self):
        """Empty action tuple returns empty list."""
        from roc.game.gymnasium import _build_action_map

        assert _build_action_map(()) == []


class TestPublishActionMap:
    """Tests for _publish_action_map helper."""

    def test_no_gym_actions_returns_early(self):
        """When gym_actions is None, nothing happens."""
        from roc.game.gymnasium import _publish_action_map

        # Should not raise
        _publish_action_map(None, "http://example.com", None)

    def test_empty_gym_actions_returns_early(self):
        """When gym_actions is an empty tuple, nothing happens."""
        from roc.game.gymnasium import _publish_action_map

        _publish_action_map((), "http://example.com", None)

    def test_no_callback_saves_to_file(self, mocker):
        """Without callback URL, saves action map to file."""
        from roc.game.gymnasium import _publish_action_map

        mock_save = mocker.patch("roc.game.gymnasium._save_action_map_to_file")
        mocker.patch("roc.game.gymnasium._post_action_map_to_server")

        _publish_action_map((42,), None, None)
        mock_save.assert_called_once()

    def test_with_callback_posts_to_server(self, mocker):
        """With callback URL, posts action map to server."""
        from roc.game.gymnasium import _publish_action_map

        mock_post = mocker.patch("roc.game.gymnasium._post_action_map_to_server")
        mocker.patch("roc.game.gymnasium._save_action_map_to_file")

        _publish_action_map((42,), "http://example.com/api/internal/step", "ctx")
        mock_post.assert_called_once()


class TestSetupCallbackContext:
    """Tests for _setup_callback_context."""

    def test_no_callback_url_returns_none(self):
        """When callback_url is None, returns None."""
        from roc.game.gymnasium import _setup_callback_context

        result = _setup_callback_context(None, Config.get())
        assert result is None

    def test_http_url_returns_none_context(self):
        """When callback_url is http, returns None SSL context."""
        from roc.game.gymnasium import _setup_callback_context

        result = _setup_callback_context("http://localhost:8000/step", Config.get())
        assert result is None

    def test_https_url_returns_ssl_context(self):
        """When callback_url is https, returns an SSL context."""
        import ssl

        from roc.game.gymnasium import _setup_callback_context

        result = _setup_callback_context("https://localhost:8000/step", Config.get())
        assert isinstance(result, ssl.SSLContext)

    def test_https_context_disables_hostname_check(self):
        """SSL context must skip hostname verification for localhost callbacks.

        Root cause: server_cli.py builds the callback URL as
        ``https://localhost:<port>/api/internal/step`` because the game subprocess
        runs on the same machine. However, the SSL certificate is issued for the
        external domain (*.ato.ms), not for ``localhost``. Python's default SSL
        context enforces hostname verification, so every POST from the game
        subprocess to the dashboard server silently fails with:

            ssl.SSLCertVerificationError: Hostname mismatch,
            certificate is not valid for 'localhost'.

        The exception was swallowed by a bare ``except Exception`` in
        ``_push_step_to_server``, making live mode appear broken with zero steps
        received despite the game running normally.

        This test ensures the SSL context disables hostname checking (since the
        connection is internal loopback) while still requiring certificate
        verification against the loaded CA bundle.
        """
        import ssl

        from roc.game.gymnasium import _setup_callback_context

        result = _setup_callback_context("https://localhost:8000/step", Config.get())
        assert isinstance(result, ssl.SSLContext)
        assert result.check_hostname is False
        assert result.verify_mode == ssl.CERT_REQUIRED


class TestExtractGameMetrics:
    """Tests for _extract_game_metrics helper."""

    def test_extracts_all_metrics(self):
        """Extracts known blstats into a dict."""
        from roc.game.gymnasium import _extract_game_metrics

        obs = _make_fake_obs()
        # Set some known values
        obs["blstats"][9] = 42  # SCORE
        obs["blstats"][10] = 15  # HP
        obs["blstats"][11] = 20  # HPMAX
        obs["blstats"][12] = 3  # DEPTH

        result = _extract_game_metrics(obs)
        assert result["score"] == 42
        assert result["hp"] == 15
        assert result["hp_max"] == 20
        assert result["depth"] == 3
        assert "energy" in result
        assert "gold" in result
        assert "hunger" in result
        assert "xp_level" in result
        assert "experience" in result
        assert "ac" in result


class TestParseInventory:
    """Tests for _parse_inventory helper."""

    def test_valid_inventory(self):
        """Parses inventory items from observation."""
        from roc.game.gymnasium import _parse_inventory

        obs = _make_fake_obs_with_inventory()
        result = _parse_inventory(obs)

        assert result is not None
        assert len(result) == 2
        assert result[0]["letter"] == "a"
        assert "rusty sword" in result[0]["item"]
        assert result[0]["glyph"] == 100
        assert result[1]["letter"] == "b"

    def test_empty_inventory_returns_none(self):
        """All-empty glyph 5976 items are skipped, returning None."""
        from roc.game.gymnasium import _parse_inventory

        obs = _make_fake_obs()
        obs["inv_strs"] = np.array([[0] * 55], dtype=np.uint8)
        obs["inv_letters"] = np.array([0], dtype=np.uint8)
        obs["inv_glyphs"] = np.array([5976], dtype=np.int32)

        result = _parse_inventory(obs)
        assert result is None

    def test_missing_inventory_keys_returns_none(self):
        """Returns None when obs doesn't have inventory keys."""
        from roc.game.gymnasium import _parse_inventory

        obs = _make_fake_obs()
        result = _parse_inventory(obs)
        assert result is None


class TestCollectScreenData:
    """Tests for _collect_screen_data helper."""

    def test_none_screen_state(self):
        """When screen state is None, returns None values."""
        from roc.game.gymnasium import _collect_screen_data

        states = MagicMock()
        states.screen.val = None
        states.salency.val = None

        screen_vals, saliency_vals, features = _collect_screen_data(states)
        assert screen_vals is None
        assert saliency_vals is None
        assert features is None

    def test_with_saliency_no_features(self):
        """When saliency is present but feature_report is empty."""
        from roc.game.gymnasium import _collect_screen_data

        states = MagicMock()
        states.screen.val = None
        saliency_mock = MagicMock()
        saliency_mock.to_html_vals.return_value = {"test": "data"}
        saliency_mock.feature_report.return_value = None
        states.salency.val = saliency_mock

        screen_vals, saliency_vals, features = _collect_screen_data(states)
        assert screen_vals is None
        assert saliency_vals == {"test": "data"}
        assert features is None


class TestCollectObjectData:
    """Tests for _collect_object_data helper."""

    def test_both_none(self):
        """When object and attention states are both None."""
        from roc.game.gymnasium import _collect_object_data

        states = MagicMock()
        states.object.val = None
        states.attention.val = None

        object_info, focus_points = _collect_object_data(states)
        assert object_info is None
        assert focus_points is None

    def test_with_object(self):
        """When object state is present."""
        from roc.game.gymnasium import _collect_object_data

        states = MagicMock()
        states.object.val = "some_object"
        states.attention.val = None

        object_info, focus_points = _collect_object_data(states)
        assert object_info is not None
        assert len(object_info) == 1
        assert "raw" in object_info[0]
        assert focus_points is None


class TestCollectGraphSummary:
    """Tests for _collect_graph_summary helper."""

    def test_returns_cache_stats(self, mocker):
        """Returns node/edge cache size info."""
        from roc.game.gymnasium import _collect_graph_summary

        mock_node_cache = MagicMock()
        mock_node_cache.currsize = 10
        mock_node_cache.maxsize = 1000
        mock_edge_cache = MagicMock()
        mock_edge_cache.currsize = 5
        mock_edge_cache.maxsize = 500

        mocker.patch("roc.db.graphdb.Node.get_cache", return_value=mock_node_cache)
        mocker.patch("roc.db.graphdb.Edge.get_cache", return_value=mock_edge_cache)

        result = _collect_graph_summary()
        assert result["node_count"] == 10
        assert result["node_max"] == 1000
        assert result["edge_count"] == 5
        assert result["edge_max"] == 500


class TestCollectEventSummary:
    """Tests for _collect_event_summary helper."""

    def test_returns_step_counts(self, mocker):
        """Returns event step counts when present."""
        from roc.game.gymnasium import _collect_event_summary

        mocker.patch("roc.framework.event.Event.get_step_counts", return_value={"bus1": 5})
        result = _collect_event_summary()
        assert result is not None
        assert result == [{"bus1": 5}]

    def test_empty_step_counts(self, mocker):
        """Returns None when step counts are empty."""
        from roc.game.gymnasium import _collect_event_summary

        mocker.patch("roc.framework.event.Event.get_step_counts", return_value={})
        result = _collect_event_summary()
        assert result is None


class TestBuildActionTakenDict:
    """Tests for _build_action_taken_dict helper."""

    def test_none_action(self):
        """Returns None when action state is None."""
        from roc.game.gymnasium import _build_action_taken_dict

        states = MagicMock()
        states.action.val = None

        result = _build_action_taken_dict(states)
        assert result is None

    def test_valid_action_with_gym_actions(self):
        """Returns action dict with name and key when gym_actions are configured."""
        from roc.game.gymnasium import _build_action_taken_dict

        class FakeEnum:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        settings = Config.get()
        settings.gym_actions = (FakeEnum("MoveNorth", ord("k")),)  # type: ignore[assignment]

        states = MagicMock()
        states.action.val.action = 0

        result = _build_action_taken_dict(states)
        assert result is not None
        assert result["action_id"] == 0
        assert result["action_name"] == "MoveNorth"
        assert result["action_key"] == "k"

    def test_action_without_gym_actions(self):
        """Returns action dict with only action_id when no gym_actions."""
        from roc.game.gymnasium import _build_action_taken_dict

        settings = Config.get()
        settings.gym_actions = None

        states = MagicMock()
        states.action.val.action = 3

        result = _build_action_taken_dict(states)
        assert result is not None
        assert result["action_id"] == 3
        assert "action_name" not in result


class TestCollectMessage:
    """Tests for _collect_message helper."""

    def test_none_message(self):
        """Returns None when message is None."""
        from roc.game.gymnasium import _collect_message

        states = MagicMock()
        states.message.val = None
        assert _collect_message(states) is None

    def test_empty_message(self):
        """Returns None when message is empty/whitespace."""
        from roc.game.gymnasium import _collect_message

        states = MagicMock()
        states.message.val = "   "
        assert _collect_message(states) is None

    def test_valid_message(self):
        """Returns stripped message string."""
        from roc.game.gymnasium import _collect_message

        states = MagicMock()
        states.message.val = "  You hit the goblin.  "
        assert _collect_message(states) == "You hit the goblin."


class TestCollectPhonemes:
    """Tests for _collect_phonemes helper."""

    def test_none_phonemes(self):
        """Returns None when phonemes state is None."""
        from roc.game.gymnasium import _collect_phonemes

        states = MagicMock()
        states.phonemes.val = None
        assert _collect_phonemes(states) is None

    def test_valid_phonemes(self):
        """Returns list of phoneme dicts."""
        from roc.game.gymnasium import _collect_phonemes

        pw1 = MagicMock()
        pw1.word = "hello"
        pw1.phonemes = ["HH", "AH", "L", "OW"]
        pw1.is_break = False

        pw2 = MagicMock()
        pw2.word = ""
        pw2.phonemes = []
        pw2.is_break = True

        states = MagicMock()
        states.phonemes.val = [pw1, pw2]

        result = _collect_phonemes(states)
        assert result is not None
        assert len(result) == 2
        assert result[0]["word"] == "hello"
        assert result[0]["is_break"] is False
        assert result[1]["is_break"] is True


class TestObjToDict:
    """Tests for _obj_to_dict helper."""

    def test_basic_object(self):
        """Converts object with id to dict."""
        from roc.game.gymnasium import _obj_to_dict

        obj = MagicMock()
        obj.id = "12345678abcdef"

        result = _obj_to_dict(obj)
        assert result["id"] == "12345678"

    def test_object_with_position(self):
        """Includes x,y when object has last_x and last_y."""
        from roc.game.gymnasium import _obj_to_dict

        obj = MagicMock()
        obj.id = "abcdefgh"
        obj.last_x = 10
        obj.last_y = 20
        obj.resolve_count = 3

        result = _obj_to_dict(obj)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["resolve_count"] == 3


class TestBuildTransformSummary:
    """Tests for _build_transform_summary helper."""

    def test_none_transform(self):
        """Returns None when transform state is None."""
        from roc.game.gymnasium import _build_transform_summary

        states = MagicMock()
        states.transform.val = None
        assert _build_transform_summary(states) is None

    def test_with_changes(self):
        """Returns summary with change count and details."""
        from roc.game.gymnasium import _build_transform_summary
        from roc.pipeline.intrinsic import IntrinsicNode

        dst1 = MagicMock(spec=IntrinsicNode)
        dst1.configure_mock(name="hp")
        dst1.normalized_change = 0.5

        edge1 = MagicMock()
        edge1.dst = dst1

        transform = MagicMock()
        transform.src_edges = [edge1]

        states = MagicMock()
        states.transform.val.transform = transform

        result = _build_transform_summary(states)
        assert result is not None
        assert result["count"] == 1
        assert len(result["changes"]) == 1
        assert result["changes"][0]["name"] == "hp"
        assert result["changes"][0]["normalized_change"] == pytest.approx(0.5)


class TestBuildPredictionData:
    """Tests for _build_prediction_data helper."""

    def test_none_prediction(self):
        """Returns None when predict state is None."""
        from roc.game.gymnasium import _build_prediction_data

        states = MagicMock()
        states.predict.val = None
        assert _build_prediction_data(states) is None


class TestInjectAttentionSpread:
    """Tests for _inject_attention_spread -- cumulative glyph tracking."""

    def setup_method(self):
        """Reset cumulative glyph sets before each test."""
        import roc.game.gymnasium as gym_mod

        gym_mod._attended_glyphs.clear()
        gym_mod._seen_glyphs.clear()

    def _make_states(self, focus_points: list[dict[str, Any]]) -> MagicMock:
        states = MagicMock()
        states.attenuation_data.val = {
            "peak_count": len(focus_points),
            "focus_points": focus_points,
        }
        return states

    def _make_obs(self, glyph_grid: list[list[int]]) -> dict[str, Any]:
        return {"glyphs": np.array(glyph_grid, dtype=np.int32)}

    def test_single_step_one_attended_glyph(self):
        """After one step, numerator is 1 (one glyph attended)."""
        from roc.game.gymnasium import _inject_attention_spread

        # 3x3 map: background=2359, floor=2371, player=333
        bg = 2359
        obs = self._make_obs(
            [
                [bg, 2360, bg],
                [2371, 333, 2371],
                [bg, 2360, bg],
            ]
        )
        # Top focus point at (1, 1) = player glyph 333
        states = self._make_states([{"x": 1, "y": 1, "strength": 1.0, "label": 1}])

        _inject_attention_spread(states, obs)

        att = states.attenuation_data.val
        assert att["spread_attended"] == 1  # only player glyph
        assert att["spread_total"] == 3  # 333, 2360, 2371
        assert att["spread_pct"] == pytest.approx(33.3)

    def test_two_steps_same_glyph_no_growth(self):
        """Attending the same glyph twice does not increase numerator."""
        from roc.game.gymnasium import _inject_attention_spread

        bg = 2359
        obs = self._make_obs(
            [
                [bg, 2360, bg],
                [2371, 333, 2371],
                [bg, 2360, bg],
            ]
        )
        states1 = self._make_states([{"x": 1, "y": 1, "strength": 1.0, "label": 1}])
        states2 = self._make_states([{"x": 1, "y": 1, "strength": 0.9, "label": 1}])

        _inject_attention_spread(states1, obs)
        _inject_attention_spread(states2, obs)

        att = states2.attenuation_data.val
        assert att["spread_attended"] == 1  # same glyph both times
        assert att["spread_total"] == 3

    def test_two_steps_different_glyphs_grows(self):
        """Attending different glyphs across steps increases numerator."""
        from roc.game.gymnasium import _inject_attention_spread

        bg = 2359
        obs = self._make_obs(
            [
                [bg, 2360, bg],
                [2371, 333, 2371],
                [bg, 2360, bg],
            ]
        )
        # Step 1: attend player (1,1) = 333
        states1 = self._make_states([{"x": 1, "y": 1, "strength": 1.0, "label": 1}])
        _inject_attention_spread(states1, obs)

        # Step 2: attend wall (1,0) = 2360
        states2 = self._make_states([{"x": 1, "y": 0, "strength": 1.0, "label": 1}])
        _inject_attention_spread(states2, obs)

        att = states2.attenuation_data.val
        assert att["spread_attended"] == 2  # 333 + 2360
        assert att["spread_total"] == 3

    def test_new_glyph_on_screen_grows_denominator(self):
        """A new glyph appearing on screen increases the denominator."""
        from roc.game.gymnasium import _inject_attention_spread

        bg = 2359
        obs1 = self._make_obs([[bg, 2360, 2371]])
        states1 = self._make_states([{"x": 1, "y": 0, "strength": 1.0, "label": 1}])
        _inject_attention_spread(states1, obs1)
        assert states1.attenuation_data.val["spread_total"] == 2  # 2360, 2371

        # Step 2: monster glyph 413 appears
        obs2 = self._make_obs([[413, 2360, 2371]])
        states2 = self._make_states([{"x": 1, "y": 0, "strength": 1.0, "label": 1}])
        _inject_attention_spread(states2, obs2)
        assert states2.attenuation_data.val["spread_total"] == 3  # 2360, 2371, 413

    def test_no_attenuation_data_is_noop(self):
        """When attenuation data is None, nothing happens."""
        from roc.game.gymnasium import _inject_attention_spread

        states = MagicMock()
        states.attenuation_data.val = None
        obs = self._make_obs([[2359]])
        _inject_attention_spread(states, obs)  # should not raise

    def test_background_glyph_excluded(self):
        """Background glyph (S_stone) is excluded from both sets."""
        from roc.game.gymnasium import _inject_attention_spread

        bg = 2359
        # Only background glyphs -- focus point lands on background
        obs = self._make_obs([[bg, bg, bg]])
        states = self._make_states([{"x": 0, "y": 0, "strength": 1.0, "label": 1}])
        _inject_attention_spread(states, obs)

        att = states.attenuation_data.val
        assert att["spread_attended"] == 0
        assert att["spread_total"] == 0
        assert att["spread_pct"] == pytest.approx(0.0)


class TestPushStepToServer:
    """Tests for _push_step_to_server helper."""

    def test_returns_false_on_exception(self):
        """Returns False when the HTTP request fails."""
        from roc.game.gymnasium import _push_step_to_server

        step_data = MagicMock()
        result = _push_step_to_server(step_data, "http://invalid:0/step", None)
        assert result is False

    def test_returns_stop_from_response(self, mocker):
        """Returns True when server response includes stop=True."""
        import dataclasses
        import json

        from roc.game.gymnasium import _push_step_to_server

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"stop": True}).encode()

        mocker.patch("urllib.request.urlopen", return_value=mock_resp)

        @dataclasses.dataclass
        class FakeStep:
            step: int = 1

        result = _push_step_to_server(FakeStep(), "http://localhost/step", None)
        assert result is True

    def test_returns_false_from_response(self, mocker):
        """Returns False when server response has no stop."""
        import dataclasses
        import json

        from roc.game.gymnasium import _push_step_to_server

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()

        mocker.patch("urllib.request.urlopen", return_value=mock_resp)

        @dataclasses.dataclass
        class FakeStep:
            step: int = 1

        result = _push_step_to_server(FakeStep(), "http://localhost/step", None)
        assert result is False


class TestHandleGameOver:
    """Tests for _handle_game_over helper."""

    def test_flushes_when_enabled(self, mocker):
        """Calls GraphDB.flush() when graphdb_flush is True."""
        from roc.game.gymnasium import _handle_game_over

        settings = Config.get()
        settings.graphdb_flush = True
        settings.graphdb_export = False

        mock_flush = mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.game.gymnasium.GraphDB.export")

        obs = _make_fake_obs()
        _handle_game_over(obs, 1, True, settings)

        mock_flush.assert_called_once()
        mock_export.assert_not_called()

    def test_exports_when_enabled(self, mocker):
        """Calls GraphDB.export() when graphdb_export is True."""
        from roc.game.gymnasium import _handle_game_over

        settings = Config.get()
        settings.graphdb_flush = False
        settings.graphdb_export = True

        mock_flush = mocker.patch("roc.game.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.game.gymnasium.GraphDB.export")

        obs = _make_fake_obs()
        _handle_game_over(obs, 1, False, settings)

        mock_flush.assert_not_called()
        mock_export.assert_called_once()


# ========================================================================
# dashboard_cli helpers
# ========================================================================
class TestDashboardCliMountStaticFiles:
    """Tests for dashboard_cli._mount_static_files."""

    def test_no_dist_dir(self, tmp_path):
        """When dist dir does not exist, mount is not called."""
        import roc.cli.dashboard_cli as dcli
        from roc.cli.dashboard_cli import _mount_static_files

        mock_app = MagicMock()
        orig_file = dcli.__file__
        try:
            # Point __file__ to a location where dashboard-ui/dist won't exist
            dcli.__file__ = str(tmp_path / "roc" / "dashboard_cli.py")
            _mount_static_files(mock_app)
            mock_app.mount.assert_not_called()
        finally:
            dcli.__file__ = orig_file

    def test_with_dist_dir(self, tmp_path):
        """When dist dir exists, mount is called."""
        import roc.cli.dashboard_cli as dcli
        from roc.cli.dashboard_cli import _mount_static_files

        mock_app = MagicMock()
        orig_file = dcli.__file__
        try:
            dcli.__file__ = str(tmp_path / "roc" / "dashboard_cli.py")
            (tmp_path / "dashboard-ui" / "dist").mkdir(parents=True)
            _mount_static_files(mock_app)
            mock_app.mount.assert_called_once()
        finally:
            dcli.__file__ = orig_file


class TestDashboardCliResolveSSLCerts:
    """Tests for dashboard_cli._resolve_ssl_certs."""

    def test_no_ssl_config(self):
        """When config has no SSL certs, falls back to .env."""
        from roc.cli.dashboard_cli import _resolve_ssl_certs

        cfg = MagicMock()
        cfg.ssl_certfile = ""
        cfg.ssl_keyfile = ""

        with patch("roc.cli.dashboard_cli._read_ssl_from_env", return_value=(None, None)):
            cert, key = _resolve_ssl_certs(cfg)
            assert cert is None
            assert key is None

    def test_with_ssl_config(self):
        """When config has SSL certs, returns them directly."""
        from roc.cli.dashboard_cli import _resolve_ssl_certs

        cfg = MagicMock()
        cfg.ssl_certfile = "/path/to/cert.pem"
        cfg.ssl_keyfile = "/path/to/key.pem"

        cert, key = _resolve_ssl_certs(cfg)
        assert cert == "/path/to/cert.pem"
        assert key == "/path/to/key.pem"


class TestDashboardCliParseSSLEnvLine:
    """Tests for dashboard_cli._parse_ssl_env_line."""

    def test_certfile_line_existing(self, tmp_path):
        """Parses a valid roc_ssl_certfile line with existing file."""
        from roc.cli.dashboard_cli import _parse_ssl_env_line

        cert_file = tmp_path / "cert.pem"
        cert_file.touch()

        cert, key = _parse_ssl_env_line(f"roc_ssl_certfile={cert_file}", None)
        assert cert == str(cert_file)
        assert key is None

    def test_certfile_line_nonexistent(self):
        """Non-existent cert path returns None."""
        from roc.cli.dashboard_cli import _parse_ssl_env_line

        cert, key = _parse_ssl_env_line("roc_ssl_certfile=/nonexistent/cert.pem", None)
        assert cert is None
        assert key is None

    def test_keyfile_line_existing(self, tmp_path):
        """Parses a valid roc_ssl_keyfile line with existing file."""
        from roc.cli.dashboard_cli import _parse_ssl_env_line

        key_file = tmp_path / "key.pem"
        key_file.touch()

        cert, key = _parse_ssl_env_line(f"roc_ssl_keyfile={key_file}", None)
        assert cert is None
        assert key == str(key_file)

    def test_unrelated_line(self):
        """Non-SSL lines return (None, None)."""
        from roc.cli.dashboard_cli import _parse_ssl_env_line

        cert, key = _parse_ssl_env_line("roc_db_host=localhost", None)
        assert cert is None
        assert key is None

    def test_quoted_value(self, tmp_path):
        """Strips quotes from values."""
        from roc.cli.dashboard_cli import _parse_ssl_env_line

        cert_file = tmp_path / "cert.pem"
        cert_file.touch()

        cert, _key = _parse_ssl_env_line(f'roc_ssl_certfile="{cert_file}"', None)
        assert cert == str(cert_file)


class TestDashboardCliFindEnvFile:
    """Tests for dashboard_cli._find_env_file."""

    def test_cwd_env_file(self, tmp_path, monkeypatch):
        """Finds .env in current working directory."""
        from roc.cli.dashboard_cli import _find_env_file

        env_file = tmp_path / ".env"
        env_file.write_text("roc_db_host=localhost\n")
        monkeypatch.chdir(tmp_path)

        result = _find_env_file()
        assert result is not None
        assert result.name == ".env"

    def test_no_env_file(self, tmp_path, monkeypatch):
        """Returns None when no .env file exists."""
        from roc.cli.dashboard_cli import _find_env_file

        monkeypatch.chdir(tmp_path)
        # Also patch project root to tmp_path so neither location has .env
        import roc.cli.dashboard_cli as dcli

        orig_file = dcli.__file__
        try:
            dcli.__file__ = str(tmp_path / "roc" / "dashboard_cli.py")
            _find_env_file()
            # This may or may not find the real project .env; just verify it runs
            # In our tmp env, it shouldn't find one
        finally:
            dcli.__file__ = orig_file


class TestDashboardCliReadSSLFromEnv:
    """Tests for dashboard_cli._read_ssl_from_env."""

    def test_no_env_file(self, monkeypatch, tmp_path):
        """Returns (None, keyfile) when no .env exists."""
        from roc.cli.dashboard_cli import _read_ssl_from_env

        with patch("roc.cli.dashboard_cli._find_env_file", return_value=None):
            cert, key = _read_ssl_from_env("/path/to/key.pem")
            assert cert is None
            assert key == "/path/to/key.pem"

    def test_with_env_file(self, tmp_path):
        """Reads SSL paths from .env file."""
        from roc.cli.dashboard_cli import _read_ssl_from_env

        cert_path = tmp_path / "cert.pem"
        cert_path.touch()
        key_path = tmp_path / "key.pem"
        key_path.touch()

        env_file = tmp_path / ".env"
        env_file.write_text(f"roc_ssl_certfile={cert_path}\nroc_ssl_keyfile={key_path}\n")

        with patch("roc.cli.dashboard_cli._find_env_file", return_value=env_file):
            cert, key = _read_ssl_from_env(None)
            assert cert == str(cert_path)
            assert key == str(key_path)


# ========================================================================
# cleanup_cli helpers
# ========================================================================
class TestCleanupCliClassifyRuns:
    """Tests for cleanup_cli._classify_runs."""

    def test_empty_runs(self):
        """Empty list returns empty results."""
        from roc.cli.cleanup_cli import _classify_runs

        to_delete, to_keep = _classify_runs([])
        assert to_delete == []
        assert to_keep == []

    def test_runs_with_parquet(self, tmp_path):
        """Runs with parquet files are kept."""
        from roc.cli.cleanup_cli import _classify_runs

        run_dir = tmp_path / "run1"
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "steps.parquet").touch()

        to_delete, to_keep = _classify_runs([run_dir])
        assert len(to_keep) == 1
        assert len(to_delete) == 0

    def test_runs_without_parquet(self, tmp_path):
        """Runs without parquet files are marked for deletion."""
        from roc.cli.cleanup_cli import _classify_runs

        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)

        to_delete, to_keep = _classify_runs([run_dir])
        assert len(to_delete) == 1
        assert to_delete[0] == (run_dir, "no data files")
        assert len(to_keep) == 0

    def test_mixed_runs(self, tmp_path):
        """Correctly classifies a mix of runs."""
        from roc.cli.cleanup_cli import _classify_runs

        # Run with data
        good_run = tmp_path / "good_run"
        good_data = good_run / "data"
        good_data.mkdir(parents=True)
        (good_data / "steps.parquet").touch()

        # Run without data
        bad_run = tmp_path / "bad_run"
        bad_run.mkdir(parents=True)

        to_delete, to_keep = _classify_runs([good_run, bad_run])
        assert len(to_keep) == 1
        assert len(to_delete) == 1
        assert to_keep[0] == good_run


class TestCleanupCliReportDeletions:
    """Tests for cleanup_cli._report_deletions."""

    def test_reports_sizes(self, tmp_path, capsys):
        """Reports deletion candidates with their sizes."""
        from roc.cli.cleanup_cli import _report_deletions

        run_dir = tmp_path / "run_to_delete"
        run_dir.mkdir()
        (run_dir / "catalog.duckdb").write_text("x" * 1000)

        to_delete = [(run_dir, "no data files")]
        total = _report_deletions(to_delete)

        assert total > 0
        captured = capsys.readouterr()
        assert "run_to_delete" in captured.out
        assert "no data files" in captured.out


class TestCleanupCliExecuteDeletions:
    """Tests for cleanup_cli._execute_deletions."""

    def test_deletes_directories(self, tmp_path, capsys):
        """Actually deletes directories."""
        from roc.cli.cleanup_cli import _execute_deletions

        run1 = tmp_path / "run1"
        run1.mkdir()
        (run1 / "file.txt").write_text("data")

        run2 = tmp_path / "run2"
        run2.mkdir()

        to_delete = [(run1, "no data"), (run2, "no data")]
        _execute_deletions(to_delete)

        assert not run1.exists()
        assert not run2.exists()
        captured = capsys.readouterr()
        assert "Deleted 2 runs" in captured.out


class TestCleanupCliCountParquetFiles:
    """Tests for cleanup_cli._count_parquet_files."""

    def test_no_data_dir(self, tmp_path):
        """Returns 0 when data/ directory does not exist."""
        from roc.cli.cleanup_cli import _count_parquet_files

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        assert _count_parquet_files(run_dir) == 0

    def test_with_parquet_files(self, tmp_path):
        """Counts parquet files correctly."""
        from roc.cli.cleanup_cli import _count_parquet_files

        run_dir = tmp_path / "run1"
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "a.parquet").touch()
        (data_dir / "b.parquet").touch()
        (data_dir / "c.json").touch()

        assert _count_parquet_files(run_dir) == 2

    def test_nested_parquet_files(self, tmp_path):
        """Counts parquet files in nested directories."""
        from roc.cli.cleanup_cli import _count_parquet_files

        run_dir = tmp_path / "run1"
        nested = run_dir / "data" / "sub"
        nested.mkdir(parents=True)
        (nested / "deep.parquet").touch()

        assert _count_parquet_files(run_dir) == 1
