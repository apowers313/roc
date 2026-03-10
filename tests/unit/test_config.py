# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/config.py."""

import pytest

from roc.config import (
    Config,
    ConfigBoolIntrinsic,
    ConfigInitWarning,
    ConfigIntIntrinsic,
    ConfigMapIntrinsic,
    ConfigPercentIntrinsic,
    config_settings,
)


class TestConfigInitAndReset:
    def test_init_with_defaults(self):
        # unit_config_init already calls reset/init, so singleton exists
        cfg = Config.get()
        assert cfg.db_host == "127.0.0.1"
        assert cfg.db_port == 7687
        assert cfg.db_conn_encrypted is False

    def test_init_with_custom_values(self):
        Config.reset()
        Config.init(config={"db_host": "10.0.0.1", "db_port": 1234})
        cfg = Config.get()
        assert cfg.db_host == "10.0.0.1"
        assert cfg.db_port == 1234

    def test_init_warns_if_already_initialized(self):
        # Config is already initialized by fixture
        with pytest.warns(ConfigInitWarning, match="already initialized"):
            Config.init()

    def test_init_force_reinit(self):
        Config.init(config={"db_host": "first.host"}, force=True)
        assert Config.get().db_host == "first.host"
        Config.init(config={"db_host": "second.host"}, force=True)
        assert Config.get().db_host == "second.host"

    def test_reset_sets_singleton_to_none(self):
        Config.reset()
        # After reset, get() should warn about uninitialized
        with pytest.warns(ConfigInitWarning, match="before config module was initialized"):
            Config.get()

    def test_get_returns_singleton(self):
        cfg1 = Config.get()
        cfg2 = Config.get()
        assert cfg1 is cfg2


class TestConfigStr:
    def test_str_formats_all_fields(self):
        cfg = Config.get()
        s = str(cfg)
        assert "db_host" in s
        assert "db_port" in s
        assert "log_level" in s
        # Each field should be on its own line
        lines = s.strip().split("\n")
        assert len(lines) > 5


class TestConfigIntrinsicTypes:
    def test_percent_intrinsic(self):
        ci = ConfigPercentIntrinsic(name="hp", config="hpmax")
        assert ci.type == "percent"
        assert ci.name == "hp"
        assert ci.config == "hpmax"

    def test_map_intrinsic(self):
        ci = ConfigMapIntrinsic(name="hunger", config={0: 0.5, 1: 1.0})
        assert ci.type == "map"
        assert ci.name == "hunger"
        assert ci.config == {0: 0.5, 1: 1.0}

    def test_int_intrinsic(self):
        ci = ConfigIntIntrinsic(name="level", config=(1, 50))
        assert ci.type == "int"
        assert ci.name == "level"
        assert ci.config == (1, 50)

    def test_bool_intrinsic(self):
        ci = ConfigBoolIntrinsic(name="alive")
        assert ci.type == "bool"
        assert ci.name == "alive"
        assert ci.config is None


class TestConfigPytestIsolation:
    def test_env_prefix_is_overridden_in_pytest(self):
        # When running under pytest, config_settings should have a weird prefix
        assert "somereallyweirdrandomstring" in config_settings["env_prefix"]

    def test_env_file_is_overridden_in_pytest(self):
        assert "somereallyweirdrandomstring" in config_settings["env_file"]


class TestConfigDefaults:
    def test_default_intrinsics_list(self):
        cfg = Config.get()
        assert len(cfg.intrinsics) >= 2
        assert isinstance(cfg.intrinsics[0], ConfigPercentIntrinsic)

    def test_default_perception_components(self):
        cfg = Config.get()
        assert len(cfg.perception_components) > 0
        # Should be list of (name, type) tuples
        assert isinstance(cfg.perception_components[0], (list, tuple))

    def test_graphdb_export_default_false(self):
        assert Config.get().graphdb_export is False

    def test_graphdb_flush_default_false(self):
        assert Config.get().graphdb_flush is False

    def test_debug_log_default_false(self):
        assert Config.get().debug_log is False

    def test_debug_log_path_default(self):
        path = Config.get().debug_log_path
        assert path.startswith("tmp/debug_log-")
        assert path.endswith(".jsonl")

    def test_debug_remote_log_default_true(self):
        assert Config.get().debug_remote_log is True

    def test_debug_remote_log_url_default(self):
        assert Config.get().debug_remote_log_url == "https://dev.ato.ms:9080/log"

    def test_debug_snapshot_interval_default_zero(self):
        assert Config.get().debug_snapshot_interval == 0


class TestConfigInitWarningDetails:
    def test_init_with_different_values_warns_with_diff(self):
        """Config.init() with different values should warn and show what changed."""
        Config.init(config={"db_host": "first.host"}, force=True)
        with pytest.warns(ConfigInitWarning, match="db_host"):
            Config.init(config={"db_host": "second.host"})

    def test_init_warning_includes_caller_info(self):
        """Warning message should include the caller's file and line."""
        Config.init(config={}, force=True)
        with pytest.warns(ConfigInitWarning, match=r"test_config\.py"):
            Config.init()

    def test_init_with_same_values_warns_without_diff(self):
        """Config.init() with same values should warn but not show diff."""
        Config.init(config={"db_host": "127.0.0.1"}, force=True)
        with pytest.warns(ConfigInitWarning, match="already initialized"):
            Config.init(config={"db_host": "127.0.0.1"})
