# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/script.py CLI option generation."""

from unittest.mock import patch

from click.testing import CliRunner

from roc.script import JsonParam, cli


class TestCliOptionGeneration:
    def test_bool_flag_true(self):
        """--graphdb-export should pass graphdb_export=True to roc.init()."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--graphdb-export"])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["graphdb_export"] is True

    def test_bool_flag_false(self):
        """--no-graphdb-export should pass graphdb_export=False to roc.init()."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--no-graphdb-export"])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["graphdb_export"] is False

    def test_int_option(self):
        """--num-games 3 should pass num_games=3."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--num-games", "3"])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["num_games"] == 3

    def test_str_option(self):
        """--db-host should pass a string value."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--db-host", "10.0.0.1"])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["db_host"] == "10.0.0.1"

    def test_json_option(self):
        """--nethack-extra-options should accept a JSON list."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--nethack-extra-options", '["autoopen", "color"]'])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["nethack_extra_options"] == ["autoopen", "color"]

    def test_json_dict_option(self):
        """--significance-weights should accept a JSON dict."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--significance-weights", '{"hp": 5.0}'])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["significance_weights"] == {"hp": 5.0}

    def test_no_options_passes_none(self):
        """No CLI args should call roc.init(config=None)."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, [])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            mock_roc.init.assert_called_once_with(config=None)

    def test_only_provided_options_are_passed(self):
        """Only explicitly-set options should appear in the config dict."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, ["--num-games", "1"])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert "num_games" in config
            assert "db_host" not in config

    def test_multiple_options_combined(self):
        """Multiple options should all appear in the config dict."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(
                cli, ["--num-games", "2", "--no-graphdb-export", "--db-host", "myhost"]
            )
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            config = mock_roc.init.call_args[1]["config"]
            assert config["num_games"] == 2
            assert config["graphdb_export"] is False
            assert config["db_host"] == "myhost"

    def test_start_is_called(self):
        """roc.start() should be called after roc.init()."""
        runner = CliRunner()
        with patch("roc.script.roc") as mock_roc:
            result = runner.invoke(cli, [])
            assert result.exit_code == 0, f"{result.output}\n{result.exception}"
            mock_roc.start.assert_called_once()


class TestJsonParam:
    def test_parses_list(self):
        param = JsonParam()
        assert param.convert('["a", "b"]', None, None) == ["a", "b"]

    def test_parses_dict(self):
        param = JsonParam()
        assert param.convert('{"k": 1}', None, None) == {"k": 1}

    def test_passes_through_non_string(self):
        param = JsonParam()
        assert param.convert([1, 2], None, None) == [1, 2]

    def test_invalid_json_fails(self):
        from click import BadParameter

        param = JsonParam()
        try:
            param.convert("not json", None, None)
            assert False, "Should have raised"
        except BadParameter:
            pass
