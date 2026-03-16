# mypy: disable-error-code="no-untyped-def"

"""Unit tests for W&B config fields in roc/config.py."""

from roc.config import Config


class TestWandbDefaults:
    def test_wandb_enabled_default(self):
        settings = Config.get()
        assert settings.wandb_enabled is False

    def test_wandb_project_default(self):
        settings = Config.get()
        assert settings.wandb_project == "ROC"

    def test_wandb_entity_default(self):
        settings = Config.get()
        assert settings.wandb_entity == ""

    def test_wandb_host_default(self):
        settings = Config.get()
        assert settings.wandb_host == ""

    def test_wandb_api_key_default(self):
        settings = Config.get()
        assert settings.wandb_api_key == ""

    def test_wandb_tags_default(self):
        settings = Config.get()
        assert settings.wandb_tags == []

    def test_wandb_log_screens_default(self):
        settings = Config.get()
        assert settings.wandb_log_screens is True

    def test_wandb_log_saliency_default(self):
        settings = Config.get()
        assert settings.wandb_log_saliency is True

    def test_wandb_log_interval_default(self):
        settings = Config.get()
        assert settings.wandb_log_interval == 1

    def test_wandb_artifacts_default(self):
        settings = Config.get()
        assert settings.wandb_artifacts == []

    def test_wandb_mode_default(self):
        settings = Config.get()
        assert settings.wandb_mode == "online"


class TestWandbEnvVars:
    def test_wandb_enabled_from_config(self):
        Config.reset()
        Config.init({"wandb_enabled": True})
        settings = Config.get()
        assert settings.wandb_enabled is True

    def test_wandb_project_from_config(self):
        Config.reset()
        Config.init({"wandb_project": "my-project"})
        settings = Config.get()
        assert settings.wandb_project == "my-project"

    def test_wandb_mode_from_config(self):
        Config.reset()
        Config.init({"wandb_mode": "disabled"})
        settings = Config.get()
        assert settings.wandb_mode == "disabled"
