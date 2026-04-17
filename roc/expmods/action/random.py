"""Random action ExpMod."""

from random import randrange

from roc.framework.config import Config
from roc.pipeline.action import DefaultActionExpMod


class DefaultActionRandom(DefaultActionExpMod):
    """Selects a uniformly random action from the configured action space."""

    name = "random"

    def get_action(self) -> int:
        """Return a uniformly random valid action ID."""
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")
        return randrange(len(settings.gym_actions))
