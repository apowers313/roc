"""Always-right action ExpMod (named ``right`` for historical reasons)."""

from roc.framework.config import Config
from roc.pipeline.action import DefaultActionExpMod


class DefaultActionLeft(DefaultActionExpMod):
    """Always selects the 'move right' action."""

    name = "right"

    def get_action(self) -> int:
        """Return the index of the 'move right' action (NetHack keycode 108, ``l``)."""
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")
        return settings.gym_actions.index(108)
