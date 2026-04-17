"""Default action ExpMod: always pass (NetHack ``.`` character)."""

from roc.pipeline.action import DefaultActionExpMod


class DefaultActionPass(DefaultActionExpMod):
    """Default action module that always passes (does nothing)."""

    name = "pass"

    def get_action(self) -> int:
        """Return the pass action (action ID 19 in NetHack gym encoding)."""
        return 19
