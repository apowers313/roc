"""Weighted random action ExpMod: biases toward movement and eating."""

import functools
from collections import Counter
from random import randrange

from roc.framework.config import Config
from roc.framework.expmod import ExpModConfig
from roc.pipeline.action import DefaultActionExpMod


class WeightedActionConfig(ExpModConfig):
    """Per-action weight map for the ``weighted`` action selector.

    Keys are NetHack keycodes (ints); values are selection weights. Any valid
    action not listed in the map is effectively weight 1.
    """

    weights: dict[int, int] = {
        ord("j"): 100,  # move down
        ord("k"): 100,  # move up
        ord("h"): 100,  # move left
        ord("l"): 100,  # move right
        ord("u"): 100,  # move up-right
        ord("y"): 100,  # move up-left
        ord("b"): 100,  # move down-left
        ord("n"): 100,  # move down-right
        ord("e"): 25,  # eat
    }


class WeightedAction(DefaultActionExpMod):
    """Selects an action weighted by a per-keycode weight map."""

    name = "weighted"
    config_schema = WeightedActionConfig

    @functools.cache
    def _weighted_list(self, actions: tuple[int, ...]) -> list[int]:
        """Build the (cached) flat list of weighted action keycodes."""
        assert isinstance(self.config, WeightedActionConfig)
        weights = Counter({k: (v - 1) for k, v in self.config.weights.items()})
        weighted_actions = Counter(list(actions)) + weights
        return list(weighted_actions.elements())

    def get_action(self) -> int:
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")

        actions = self._weighted_list(settings.gym_actions)
        val = actions[randrange(len(actions))]
        return settings.gym_actions.index(val)
