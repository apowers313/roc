import functools
from collections import Counter
from random import randrange

from roc.action import DefaultActionExpMod
from roc.config import Config

# LEGEND:
# settings.gym_actions (
# <CompassDirection.N: 107>,
# <CompassDirection.E: 108>,
# <CompassDirection.S: 106>,
# <CompassDirection.W: 104>,
# <CompassDirection.NE: 117>,
# <CompassDirection.SE: 110>,
# <CompassDirection.SW: 98>,
# <CompassDirection.NW: 121>,
# <CompassDirectionLonger.N: 75>,
# <CompassDirectionLonger.E: 76>,
# <CompassDirectionLonger.S: 74>,
# <CompassDirectionLonger.W: 72>,
# <CompassDirectionLonger.NE: 85>,
# <CompassDirectionLonger.SE: 78>,
# <CompassDirectionLonger.SW: 66>,
# <CompassDirectionLonger.NW: 89>,
# <MiscDirection.UP: 60>,
# <MiscDirection.DOWN: 62>,
# <MiscDirection.WAIT: 46>,
# <MiscAction.MORE: 13>,
# <Command.ADJUST: 225>,
# <Command.APPLY: 97>,
# <Command.ATTRIBUTES: 24>,
# <Command.CALL: 67>,
# <Command.CAST: 90>,
# <Command.CHAT: 227>,
# <Command.CLOSE: 99>,
# <Command.DIP: 228>,
# <Command.DROP: 100>,
# <Command.DROPTYPE: 68>,
# <Command.EAT: 101>,
# <Command.ENGRAVE: 69>,
# <Command.ENHANCE: 229>,
# <Command.ESC: 27>,
# <Command.FIGHT: 70>,
# <Command.FIRE: 102>,
# <Command.FORCE: 230>,
# <Command.INVENTORY: 105>,
# <Command.INVENTTYPE: 73>,
# <Command.INVOKE: 233>,
# <Command.JUMP: 234>,
# <Command.KICK: 4>,
# <Command.LOOK: 58>,
# <Command.LOOT: 236>,
# <Command.MONSTER: 237>,
# <Command.MOVE: 109>,
# <Command.MOVEFAR: 77>,
# <Command.OFFER: 239>,
# <Command.OPEN: 111>,
# <Command.PAY: 112>,
# <Command.PICKUP: 44>,
# <Command.PRAY: 240>,
# <Command.PUTON: 80>,
# <Command.QUAFF: 113>,
# <Command.QUIVER: 81>,
# <Command.READ: 114>,
# <Command.REMOVE: 82>,
# <Command.RIDE: 210>,
# <Command.RUB: 242>,
# <Command.RUSH: 103>,
# <Command.RUSH2: 71>,
# <Command.SEARCH: 115>,
# <Command.SEEARMOR: 91>,
# <Command.SEERINGS: 61>,
# <Command.SEETOOLS: 40>,
# <Command.SEETRAP: 94>,
# <Command.SEEWEAPON: 41>,
# <Command.SHELL: 33>,
# <Command.SIT: 243>,
# <Command.SWAP: 120>,
# <Command.TAKEOFF: 84>,
# <Command.TAKEOFFALL: 65>,
# <Command.THROW: 116>,
# <Command.TIP: 212>,
# <Command.TURN: 244>,
# <Command.TWOWEAPON: 88>,
# <Command.UNTRAP: 245>,
# <Command.VERSIONSHORT: 118>,
# <Command.WEAR: 87>,
# <Command.WIELD: 119>,
# <Command.WIPE: 247>,
# <Command.ZAP: 122>,
# <TextCharacters.PLUS: 43>,
# <TextCharacters.QUOTE: 34>,
# <TextCharacters.DOLLAR: 36>,
# <TextCharacters.SPACE: 32>
# )


# @DefaultActionExpMod.register("random")
class DefaultActionRandom(DefaultActionExpMod):
    name = "random"

    def get_action(self) -> int:
        """A default action for taking random Nethack actions. Random movement."""
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")
        num_actions = len(settings.gym_actions)

        ret = randrange(num_actions)
        return ret


# @DefaultActionExpMod.register("random")
class DefaultActionLeft(DefaultActionExpMod):
    name = "right"

    def get_action(self) -> int:
        """Always move right."""
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")

        idx = settings.gym_actions.index(108)

        return idx


# @DefaultActionExpMod.register("weighted")
class WeightedAction(DefaultActionExpMod):
    name = "weighted"

    @functools.cache
    def actions(self, actions: tuple[int]) -> list[int]:
        weighted_map = {
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
        weights = Counter({k: (v - 1) for k, v in weighted_map.items()})
        weighted_actions = Counter(list(actions)) + weights

        return list(weighted_actions.elements())

    def get_action(self) -> int:
        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")

        actions = self.actions(settings.gym_actions)
        num_actions = len(actions)
        val_idx = randrange(num_actions)
        val = actions[val_idx]
        idx = settings.gym_actions.index(val)
        # print(f"action ({num_actions}) val: {val}, idx: {idx}")
        return idx
