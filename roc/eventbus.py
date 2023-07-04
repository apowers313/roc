from __future__ import annotations

from typing import Dict

eb_dict: dict[str, EventBus] = {}


class EventBus:
    def __init__(self, name: str):
        self._name = name

    @staticmethod
    def get(name: str) -> EventBus | None:
        try:
            return eb_dict[name]
        except KeyError:
            return None
