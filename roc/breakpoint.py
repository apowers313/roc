import inspect
import os
from typing import Callable, Dict, TypeAlias

from tabulate import tabulate

ConditionFn: TypeAlias = Callable[[], bool]

_breakpoints_dict: Dict[str, ConditionFn] = dict()


class Breakpoint:
    def __init__(self) -> None:
        self.brk = False
        self.trigger: str | None = None

    def __len__(self) -> int:
        global _breakpoints_dict

        return len(_breakpoints_dict)

    def __contains__(self, key: str) -> bool:
        global _breakpoints_dict

        return key in _breakpoints_dict

    def __str__(self) -> str:
        global _breakpoints_dict

        def mkrow(k: str) -> list[str | bool]:
            fn = _breakpoints_dict[k]
            filename = inspect.getfile(fn).split(os.path.sep)[-1]
            line = inspect.getsourcelines(fn)[1]
            triggered = " " if (self.trigger is None or self.trigger != k) else "*"
            return [triggered, k, f"{filename}:{line}"]

        rows = map(mkrow, _breakpoints_dict.keys())

        hdr = f"{self.count} breakpoint(s). State: {self.state}."
        tbl = (
            ""
            if self.count == 0
            else "\n\n"
            + tabulate(
                rows, headers=[" ", "Breakpoints", "File"], showindex="always", tablefmt="simple"
            )
        )

        return f"{hdr}{tbl}"

    def add(self, name: str, fn: ConditionFn, overwrite: bool = False) -> None:
        global _breakpoints_dict

        if name in _breakpoints_dict and not overwrite:
            raise Exception(
                f"'{name}' already exists in breakpoints, call 'remove' first or specify 'overwrite=True' while adding"
            )

        _breakpoints_dict[name] = fn

    def remove(self, name: str) -> None:
        global _breakpoints_dict

        if name not in _breakpoints_dict:
            raise Exception(f"can't remove '{name}' from breakpoints because it doesn't exist")

        del _breakpoints_dict[name]

    def clear(self) -> None:
        global _breakpoints_dict

        _breakpoints_dict.clear()

    @property
    def count(self) -> int:
        return len(self)

    @property
    def state(self) -> str:
        return "stopped" if self.brk else "running"

    def list(self) -> None:
        print(str(self))  # noqa: T201

    def resume(self) -> None:
        self.brk = False
        self.trigger = None

    def check(self) -> None:
        global _breakpoints_dict

        for b in _breakpoints_dict:
            if _breakpoints_dict[b]():
                self.brk = True
                self.trigger = b
                break


breakpoints = Breakpoint()
