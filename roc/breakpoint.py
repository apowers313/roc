import inspect
import os
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, TypeAlias

from tabulate import tabulate

from .logger import logger

ConditionFn: TypeAlias = Callable[[], bool]


@dataclass
class BreakpointInfo:
    fn: ConditionFn
    src: str | None


_breakpoints_dict: Dict[str, BreakpointInfo] = dict()


class Breakpoint:
    def __init__(self) -> None:
        self.brk = False
        self.trigger: str | None = None
        self.lock = Lock()

    def __len__(self) -> int:
        global _breakpoints_dict

        return len(_breakpoints_dict)

    def __contains__(self, key: str) -> bool:
        global _breakpoints_dict

        return key in _breakpoints_dict

    def __str__(self) -> str:
        global _breakpoints_dict

        def mkrow(k: str) -> list[str | bool]:
            fn = _breakpoints_dict[k].fn
            src = _breakpoints_dict[k].src
            if not src:
                filename = inspect.getfile(fn).split(os.path.sep)[-1]
                line = inspect.getsourcelines(fn)[1]
                src = f"{filename}:{line}"
            triggered = " " if (self.trigger is None or self.trigger != k) else "*"
            return [triggered, k, src]

        rows = map(mkrow, _breakpoints_dict.keys())

        hdr = f"{self.count} breakpoint(s). State: {self.state}."
        tbl = (
            ""
            if self.count == 0
            else "\n\n"
            + tabulate(
                rows, headers=[" ", "Breakpoints", "Source"], showindex="always", tablefmt="simple"
            )
        )

        return f"{hdr}{tbl}"

    def add(
        self,
        fn: ConditionFn,
        *,
        name: str | None = None,
        overwrite: bool = False,
        src: str | None = None,
    ) -> None:
        global _breakpoints_dict

        if not name:
            name = fn.__name__

        if name in _breakpoints_dict and not overwrite:
            raise Exception(
                f"'{name}' already exists in breakpoints, call 'remove' first or specify 'overwrite=True' while adding"
            )

        _breakpoints_dict[name] = BreakpointInfo(fn=fn, src=src)

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

    def resume(self, quiet: bool = False) -> None:
        if not self.brk:
            return

        self.brk = False
        self.trigger = None
        self.lock.release()

        if not quiet:
            logger.info("resuming")

    def do_break(self, trigger: str = "<user request>", quiet: bool = False) -> None:
        if self.brk:
            return

        self.brk = True
        self.trigger = trigger

        if not quiet:
            logger.info(f"breaking due to: {trigger}")

        self.lock.acquire()

    def check(self) -> None:
        global _breakpoints_dict

        waslocked = False
        if self.lock.locked():
            waslocked = True

        # stop here if we are in a break
        with self.lock:
            # if we were stopped and are continuing now, don't immediately stop
            # again due to another breakpoint
            if not waslocked:
                for b in _breakpoints_dict:
                    if _breakpoints_dict[b].fn():
                        self.do_break(trigger=b)
                        waslocked = True
                        break

        if waslocked:
            logger.info("break done, continuing operations...")


breakpoints = Breakpoint()
