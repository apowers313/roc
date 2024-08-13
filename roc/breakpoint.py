"""Execution control for ROC, primarily used in Jupyter notebooks"""

import inspect
import os
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, Optional, TypeAlias

from tabulate import tabulate

from .logger import logger

ConditionFn: TypeAlias = Callable[[], bool]


@dataclass
class BreakpointInfo:
    """Information about a specific breakpoint"""

    fn: ConditionFn
    src: str | None


_breakpoints_dict: Dict[str, BreakpointInfo] = dict()


class Breakpoint:
    """Controls the state of breakpoints and program execution. This should
    probably be renamed to `BreakpointControl`. :)
    """

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
        overwrite: Optional[bool] = False,
        src: str | None = None,
    ) -> None:
        """Adds a new breakpoint

        Args:
            fn (ConditionFn): A function that returns True when program
                execution should stop
            name (str | None, optional): The name of this breakpoint. Defaults
                to the __name__ of the specified function, or '<unknown>' if name
                isn't specified and __name__ doesn't exist.
            overwrite (bool, optional): If True, this will overwrite existing
                breakpoints with the same name. Defaults to False.
            src (str | None, optional): The source / sourcecode for the
                breakpoint. If not specified, it will use the file and line
                number of the specified function. Mostly used by the Jupyter
                shell to pass in 'iPython' as the source for functions defined
                in Jupyter, because knowing that the function is stored in a
                tempfile isn't especially helpful.

        Raises:
            Exception: Raises an exception if the specified breakpoint name
            already exists and overwrite is False.
        """
        global _breakpoints_dict

        if not name and hasattr(fn, "__name__"):
            name = fn.__name__

        if not name:
            name = "<unknown>"

        if name in _breakpoints_dict and not overwrite:
            raise Exception(
                f"'{name}' already exists in breakpoints, call 'remove' first or specify 'overwrite=True' while adding"
            )

        _breakpoints_dict[name] = BreakpointInfo(fn=fn, src=src)

    def remove(self, name: str) -> None:
        """Removes the breakpoint with the specified name

        Args:
            name (str): The name of the breakpoint to remove

        Raises:
            Exception: If the specified breakpoint doesn't exist
        """
        global _breakpoints_dict

        if name not in _breakpoints_dict:
            raise Exception(f"can't remove '{name}' from breakpoints because it doesn't exist")

        del _breakpoints_dict[name]

    def clear(self) -> None:
        """Removes all breakpoints"""
        global _breakpoints_dict

        _breakpoints_dict.clear()

    @property
    def count(self) -> int:
        """The number of breakpoints that have been added."""
        return len(self)

    @property
    def state(self) -> str:
        """The state of execution, either 'stopped' or 'running'"""
        return "stopped" if self.brk else "running"

    def list(self) -> None:
        """Prints all current breakpoints and the state of execution"""
        print(str(self))  # noqa: T201

    def resume(self, quiet: Optional[bool] = False) -> None:
        """If execution is currently stopped, this will resume execution. If not
        stopped, this does nothing.

        Args:
            quiet (bool, optional): If True, this function won't log status
                messages about resuming. Defaults to False.
        """
        if not self.brk:
            return

        self.brk = False
        self.trigger = None
        self.lock.release()

        if not quiet:
            logger.info("resuming")

    def do_break(
        self, trigger: Optional[str] = "<user request>", quiet: Optional[bool] = False
    ) -> None:
        """Sets the state of execution to 'stopped' and blocks further execution
        until `resume` is called. This is typically called by `check` inside the
        program, but may be called directly (e.g. by a Jupyter magic command) to
        manually stop execution.

        Args:
            trigger (str, optional): The name of the breakpoint that caused the
                break. Defaults to "<user request>".
            quiet (bool, optional): If True, this function won't log status
                messages about resuming. Defaults to False.
        """
        if self.brk:
            return

        self.brk = True
        self.trigger = trigger

        if not quiet:
            logger.info(f"breaking due to: {trigger}")

        self.lock.acquire()

    def check(self) -> None:
        """Checks whether any breakpoints are currently triggered by calling the
        functions that were submitted by `add`. If / when the first function
        returns True, program execution stops and will only resume when `resume`
        is called.

        This function is intended to be included in a execution loop or any
        place you want to check for breakpoints before continuing to execute code.
        """
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
