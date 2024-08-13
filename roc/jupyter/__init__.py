"""Jupyter magics for flow control and debugging"""

from __future__ import print_function

import functools
import traceback
from typing import Any

from IPython.core.magic import Magics, line_magic, magics_class

from roc.logger import logger

from .brk import brk_cli
from .cont import cont_cli
from .roc import roc_cli
from .save import save_cli
from .state import state_cli
from .step import step_cli


def is_jupyter() -> bool:
    try:
        global get_ipython

        # jupyter environment defines a global function 'get_ipython'
        get_ipython()  # type: ignore # noqa: F821
        return True
    except Exception:
        return False


def magic_cli_decorator(cli):  # type: ignore
    def magic_cli_decorator(func):  # type: ignore
        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):  # type: ignore
            # Do something before
            # print("args", " ".join(args[1:]))
            # print("kwargs", kwargs)

            value = func(*args, **kwargs)

            # Do something after
            try:
                cli(args=args[1].split(), prog_name=func.__name__, standalone_mode=False)
            except Exception as e:
                print("ERROR:", e)  # noqa: T201
                print(traceback.format_exc())  # noqa: T201

            return value

        return wrapper_decorator

    return magic_cli_decorator


# The class MUST call this class decorator at creation time
@magics_class
class RocJupyterMagics(Magics):

    @line_magic
    @magic_cli_decorator(roc_cli)
    def roc(self, line: str) -> None:
        """Start executing roc, use '%roc --help' for more information"""
        pass

    @line_magic
    @magic_cli_decorator(brk_cli)
    def brk(self, line: str) -> None:
        """Halt execution of roc, use '%brk --help' for more information"""
        pass

    @line_magic
    @magic_cli_decorator(cont_cli)
    def cont(self, line: str) -> None:
        """Resume execution after calling break, use '%cont --help' for more information"""
        pass

    @line_magic
    @magic_cli_decorator(step_cli)
    def step(self, line: str) -> None:
        """Runs the ROC loop for <n> more steps and then breaks, use '%step --help' for more information"""
        pass

    @line_magic
    @magic_cli_decorator(state_cli)
    def state(self, line: str) -> None:
        """Displays information about the internal state of ROC, use '%state --help' for more information"""
        pass

    @line_magic
    @magic_cli_decorator(save_cli)
    def save(self, line: str) -> None:
        """Exports the graph to a file, use '%state --help' for more information"""
        pass

    @staticmethod
    def init() -> None:
        try:
            global get_ipython

            # jupyter environment defines a global function 'get_ipython'
            ip = get_ipython()  # type: ignore # noqa: F821
            logger.debug("jupyter environment found")
            load_ipython_extension(ip)
            logger.debug("jupyter magics loaded")
        except Exception:
            logger.debug("not running in ipython")


# In order to actually use these magics, you must register them with a
# running IPython.


def load_ipython_extension(ipython: Any) -> None:
    """Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(RocJupyterMagics)
