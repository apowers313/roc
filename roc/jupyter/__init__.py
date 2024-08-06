"""Jupyter magics for flow control and debugging"""

from __future__ import print_function

from typing import Any

from IPython.core.magic import Magics, cell_magic, line_cell_magic, line_magic, magics_class


# The class MUST call this class decorator at creation time
@magics_class
class RocJupyterMagics(Magics):

    @line_magic
    def lmagic(self: Any, line: str) -> str:
        "my line magic"
        print("Full access to the main IPython object:", self.shell)
        print("Variables in the user namespace:", list(self.shell.user_ns.keys()))
        return line

    @cell_magic
    def cmagic(self, line: str, cell: str) -> tuple[str, str]:
        "my cell magic"
        return line, cell

    @line_cell_magic
    def lcmagic(self, line: str, cell: str|None=None) -> str | tuple[str, str]:
        "Magic that works both as %lcmagic and as %%lcmagic"
        if cell is None:
            print("Called as line magic")
            return line
        else:
            print("Called as cell magic")
            return line, cell

    @staticmethod
    def init() -> None:
        try:
            global get_ipython

            # jupyter environment defines a global function 'get_ipython'
            ip = get_ipython() # type: ignore # noqa: F821
            print("jupyter environment found")
            load_ipython_extension(ip)
            print("jupyter magics loaded")
        except Exception:
            print("not running in ipython")


# In order to actually use these magics, you must register them with a
# running IPython.

def load_ipython_extension(ipython: Any) -> None:
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(RocJupyterMagics)