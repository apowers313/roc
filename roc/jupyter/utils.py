from typing import Any


def get_symbol(s: str) -> Any:
    ip = get_ipython()  # type: ignore

    if s not in ip.user_ns:
        print(f"ERROR: symbol '{s}' not found in iPython shell")  # noqa: T201
        return

    return ip.user_ns[s]
