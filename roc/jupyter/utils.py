from typing import Any


def get_symbol(s: str) -> Any:
    ip = get_ipython()  # type: ignore

    if s not in ip.user_ns:
        print(f"ERROR: symbol '{s}' not found in iPython shell")  # noqa: T201
        return

    return ip.user_ns[s]


def bytes2human(n: int) -> str:
    # stolen from: https://psutil.readthedocs.io/en/latest/#recipes
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return "{:.1f}{}B".format(value, s)
    return "%sB" % n
