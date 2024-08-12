import click

from roc.breakpoint import breakpoints

from .utils import get_symbol


@click.group(invoke_without_command=True)
@click.pass_context
def brk_cli(ctx: click.Context) -> None:
    """Controls breakpoints for ROC"""
    if ctx.invoked_subcommand is None:
        breakpoints.do_break()


@brk_cli.command()
def list() -> None:
    """List all breakpoints"""
    breakpoints.list()


@brk_cli.command()
@click.argument("name")
def remove(name: str) -> None:
    """Remove a breakpoint"""
    breakpoints.remove(name)


@brk_cli.command()
@click.argument("function")
def add(function: str) -> None:
    """Add a breakpoints by name or function"""

    sym = get_symbol(function)

    breakpoints.add(sym, name=function, src="<iPython>")
    print(f"Added breakpoint: '{function}'")  # noqa: T201


@brk_cli.command()
def clear() -> None:
    """Remove all breakpoints"""
    breakpoints.clear()
