import click

from roc.game.breakpoint import breakpoints


@click.command()
def cont_cli() -> None:
    breakpoints.resume()
