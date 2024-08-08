import click

from roc.breakpoint import breakpoints


@click.command()
def cont_cli() -> None:
    breakpoints.resume()