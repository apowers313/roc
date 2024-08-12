import click

import roc

from .step import add_step


@click.command()
@click.argument("num_steps", type=int, default=-1)
def roc_cli(num_steps: int) -> None:
    """Starts running ROC. Assumes that roc.init() has already been called
    (otherwise how are magics available?)"""

    roc.start()
    if num_steps >= 0:
        add_step(num_steps)
