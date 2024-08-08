import click

from roc.breakpoint import breakpoints
from roc.logger import logger

step_cnt = 0

@click.command()
@click.argument("num_steps", type=int, default=1)
def step_cli(num_steps: int) -> None:
    add_step(num_steps)

def add_step(num_steps: int = 1) -> None:
    global step_cnt

    step_cnt = num_steps
    logger.info(f"stepping {step_cnt} times...")

    breakpoints.add(do_step, name="step")
    breakpoints.resume(quiet=True)

def do_step() -> bool:
    global step_cnt

    step_cnt -= 1
    if step_cnt <= 0:
        breakpoints.remove("step")
        return True

    return False