import click

from ..reporting.state import State


@click.command()
@click.argument(
    "var",
    nargs=-1,
    type=click.Choice(State.get_state_names(), case_sensitive=False),
)
def state_cli(var: list[str]) -> None:
    if var is None or len(var) < 1:
        # if no state is specified, print a selection of the most interesting states
        State.print()
        return

    for v in var:
        s = getattr(State.get_states(), v)
        print(str(s))  # noqa: T201
