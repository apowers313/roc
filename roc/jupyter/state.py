import click

from ..reporting.state import all_states, print_state, states


@click.command()
@click.argument(
    "var",
    nargs=-1,
    type=click.Choice(all_states, case_sensitive=False),
)
def state_cli(var: list[str]) -> None:
    if var is None or len(var) < 1:
        # if no state is specified, print a selection of the most interesting states
        print_state()
        return

    for v in var:
        s = getattr(states, v)
        print(str(s))  # noqa: T201
