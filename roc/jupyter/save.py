import click

from roc.graphdb import GraphDB


@click.command()
@click.option(
    "-f",
    "--format",
    default="gexf",
    type=click.Choice(
        [
            "gexf",
            "gml",
            "dot",
            "graphml",
            "json-node-link",
            "json-adj",
            "cytoscape",
            "pajek",
            "matrix-market",
            "adj-list",
            "multi-adj-list",
            "edge-list",
        ],
        case_sensitive=False,
    ),
)
@click.option("--timestamp/--no-timestamp", is_flag=True, default=True)
@click.argument("filename", nargs=1, type=click.Path(), default="graph", required=False)
def save_cli(format: str, filename: str, timestamp: bool) -> None:
    GraphDB.export(format, filename, timestamp)
