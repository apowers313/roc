import json
import time
from datetime import datetime, timedelta

import click
import networkx as nx
import scipy as sp
from networkx.drawing.nx_pydot import write_dot
from tqdm import tqdm

from roc.graphdb import Edge, GraphDB, Node


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
    ids = Node.all_ids()
    print(f"Saving {len(ids)} nodes...")  # noqa: T201
    start_time = time.time()

    # tqdm options: https://github.com/tqdm/tqdm?tab=readme-ov-file#parameters
    with tqdm(total=len(ids), desc="Nodes", unit="node", ncols=80, colour="blue") as pbar:

        def progress_update(n: Node) -> bool:
            pbar.update(1)
            return True

        G = GraphDB.to_networkx(node_ids=ids, filter=progress_update)

    # format timestamp
    if timestamp:
        # time format: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
        timestr = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        filename = f"{filename}-{timestr}"

    print(f"Writing graph to '{filename}'...")  # noqa: T201
    match format:
        case "gexf":
            nx.write_gexf(G, f"{filename}.gexf")
        case "gml":
            nx.write_gml(G, f"{filename}.gml")
        case "dot":
            # XXX: pydot uses the 'name' attribute internally, so rename ours if it exists
            for n in G.nodes(data=True):
                if "name" in n[1]:
                    n[1]["nme"] = n[1]["name"]
                    del n[1]["name"]
            write_dot(G, f"{filename}.dot")
        case "graphml":
            nx.write_graphml(G, f"{filename}.graphml")
        # case "json-tree":
        #     with open(f"{filename}.tree.json", "w", encoding="utf8") as f:
        #         json.dump(nx.tree_data(G), f)
        case "json-node-link":
            with open(f"{filename}.node-link.json", "w", encoding="utf8") as f:
                json.dump(nx.node_link_data(G), f)
        case "json-adj":
            with open(f"{filename}.adj.json", "w", encoding="utf8") as f:
                json.dump(nx.adjacency_data(G), f)
        case "cytoscape":
            with open(f"{filename}.cytoscape.json", "w", encoding="utf8") as f:
                json.dump(nx.cytoscape_data(G), f)
        case "pajek":
            nx.write_pajek(G, f"{filename}.pajek")
        case "matrix-market":
            np_graph = nx.to_numpy_array(G)
            sp.io.mmwrite(f"{filename}.mm", np_graph)
        case "adj-list":
            nx.write_adjlist(G, f"{filename}.adjlist")
        case "multi-adj-list":
            nx.write_multiline_adjlist(G, f"{filename}.madjlist")
        case "edge-list":
            nx.write_edgelist(G, f"{filename}.edges")

    end_time = time.time()

    nc = Node.get_cache()
    ec = Edge.get_cache()
    assert len(nc) == len(ids)
    print(  # noqa: T201
        f"Saved {len(ids)} nodes and {len(ec)} edges. Elapsed time: {timedelta(seconds=(end_time-start_time))}"
    )
