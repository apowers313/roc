"""Helpers for translating original GoT dataset IDs to current Memgraph IDs.

The GoT cypherl export uses __mg_id__ starting from 0. When loaded into Memgraph,
the actual IDs may be offset if the internal counter has been incremented by prior
data loads. These helpers detect the offset and translate IDs.
"""

from roc.graphdb import GraphDB


def _detect_got_id_offset() -> tuple[int, int]:
    """Detect the offset between original GoT __mg_id__ values and current Memgraph IDs.

    Returns:
        Tuple of (node_offset, edge_offset).
    """
    db = GraphDB.singleton()

    # Waymar Royce had __mg_id__ = 0 in the original export
    results = list(
        db.raw_fetch("MATCH (n:Character) WHERE n.name = 'Waymar Royce' RETURN id(n) as nid")
    )
    if not results:
        return (0, 0)
    node_offset = results[0]["nid"] - 0  # __mg_id__ was 0

    # First edge in the cypherl is LOYAL_TO from Waymar Royce to Nights Watch
    # It had edge __mg_id__ = 0 in the original export
    results = list(
        db.raw_fetch(
            "MATCH (n:Character)-[e:LOYAL_TO]->(m:Allegiance) "
            "WHERE n.name = 'Waymar Royce' AND m.name = 'Nights Watch' "
            "RETURN id(e) as eid"
        )
    )
    if not results:
        return (node_offset, 0)
    edge_offset = results[0]["eid"] - 0  # first edge __mg_id__ was 0

    return (node_offset, edge_offset)


# Cached offsets
_node_offset: int | None = None
_edge_offset: int | None = None


def _ensure_offsets() -> tuple[int, int]:
    global _node_offset, _edge_offset
    if _node_offset is None:
        _node_offset, _edge_offset = _detect_got_id_offset()
    return _node_offset, _edge_offset  # type: ignore[return-value]


def got_node_id(original_id: int) -> int:
    """Translate an original GoT node ID to the current Memgraph ID."""
    node_off, _ = _ensure_offsets()
    return original_id + node_off


def got_edge_id(original_id: int) -> int:
    """Translate an original GoT edge ID to the current Memgraph ID."""
    _, edge_off = _ensure_offsets()
    return original_id + edge_off


def got_node_ids(original_ids: set[int]) -> set[int]:
    """Translate a set of original GoT node IDs to current Memgraph IDs."""
    node_off, _ = _ensure_offsets()
    return {i + node_off for i in original_ids}


def got_edge_ids(original_ids: set[int]) -> set[int]:
    """Translate a set of original GoT edge IDs to current Memgraph IDs."""
    _, edge_off = _ensure_offsets()
    return {i + edge_off for i in original_ids}
