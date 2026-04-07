# Graph Database (roc/db)

## Why This Design

This module is a standalone graph ORM over Memgraph. Its purpose: consumers should
never write raw Cypher or manually manage persistence. Creating a Node or Edge
automatically handles caching, ID assignment, and eventual database writes. The
syntactic sugar (NodeList, EdgeList, select, walk, neighborhood) exists because
hard-won experience showed that transparent graph traversal without manual lookups
is essential for usability.

This module is designed to be ROC-independent. It must not reference ROC domain
concepts (objects, features, frames, attention) or import from ROC pipeline code.
Current imports of Config, logger, and Observability from the framework are known
coupling points to be abstracted during future extraction to a standalone package.

The graph structure is the most critical data in the system -- it drives concept
formation. Changes to graph schema (Node subclasses, Edge subclasses,
allowed_connections) require explicit user approval. Do not add, remove, or modify
schema elements without asking first.

## Key Decisions

- **Save-on-GC via `__del__`, not save-on-modify** -- Nodes and Edges save
  themselves when garbage collected. There is no `__setattr__` hook. Objects live in
  cache until evicted or collected, and writes batch naturally by GC timing. The
  tradeoff: `__del__` during interpreter shutdown can reference already-collected
  objects (the GraphDB connection, other nodes), producing
  `ErrorSavingDuringDelWarning`. This is a known Python `__del__` ordering limitation
  during shutdown, not a bug.

- **Negative IDs for unsaved objects** -- New objects get decrementing negative IDs
  (-1, -2, ...). Positive IDs come from Memgraph on persist. On save, the cache entry
  moves from negative to positive key and all edge references update. This allows full
  in-memory graph construction before any DB writes.

- **Cache sized to server RAM, not working set** -- Defaults are `2**30` (~1 billion
  entries), sized to this server's 92GB RAM. DB writes are expensive and graph usage is
  heavy, so the goal is zero eviction during normal operation. Small caches thrash
  catastrophically -- the cache logs a warning when a miss occurs at full capacity. If
  you see that warning, performance is already severely degraded.

- **`allowed_connections` as compile-time schema** -- Edge subclasses declare which
  `(src_class_name, dst_class_name)` pairs are valid. Checked at edge creation time,
  not after. `db_strict_schema=True` by default, so missing `allowed_connections`
  raises. Schema validation also runs at `GraphDB.__init__` to catch edges referencing
  unregistered Node types early.

- **Lazy-loading collections** -- NodeList and EdgeList store only IDs and fetch
  objects from cache on iteration. Memory stays proportional to the ID list, not the
  full object graph. Consequence: iteration triggers per-element cache lookups.

## Invariants

- **Node class name AND label frozenset must both be globally unique.**
  `__init_subclass__` enforces at class definition time. Collision crashes at import
  with `ValueError`. Labels auto-generate from the MRO (class `Foo(Bar(Node))` gets
  labels `{"Foo", "Bar"}`).

- **Edge connections must satisfy `allowed_connections`.** `Edge.connect()` checks
  against both the node's class name and all parent Node names (inheritance-aware).
  Violating this raises `ValueError`. Update `allowed_connections` on the Edge
  subclass first, then use the new connection pattern.

- **Delete edges before nodes.** `Node.delete()` handles this correctly. Deleting a
  node without cleaning up its edges leaves dangling references -- those edges' `src`
  or `dst` properties will raise `NodeNotFound` on access, and `__del__` may try to
  save edges pointing at nonexistent nodes.

- **`_no_save` must be set before deletion completes.** `Node.delete()` and
  `Edge.delete()` set `_no_save=True` to prevent `__del__` from re-persisting a
  deleted object. Forgetting this causes the GC to write the deleted object back.

## Non-Obvious Behavior

- **Creating an Edge auto-saves its endpoints.** `Edge.create()` calls `Node.save()`
  on src and dst if they are still new (negative ID). This cascading save ensures
  Memgraph has both nodes before the edge references them by positive ID.

- **Schema validation is inheritance-aware.** An `allowed_connections` entry like
  `("FeatureNode", "Object")` permits any FeatureNode subclass as source, because
  `_check_schema` collects all parent Node class names and checks against the full
  set.

- **`__del__` save failures during shutdown are expected.** Python's non-deterministic
  GC ordering at shutdown means a Node's `__del__` may fire after the GraphDB
  connection closes. This produces `ErrorSavingDuringDelWarning`, not an error. It
  does not indicate data loss if `flush()` was called before shutdown.

- **Cache eviction triggers persistence.** When the LRU cache is full and a new object
  is added, the evicted object's `__del__` fires, triggering a save. In tight loops
  creating many objects, this means every new object may cause a synchronous DB write
  for the evicted one.

## Anti-Patterns

- **Do not create many temporary objects without `_no_save=True`.** In loops, cache
  eviction triggers `__del__` saves on evicted objects. Set `_no_save=True` on objects
  that should never be persisted (temps, test fixtures).

- **Do not delete nodes by removing them from cache manually.** Always use
  `Node.delete()` / `Edge.delete()`. These handle edge cleanup, cache removal, DB
  deletion, and the `_no_save` flag. Manual cache manipulation leaves orphaned edges.

- **Do not create Edge subclasses without `allowed_connections`.** In strict schema
  mode (the default), this raises an error at connection time. Missing schemas allow
  invalid graph structures that are very hard to diagnose after the fact.

- **Do not add ROC-specific concepts to this module.** No imports from pipeline,
  perception, or game packages. Domain-agnostic for future standalone extraction.

- **Do not bypass `Edge.connect()` to create edges.** Direct `Edge()` construction
  skips schema validation. Always use `Edge.connect(src, dst)` or
  `Node.connect(src, dst, type)`.
