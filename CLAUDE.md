# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROC (Reinforcement Learning of Concepts) is a Python agent system that plays NetHack using a component-based, event-driven architecture. The agent perceives the game environment, identifies objects, tracks changes between frames, and decides actions.

## Common Commands

```bash
# Setup
make setup              # Install deps + pre-commit hooks
make install            # Create venv (Python 3.13) and sync deps

# Testing
make test               # Run all tests (excludes slow and requires_observability)
uv run pytest -c pyproject.toml tests/object_test.py                    # Run one test file
uv run pytest -c pyproject.toml tests/object_test.py -k test_name      # Run single test by name
uv run pytest -c pyproject.toml tests/object_test.py::TestClass::test   # Run specific test method

# Linting & Formatting
make lint               # Run mypy + check-codestyle + deptry
make check-codestyle    # Run ruff + black
make mypy               # Type checking only
make codestyle          # Auto-format with black

# Coverage
make coverage           # Run tests with coverage (90% minimum threshold)

# Docs
make docs               # Build mkdocs
make edit-docs          # Serve docs locally on 0.0.0.0:9000
```

## Architecture

### Component System

Everything is built on `Component` (component.py) -- an abstract base class with:
- **Auto-registration**: Subclasses register themselves in a global `component_registry` via `__init_subclass__`. Each component has a unique `name`+`type` pair.
- **Auto-loading**: Components with `auto = True` are loaded at startup. Perception components are loaded from config.
- **Bus connections**: Components communicate via `connect_bus()` which attaches them to typed EventBus channels.

### Event System

Communication uses RxPy reactive streams (event.py):
- `EventBus[T]` -- typed communication channel (backed by `rx.Subject`)
- `Event[T]` -- carries data + source ComponentId
- `BusConnection[T]` -- component-to-bus link for sending/listening
- Listeners run on a ThreadPoolScheduler; events are filtered to exclude self-sends by default

### Data Pipeline

The agent processes each game step through this chain:

```
Gymnasium (NetHack env) -> Perception (feature extractors) -> Attention (saliency)
-> ObjectResolver (identify objects) -> Sequencer (build frames)
-> Transformer (detect changes) -> Action (decide action)
```

Each stage is a Component that listens on one EventBus and sends on the next.

### Graph Database

`graphdb.py` wraps Memgraph (via mgclient) with:
- `Node` and `Edge` classes built on Pydantic models with auto-save-on-modify
- LRU caching to reduce DB queries
- `NodeList` and `EdgeList` collection types with filtering/selection
- Cypher query generation for CRUD operations

CI requires a Memgraph service container. Tests that need the DB will fail without it.

### Key Abstractions

- **Transformable** (transformable.py): Interface for objects that can detect and represent changes between frames. Implemented by objects, intrinsics, etc.
- **Transform** (transformable.py): A Node subclass representing the diff between two frame states.
- **Intrinsic** (intrinsic.py): Agent internal state (HP, energy, hunger) with normalization operations, converted to graph nodes.
- **Frame** (sequencer.py): A snapshot of the game state at one timestep, containing objects, intrinsics, and actions.
- **Perception** (perception.py): Abstract base for feature extractors. Concrete implementations live in `roc/feature_extractors/`.

### Config

`Config` (config.py) uses pydantic-settings, reads from `.env` file with `roc_` prefix. During pytest, env vars and .env file are deliberately ignored to isolate tests.

## Code Style

- **Line length**: 100 characters (black + ruff)
- **Type checking**: Strict mypy with pydantic and numpy plugins
- **Docstrings**: Google convention (enforced by ruff). Not required in tests or magic methods.
- **Python version**: 3.13
- **Package manager**: uv (not pip, not poetry)

## Testing Notes

- Tests use fixtures from `conftest.py` for clearing caches, restoring registries, and creating test components
- `clear_cache` fixture resets graphdb Node/Edge caches between tests
- `restore_registries` fixture saves and restores the component registry
- Pytest runs with `--doctest-modules` so docstrings with `>>>` examples are tested
- Test markers: `@pytest.mark.slow`, `@pytest.mark.requires_observability`, `@pytest.mark.requires_graphviz`
