"""Concrete ExpMod implementations.

Each subpackage holds implementations for one ``modtype``. Importing this package
pulls them all in so their ``__init_subclass__`` hooks register every class into
``expmod_registry``. The abstract base classes (one per modtype) remain in the
pipeline module they serve (e.g. ``roc.pipeline.action.DefaultActionExpMod``).
"""

from roc.expmods import (
    action,
    object_resolution,
    prediction_candidate,
    saliency_attenuation,
)

__all__ = [
    "action",
    "object_resolution",
    "prediction_candidate",
    "saliency_attenuation",
]
