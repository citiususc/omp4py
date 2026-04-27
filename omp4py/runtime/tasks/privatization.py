"""Variable privatization helpers.

This module provides small runtime utilities used to create, initialize,
and duplicate variables that require independent storage during OpenMP
execution.

Many OpenMP constructs need per-thread or per-task values derived from an
existing variable. Depending on the construct, a new default instance may
be required or the original value may need to be copied into isolated
storage.

These helpers are used by generated runtime code to implement those data
management semantics during parallel execution.
"""

from __future__ import annotations

import copy


def new_var(orig: object) -> object:
    """Create a new private variable instance.

    A new object is created using the original value type constructor.

    Args:
        orig (object): Original variable used as type reference.

    Returns:
        object: New initialized instance.
    """
    return type(orig)()


def copy_var(orig: object) -> object:
    """Create a copied private variable value.

    Performs a shallow copy of the original value. Used to implement
    `firstprivate` like semantics.

    Args:
        orig (object): Original variable value.

    Returns:
        object: Copied value.
    """
    return copy.copy(orig)
