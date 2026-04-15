"""Thread-private variable support for the `omp4py` runtime.

This module implements the OpenMP `threadprivate` mechanism, which allows
global variables to have independent copies per thread. Each thread
accesses its own instance of a variable, ensuring isolation between
concurrent executions.

The implementation maintains:
    - A global mapping from variable names to internal numeric identifiers
    - A per-thread storage area managed through the task context
      (`omp_ctx`)
    - Lazy initialization of thread-private storage when new variables
      are registered or accessed

Thread-private values are stored in a list of `TPrivRef` objects, where
each entry represents the value associated with a specific variable ID.

The system ensures thread safety during registration using a mutex and
supports dynamic growth of the per-thread storage structure.
"""

from __future__ import annotations

import copy

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.mutex import Mutex
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from omp4py.runtime.tasks.context import omp_ctx

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["TPrivRef", "copy_private", "map_privates", "threadprivate", "threadprivates", "update_privates"]

_threadprivate_ids_last: pyint
# END_CYTHON_IGNORE

_threadprivate_ids: dict[str, pyint] = {}
_threadprivate_ids_last = 0
_threadprivate_ids_mutex = Mutex.new()


class TPrivRef:
    """Container for a thread-private value reference.

    Each instance holds the value associated with a thread-private
    variable for a specific thread context.
    """

    v: object

    @staticmethod
    def new() -> TPrivRef:
        """Create an empty thread-private reference.

        Returns:
            TPrivRef: A new reference initialized with `None`.
        """
        obj: TPrivRef = TPrivRef.__new__(TPrivRef)
        obj.v = None
        return obj


def threadprivate(name: str, value: object) -> pyint:
    """Register or update a threadprivate variable.

    If the variable name is not yet registered, it is assigned a new
    internal identifier. If a value is provided, it is stored in the
    current thread's private storage.

    Args:
        name: Name of the variable.
        value: Initial value to assign (if not `None`).

    Returns:
        pyint: Internal identifier assigned to the variable.
    """
    i = _threadprivate_ids.get(name)
    if i is None:
        global _threadprivate_ids_last  # noqa: PLW0603
        _threadprivate_ids_mutex.lock()
        _threadprivate_ids[name] = i = _threadprivate_ids_last = len(_threadprivate_ids)
        _threadprivate_ids_mutex.unlock()
    if value is not None:
        update_privates(omp_ctx().tpvars)
        threadprivates(i).v = value

    return i


# BEGIN_CYTHON_IGNORE
def threadprivates(i: pyint) -> TPrivRef:
    """Access a thread-private value by its internal identifier.

    Args:
        i: Internal threadprivate identifier.

    Returns:
        TPrivRef: Reference to the thread-local value container.
    """
    tpvars = omp_ctx().tpvars
    if i >= len(tpvars):
        update_privates(tpvars)
    return tpvars[i]


# END_CYTHON_IGNORE


def update_privates(tpvars: list[TPrivRef]) -> None:
    """Ensure thread-private storage is large enough for all variables.

    Extends the per-thread storage list with new empty `TPrivRef`
    instances until it matches the number of registered variables.

    Args:
        tpvars: Thread-private storage list for the current thread.
    """
    for _ in range(_threadprivate_ids_last - len(tpvars) + 1):
        tpvars.append(TPrivRef.new())  # noqa: PERF401


def map_privates(names: tuple[str, ...]) -> pyint_array:
    """Map variable names to internal threadprivate identifiers.

    Args:
        names: Tuple of variable names.

    Returns:
        pyint_array: Array of internal identifiers (-1 if not found).
    """
    ids = new_pyint_array(len(names))
    i: pyint
    for i in range(len(ids)):
        ids[i] = _threadprivate_ids.get(names[i], -1)
    return ids


def copy_private(obj: object) -> object:
    """Create a copy of a threadprivate value.

    This function currently uses shallow copying and serves as the
    default copy strategy for threadprivate variables.

    Args:
        obj: Object to copy.

    Returns:
        object: Shallow copy of the input object.
    """
    return copy.copy(obj)  # TODO: replicate declare reduction copy strategy
