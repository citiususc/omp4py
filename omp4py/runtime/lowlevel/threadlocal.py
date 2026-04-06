"""Thread-local storage utilities used by the `omp4py` runtime.

This module provides a minimal abstraction over thread-local storage used
internally by the runtime. It exposes a simple interface to store and
retrieve a value associated with the current thread.

The default implementation relies on Python's `threading.local`. When
`omp4py` is compiled with Cython, this module is optimized by providing
a more efficient low-level mechanism defined in the `.pxd`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cython

if cython.compiled:
    from cython.cimports.omp4py.runtime.lowlevel.threadlocal0 import omp4py_local0_ptr  # type:ignore[unresolved-import]

    threadlocal_link = omp4py_local0_ptr

# BEGIN_CYTHON_IGNORE
__all__ = ["threadlocal_get", "threadlocal_init", "threadlocal_set"]
# END_CYTHON_IGNORE
_local: threading.local = threading.local()


def threadlocal_default_set(value: object) -> None:
    """Store a value for the current thread.

    Args:
        value (object): Value to store.
    """
    _local.value = value


# BEGIN_CYTHON_IGNORE


def threadlocal_set(value: object) -> None:
    """Store a value for the current thread.

    Args:
        value (object): Value to store.
    """
    threadlocal_default_set(value)


def threadlocal_get() -> object | None:
    """Return the value associated with the current thread.

    If no value is currently stored, the thread-local storage is initialized
    by calling `threadlocal_init`.

    Returns:
        object | None: Stored value, or `None` if not set.
    """
    value: object = getattr(_local, "value", None)
    if value is None:
        threadlocal_init()
        return getattr(_local, "value", None)
    return value


# END_CYTHON_IGNORE


def threadlocal_init_none() -> None:
    """Default thread-local initializer that sets no initial value."""
    msg = "threadlocal_init_none() should never be called"
    raise ValueError(msg)


# BEGIN_CYTHON_IGNORE
threadlocal_init: Callable[[], None]
# END_CYTHON_IGNORE
threadlocal_init = threadlocal_init_none
