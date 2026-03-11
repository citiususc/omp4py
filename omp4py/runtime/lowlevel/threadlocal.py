"""Thread-local storage utilities used by the `omp4py` runtime.

This module provides a minimal abstraction over thread-local storage used
internally by the runtime. It exposes a simple interface to store and
retrieve a value associated with the current thread.

The default implementation relies on Python's `threading.local`. When
`omp4py` is compiled with Cython, this module may be replaced by a
compiled implementation providing the same interface but using a more
efficient low-level mechanism.
"""
import threading

import cython

if cython.compiled:
    import cython.imports.omp4py.runtime.lowlevel.threadlocal0  # type:ignore[unresolved-import]


__all__ = ["ThreadLocal"]

_local: threading.local = threading.local()


class ThreadLocal:
    """Utility class providing access to thread-local storage."""

    @staticmethod
    def default_set(value: object) -> None:
        """Store a value for the current thread.

        Args:
            value (object): Value to store.
        """
        _local.value = value

    @staticmethod
    def set(value: object) -> None:
        """Store a value for the current thread.

        Args:
            value (object): Value to store.
        """
        ThreadLocal.default_set(value)

    @staticmethod
    def get() -> object | None:
        """Return the value associated with the current thread.

        Returns:
            object | None: Stored value, or `None` if not set.
        """
        return getattr(_local, "value", None)
