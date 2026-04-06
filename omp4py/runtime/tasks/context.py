"""Thread-local task context management for the `omp4py` runtime.

This module defines the high-level thread-local context used by the
`omp4py` runtime to store per-thread execution state.

The context is represented by the `TaskContext` class, which encapsulates
all runtime information (e.g., ICVs). A thread-local instance of this
context is created on demand using the low-level thread-local
infrastructure.

The `context_init` function is registered as the initialization routine
for the underlying thread-local storage. It creates a new `TaskContext`
instance, initializes its internal state, and stores it in thread-local
storage.

Other modules must use `omp_ctx()` to safely access the current thread's
context. This ensures proper initialization and avoids direct interaction
with the low-level thread-local API.
"""

from __future__ import annotations

import typing

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.icvs.icvs import Data, defaults
from omp4py.runtime.lowlevel import threadlocal

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["TaskContext", "omp_ctx"]
# END_CYTHON_IGNORE


class TaskContext:
    """Container for per-thread runtime state.

    Attributes:
        icvs (Data): Internal control variables for the runtime.
    """

    icvs: Data


def context_init() -> None:
    """Initialize the thread-local task context.

    This function is used as the initialization hook for the underlying
    thread-local storage. It creates a new `TaskContext` instance,
    initializes it with default internal control variables, and stores it
    in thread-local storage.
    """
    ctx: TaskContext = TaskContext.__new__(TaskContext)
    ctx.icvs = defaults.copy()
    threadlocal.threadlocal_set(ctx)


threadlocal.threadlocal_init = context_init


# BEGIN_CYTHON_IGNORE
def omp_ctx() -> TaskContext:
    """Return the current thread's task context.

    This function provides safe access to the thread-local `TaskContext`.
    It ensures that the context has been properly initialized before
    being returned.

    Returns:
        TaskContext: The current thread's context instance.
    """
    return typing.cast("TaskContext", threadlocal.threadlocal_get())


# END_CYTHON_IGNORE
