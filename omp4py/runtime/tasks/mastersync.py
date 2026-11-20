"""Barrier synchronization construct for the `omp4py` runtime.

This module implements the OpenMP barrier construct, which forces all
threads in a team to synchronize at a common point before continuing
execution.

During the waiting phase, threads may opportunistically execute other
available runtime tasks if they exist, allowing useful work to be done
instead of idle spin or block until all threads reach the barrier.
"""

from __future__ import annotations

import typing

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.mutex import Mutex
from omp4py.runtime.tasks.context import omp_ctx

if typing.TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import pyint
    from omp4py.runtime.tasks.task import Barrier

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = [
    "barrier",
    "critical_lock",
    "critical_named",
    "critical_named_lock",
    "critical_named_unlock",
    "critical_unlock",
    "masked",
    "team_lock",
    "team_unlock",
]
# END_CYTHON_IGNORE

_critical_ids: dict[str, Mutex] = {}
_critical_ids_last = 0
_critical_mutex = Mutex.new()
_critical_unnamed = Mutex.new()


def barrier() -> None:
    """Synchronize all threads in the current team at a barrier point.

    This function implements the OpenMP barrier construct. Each thread
    blocks at the barrier until all threads in the team have reached the
    same synchronization point.

    During the waiting phase, a thread may either idle spin or block,
    depending on the runtime conditions, and may also execute available
    tasks if the runtime scheduler provides work, improving overall
    utilization.
    """
    barrier_: Barrier = omp_ctx().task.barrier

    while not barrier_.wait():
        pass  # here a thread can consume tasks


def masked(select: pyint) -> bool:
    """Determine whether the current thread matches the specified thread ID.

    Args:
        select (pyint): The target thread number to check against.

    Returns:
        bool: True if the current thread number matches the selected ID, False otherwise.
    """
    return omp_ctx().icvs.thread_num == select


def critical_named(name: str) -> Mutex:
    """Get or create a named mutex.

    If the mutex name is not yet registered, a new mutex is created
    and stored in the critical storage.

    Args:
        name (str): The name of the mutex.

    Returns:
        Mutex: The mutex assigned to the specified name.
    """
    m: Mutex | None = _critical_ids.get(name)
    if m is None:
        _critical_mutex.lock()
        _critical_ids[name] = m = Mutex.new()
        _critical_mutex.unlock()
    return m


def critical_named_lock(m: Mutex) -> None:
    """Acquire the named critical section lock.

    Args:
        m (Mutex): The named mutex.
    """
    m.lock()


def critical_named_unlock(m: Mutex) -> None:
    """Release the named critical section lock.

    Args:
        m (Mutex): The named mutex.
    """
    m.unlock()


def critical_lock() -> None:
    """Acquire the unnamed critical section lock."""
    _critical_unnamed.lock()


def critical_unlock() -> None:
    """Release the unnamed critical section lock."""
    _critical_unnamed.unlock()


def team_lock() -> None:
    """Acquire the thread team critical lock."""
    _critical_unnamed.lock() # TODO: create a lock in task


def team_unlock() -> None:
    """Release the thread team critical lock."""
    _critical_unnamed.unlock()
