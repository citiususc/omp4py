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
from omp4py.runtime.tasks.context import omp_ctx

if typing.TYPE_CHECKING:
    from omp4py.runtime.tasks.task import Barrier

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["barrier"]
# END_CYTHON_IGNORE



def barrier() -> None:
    """Synchronize all threads in the current team at a barrier point.

    This function implements the OpenMP barrier construct. Each thread
    blocks at the barrier until all threads in the team have reached the
    same synchronization point.

    During the waiting phase, a thread may either idle spin or block,
    depending on the runtime conditions, and may also execute available
    tasks if the runtime scheduler provides work, improving overall
    utilization.

    Returns:
        None
    """
    barrier_ : Barrier = omp_ctx().task.barrier

    while not barrier_.wait():
        pass # here a thread can consume tasks
