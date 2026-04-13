# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
"""Thread team management routines for the `omp4py` api.

This module provides functions to manage and query thread teams
within OpenMP parallel regions. These routines correspond to the
OpenMP Thread Team Routines and allow inspection and control of
the threads participating in the current contention group.
"""

from __future__ import annotations

import sys
import typing
from warnings import deprecated

import cython

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from omp4py.runtime.tasks.context import omp_ctx

if typing.TYPE_CHECKING:
    from omp4py.runtime.icvs import RunSched

# END_CYTHON_IMPORTS

__all__ = [
    "omp_get_active_level",
    "omp_get_ancestor_thread_num",
    "omp_get_cancellation",
    "omp_get_dynamic",
    "omp_get_level",
    "omp_get_max_active_levels",
    "omp_get_max_threads",
    "omp_get_nested",
    "omp_get_num_threads",
    "omp_get_schedule",
    "omp_get_team_size",
    "omp_get_thread_limit",
    "omp_get_thread_num",
    "omp_in_parallel",
    "omp_sched_auto",
    "omp_sched_dynamic",
    "omp_sched_guided",
    "omp_sched_monotonic",
    "omp_sched_static",
    "omp_sched_t",
    "omp_set_dynamic",
    "omp_set_max_active_levels",
    "omp_set_nested",
    "omp_set_num_threads",
    "omp_set_schedule",
]


def omp_set_num_threads(num_threads: pyint) -> None:
    """Set the number of threads to use for subsequent parallel regions.

    Affects the number of threads used for subsequent parallel regions not specifying a num_threads clause, by
    setting the value of the first element of the `nthreads-var` ICV of the current task to num_threads.

    Args:
        num_threads (pyint): Desired number of threads.
    """
    omp_ctx().icvs.nthreads[0] = num_threads


def omp_get_num_threads() -> pyint:
    """Returns the number of threads in the current team.

    The binding region for an `omp_get_num_threads` region is the innermost enclosing parallel region.

    Returns:
        pyint: Number of threads in the current parallel region.
    """
    return omp_ctx().icvs.team_size


def omp_get_max_threads() -> pyint:
    """Returns the maximum number of threads available for parallel regions.

    Returns an upper bound on the number of threads that could be used to form a new team if a parallel construct
    without a num_threads clause were encountered after execution returns from this routine.

    Returns:
        pyint: Maximum threads allowed.
    """
    return omp_ctx().icvs.nthreads[0]


def omp_get_thread_num() -> pyint:
    """Returns the thread number of the calling thread within the current team.

    Returns:
        pyint: Zero-based thread number.
    """
    return omp_ctx().icvs.thread_num


def omp_in_parallel() -> bool:
    """Check if the current context is inside a parallel region.

    Returns true if the `active-levels-var` ICV is greater than zero; otherwise it returns false.

    Returns:
        bool: True if inside a parallel region, False otherwise.
    """
    return omp_ctx().icvs.active_levels > 0


def omp_set_dynamic(dynamic_threads: bool) -> None:
    """Enable or disable dynamic adjustment of the number of threads.

    Enables or disables dynamic adjustment of the number of threads available for the execution of subsequent parallel
    regions by setting the value of the `dyn-var` ICV.

    Args:
        dynamic_threads (bool): True to allow dynamic threads, False to disable.
    """
    omp_ctx().icvs.dyn = dynamic_threads


def omp_get_dynamic() -> bool:
    """Check if dynamic adjustment of threads is enabled.

    This routine returns the value of the dyn-var ICV, which is true if dynamic adjustment of the number of threads is
    enabled for the current task.

    Returns:
        bool: True if dynamic threads are enabled, False otherwise.
    """
    return omp_ctx().icvs.dyn


def omp_get_cancellation() -> bool:
    """Check if cancellation points are enabled globally.

    Returns the value of the `cancel-var` ICV, which is true if cancellation is activated; otherwise it returns false

    Returns:
        bool: True if OpenMP cancellation is enabled.
    """
    return omp_ctx().icvs.global_vars.cancel


@deprecated("Use omp_set_max_active_levels instead")
def omp_set_nested(nested: bool) -> None:
    """Enables or disables nested parallelism, by setting the `nest-var` ICV. (deprecated).

    Args:
        nested (bool): True to enable nested parallelism, False to disable.
    """
    omp_ctx().icvs.device_vars.max_active_levels = 2**31 if nested else 1


@deprecated("Use omp_get_max_active_levels instead")
def omp_get_nested() -> bool:
    """Check if nested parallelism is enabled (deprecated).

    Returns the value of the `nest-var` ICV, which indicates if nested parallelism is enabled or disabled.

    Returns:
        bool: True if nested parallelism is enabled.
    """
    return omp_ctx().icvs.device_vars.max_active_levels > 1


# OpenMP scheduling types
# BEGIN_CYTHON_IGNORE: conflict with pxd declarations
type omp_sched_t = pyint  # noqa: PYI042

omp_sched_static: omp_sched_t
omp_sched_dynamic: omp_sched_t
omp_sched_guided: omp_sched_t
omp_sched_auto: omp_sched_t
omp_sched_monotonic: omp_sched_t

_omp_sched2run: pyint_array
_run2omp_sched: pyint_array
# END_CYTHON_IGNORE


def omp_set_schedule(kind: omp_sched_t, chunk_size: pyint) -> None:
    """Set the schedule type and chunk size for loop scheduling.

    Affects the schedule that is applied when runtime is used as schedule kind, by setting the value of the
    `run-sched-var` ICV.

    Args:
        kind (omp_sched_t): Scheduling policy (static, dynamic, guided, auto).
        chunk_size (pyint): Number of loop iterations per chunk. Use -1 to reset.
    """
    typekind = kind & 0x3
    if typekind < omp_sched_static or typekind > omp_sched_auto:
        return
    run_sched: RunSched = omp_ctx().icvs.run_sched
    run_sched.chunksize = chunk_size if chunk_size > 0 else -1
    run_sched.type = _omp_sched2run[typekind]
    run_sched.monotonic = typekind != kind


def omp_get_schedule() -> tuple[omp_sched_t, pyint]:
    """Returns the current loop schedule type and chunk size.

    Returns the value of `run-sched-var` ICV, which is the schedule applied when runtime schedule is used.

    Returns:
        tuple[omp_sched_t, pyint]: A tuple containing the scheduling policy and chunk size.
    """
    run_sched: RunSched = omp_ctx().icvs.run_sched
    kind: omp_sched_t = _run2omp_sched[run_sched.type - _run2omp_sched_offset]
    if run_sched.monotonic:
        kind |= omp_sched_monotonic
    return kind, run_sched.chunksize


def omp_get_thread_limit() -> pyint:
    """Returns the maximum number of threads available to the program.

    Returns the value of the `thread-limit-var` ICV, which is the maximum number of OpenMP threads available.

    Returns:
        pyint: Thread limit.
    """
    return omp_ctx().icvs.thread_limit


def omp_set_max_active_levels(max_active_levels: pyint) -> None:
    """Set the maximum number of nested active parallel levels.

    Limits the number of nested active parallel regions, by setting `max-active-levels-var` ICV.

    Args:
        max_active_levels (pyint): Maximum allowed active levels.
    """
    if max_active_levels >= 0:
        omp_ctx().icvs.device_vars.max_active_levels = max_active_levels


def omp_get_max_active_levels() -> pyint:
    """Returns the maximum number of nested active parallel levels.

    Returns the value of `max-active-levels-var` ICV, which determines the maximum number of nested active parallel
    regions.

    Returns:
        pyint: Maximum active levels.
    """
    return omp_ctx().icvs.device_vars.max_active_levels


def omp_get_level() -> pyint:
    """Returns the current nested parallel level.

    For the enclosing device region, returns the `levels-vars` ICV, which is the number of nested parallel regions that
    enclose the task containing the call.

    Returns:
        pyint: Level of the current parallel region.
    """
    return omp_ctx().icvs.levels


def omp_get_ancestor_thread_num(level: pyint) -> pyint:
    """Returns the thread number of the ancestor thread at the current level.

    Returns, for a given nested level of the current thread, the thread number of the ancestor of the current thread.

    The levels are defined as:
        - Level 0: the initial (master) thread
        - Level 1: the first parallel region
        - Level N: deeper nested parallel regions

    Args:
        level (pyint): Nesting level for which the ancestor thread number is requested.

    Returns:
        pyint: Thread number of ancestor.
    """
    return 0  # TODO


def omp_get_team_size(level: pyint) -> pyint:
    """Returns the number of threads in the current team at the current level.

    Returns, for a given nested level of the current thread, the size of the thread team to which the ancestor or the
    current thread belongs.

    Args:
        level (pyint): Nesting level for which the team size is requested.

    Returns:
        pyint: Team size.
    """
    return 1  # TODO


def omp_get_active_level() -> pyint:
    """Returns the number of nested active parallel regions.

    Returns the value of the `active-level-vars` ICV, which determines the number of active, nested parallel regions
    enclosing the task that contains the call.

    Returns:
        pyint: Number of active levels.
    """
    return omp_ctx().icvs.active_levels


# Initializations
omp_sched_static = 1
omp_sched_dynamic = 2
omp_sched_guided = 3
omp_sched_auto = 4
omp_sched_monotonic = 0x80000000

_omp_sched2run = new_pyint_array(5)
_omp_sched2run[omp_sched_static] = ord("s")
_omp_sched2run[omp_sched_dynamic] = ord("d")
_omp_sched2run[omp_sched_guided] = ord("g")
_omp_sched2run[omp_sched_auto] = ord("a")

_run2omp_sched_offset = ord("a")
_run2omp_sched = new_pyint_array(25)
_run2omp_sched[_omp_sched2run[omp_sched_static] - _run2omp_sched_offset] = omp_sched_static
_run2omp_sched[_omp_sched2run[omp_sched_dynamic] - _run2omp_sched_offset] = omp_sched_dynamic
_run2omp_sched[_omp_sched2run[omp_sched_guided] - _run2omp_sched_offset] = omp_sched_guided
_run2omp_sched[_omp_sched2run[omp_sched_auto] - _run2omp_sched_offset] = omp_sched_auto

if cython.compiled: # export module global variables
    globals()["omp_sched_t"] = type(omp_sched_static)
    globals()["omp_sched_static"] = omp_sched_static
    globals()["omp_sched_dynamic"] = omp_sched_dynamic
    globals()["omp_sched_guided"] = omp_sched_guided
    globals()["omp_sched_auto"] = omp_sched_auto
    globals()["omp_sched_monotonic"] = omp_sched_monotonic
