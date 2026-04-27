"""Parallel region execution for the `omp4py` runtime.

This module implements the creation and execution of parallel regions,
where a team of threads is spawned to execute a given function in
parallel.

Each parallel region creates a `ParallelTask` for each thread in the
team and manages thread execution state through the `TaskContext`. It
also handles initialization and copyin-based update of thread-private
variables (`tpvars`).
"""

from __future__ import annotations

import threading
import typing
from collections.abc import Callable

import cython

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint
from omp4py.runtime.lowlevel.threadlocal import threadlocal_set
from omp4py.runtime.tasks.barrier import barrier
from omp4py.runtime.tasks.context import TaskContext, omp_ctx
from omp4py.runtime.tasks.task import Barrier, SharedContext, Task
from omp4py.runtime.tasks.threadprivate import TPrivRef, copy_private, map_privates, update_privates

if typing.TYPE_CHECKING:
    from omp4py.runtime.icvs import Data

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["parallel"]
# END_CYTHON_IGNORE


class ParallelTask(Task):
    """Task representing a unit of work executed inside a parallel region.

    This task encapsulates the function executed by each thread in a
    parallel region, along with an optional initialization routine for
    thread-private state setup.

    Attributes:
        init (Callable[[], None] | None): Optional initialization function
            executed before the parallel function.
        f (Callable[[], None]): Parallel function executed by the thread.
    """

    init: Callable[[], None] | None
    f: Callable[[], None]

    @staticmethod
    def new(
        init: Callable[[], None] | None,
        f: Callable[[], None],
        shared: SharedContext,
        icvs: Data,
        barrier: Barrier,
    ) -> ParallelTask:
        """Create a new parallel task instance.

        Args:
            init (Callable[[], None] | None): Optional initialization routine
                for thread-private setup.
            f (Callable[[], None]): Function executed in the parallel region.
            shared (SharedContext): Shared execution context across threads.
            icvs (Data): Internal control variables for execution.
            barrier (Barrier): Synchronization barrier for the task.

        Returns:
            ParallelTask: A newly initialized parallel task.
        """
        obj: ParallelTask = ParallelTask.__new__(ParallelTask)
        obj._new_task(shared, icvs, barrier)
        obj.init = init
        obj.f = f
        return obj


def set_nthreads(ctx: TaskContext, icvs: Data, active: bool, num_threads: tuple[pyint, ...]) -> None:  # noqa: C901
    """Determine and set the number of threads for a parallel region.

    This function computes the size of the thread team based on runtime
    constraints such as nesting level, dynamic thread allocation, and
    hardware limits. It updates the internal control variables (ICVs)
    accordingly.

    Args:
        ctx (TaskContext): Current thread execution context.
        icvs (Data): Internal control variables for the new parallel region.
        active (bool): Whether the parallel region is active.
        num_threads (tuple[pyint, ...]): Requested number of threads.
    """
    num_threads_len: pyint = len(num_threads)
    if not active:  # noqa: SIM114
        icvs.team_size = 1
    elif ctx.icvs.active_levels > 0 and not ctx.icvs.nest:  # noqa: SIM114
        icvs.team_size = 1
    elif ctx.icvs.nest and ctx.icvs.active_levels == ctx.icvs.device_vars.max_active_levels:
        icvs.team_size = 1
    else:
        requested: pyint = num_threads[0] if num_threads_len > 0 and num_threads[0] > 0 else ctx.icvs.nthreads[0]

        while requested != 1:
            threads_busy = ctx.icvs.device_vars.threads_busy.get()
            available: pyint = ctx.icvs.thread_limit - threads_busy - requested + 1

            if available < 0 and icvs.dyn:
                requested = ctx.icvs.thread_limit - threads_busy + 1
            else:
                pass  # Error
            if requested == 1:
                break

            if ctx.icvs.device_vars.threads_busy.compare_exchange_weak(threads_busy, available):
                break
        icvs.team_size = requested

    if num_threads_len > 1:
        ctx.icvs.nthreads = new_pyint_array(num_threads_len - 1)
        i: pyint
        for i in range(num_threads_len - 1):
            ctx.icvs.nthreads[i] = num_threads[i + 1]
    elif len(ctx.icvs.nthreads) > 1:
        nthreads = ctx.icvs.nthreads
        ctx.icvs.nthreads = new_pyint_array(len(ctx.icvs.nthreads) - 1)
        ctx.icvs.nthreads[:] = nthreads[1:]


def _parallel_thread_init(ctx: TaskContext) -> None:
    """Initialize a new thread for a parallel region.

    This function sets the thread-local context for the new thread and
    transfers execution to the main parallel entry point.
    """
    threadlocal_set(ctx)
    _parallel_main()


def _parallel_main() -> None:
    """Execute the main logic of a parallel region.

    This function retrieves the current thread context, executes the
    assigned parallel task (including optional initialization), and
    synchronizes at the end of the region using a barrier.
    """
    ctx = omp_ctx()
    task = cython.cast(ParallelTask, ctx.task)
    if task.init is not None:
        task.init()
    task.f()
    barrier()
    ctx.pop()


def get_copyin(ctx: TaskContext, copyin: tuple[str, ...]) -> Callable[[], None]:
    """Create initialization routine for OpenMP `copyin` semantics.

    This function builds the initialization closure used during the setup
    of a parallel region when `copyin` variables are specified.

    The resulting function initializes thread-private variables (`tpvars`)
    in new threads by copying their values from the master thread before
    the parallel region begins execution. This ensures that each thread
    starts with a consistent snapshot of the master's thread-private state
    for the selected variables.

    Args:
        ctx (TaskContext): Current execution context containing the master
            thread state.
        copyin (tuple[str, ...]): Names of thread-private variables that
            must be copied from the master thread to new threads.

    Returns:
        Callable[[], None]: Initialization function executed by new
        threads before entering the parallel region.
    """
    copyids = map_privates(copyin)
    master_tpvars = ctx.tpvars

    def init() -> None:
        my_ctx = omp_ctx()
        if my_ctx.icvs.thread_num > 0:
            tpvars = my_ctx.tpvars
            update_privates(tpvars)
            i: pyint
            for i in range(len(copyids)):
                idx: pyint = copyids[i]
                if idx > -1:
                    tpvars[i].v = copy_private(master_tpvars[i].v)
        barrier()

    return init


def parallel(
    f: Callable[[], None],
    active: bool,
    num_threads: tuple[pyint, ...],
    proc_bind: pyint,  # TODO: affinity
    copyin: tuple[str, ...],
) -> None:
    """Execute a function inside a parallel region.

    This function implements the creation of a parallel region according to
    the OpenMP execution model. It establishes a team of threads that execute
    the function `f` concurrently, while configuring the runtime execution
    context for the region.

    During execution, the runtime determines the appropriate team size,
    sets up thread-local state for each new thread, and initializes thread-private
    variables (`tpvars`) when required. If `copyin` is specified, these
    variables are initialized from the master thread before the parallel
    region begins execution in the threads.

    The function also ensures proper synchronization of the thread team at
    the end of the region, so that all threads complete the parallel section
    consistently before control returns to the caller.

    Args:
        f (Callable[[], None]): Function to execute in parallel.
        active (bool): Whether the parallel region is active.
        num_threads (tuple[pyint, ...]): Requested number of threads.
        proc_bind (pyint): Processor binding policy.
        copyin (tuple[str, ...]): Names of thread-private variables to copy
            from the master thread to new threads.
    """
    ctx = omp_ctx()
    new_icvs: Data = ctx.icvs.copy()
    set_nthreads(ctx, new_icvs, active, num_threads)

    new_icvs.active_levels +=1
    all_tpvars: list[list[TPrivRef]] | None = None
    if new_icvs.active_levels == 1 and not ctx.icvs.dyn:
        if ctx.all_tpvars is None or len(ctx.all_tpvars) != new_icvs.team_size:
            ctx.all_tpvars = all_tpvars = [[] for _ in range(new_icvs.team_size)]
            all_tpvars[0] = ctx.tpvars
        else:
            all_tpvars = ctx.all_tpvars

    init: Callable[[], None] | None = get_copyin(ctx, copyin) if len(copyin) else None
    shared = SharedContext.new()
    barrier = Barrier.new(new_icvs.team_size)

    i: pyint
    threads: list[threading.Thread] = []
    for i in range(new_icvs.team_size - 1, -1, -1):  # TODO: affinity and timeout pool
        thread_ctx = TaskContext.new(ctx.task, [] if all_tpvars is None else all_tpvars[i]) if i > 0 else ctx
        thread_ctx.push(ParallelTask.new(init, f, shared.mirror(), new_icvs.copy(), barrier))
        thread_ctx.icvs.thread_num = i
        if i > 0:
            threads.append(threading.Thread(target=_parallel_thread_init, args=(thread_ctx,)))
            threads[len(threads) - 1].start()

    _parallel_main()
    for i in range(len(threads)):
        threads[i].join()
