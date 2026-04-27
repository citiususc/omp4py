"""Runtime implementations of OpenMP worksharing constructs.

This module provides the runtime logic used by code generated from OpenMP
worksharing directives such as `for`, `sections`, `single`, and `ordered`.

It is responsible for distributing loop iterations between threads,
coordinating execution order, synchronizing implicit barriers, and sharing
state between worker threads during a parallel region.

The implementation supports the main OpenMP scheduling policies, including
static, dynamic, guided, and runtime scheduling. It also provides the
internal data structures required to manage collapsed loops, ordered loop
regions, section dispatching, and `copyprivate` data exchange.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable

import cython

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.atomic import AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.mutex import Event
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from omp4py.runtime.tasks.barrier import barrier
from omp4py.runtime.tasks.context import TaskContext, omp_ctx
from omp4py.runtime.tasks.task import Task, instanceof

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = [
    "ForBounds",
    "SingleCopyPrivate",
    "for_bounds",
    "for_end",
    "for_init",
    "for_next_dynamic",
    "for_next_guided",
    "for_next_runtime",
    "for_next_static",
    "ordered_next",
    "ordered_start",
    "sections_end",
    "sections_init",
    "sections_next",
    "single_copy_get",
    "single_copy_notify",
    "single_copy_wait",
    "single_end",
    "single_init",
]
# END_CYTHON_IGNORE

_static = ord("s")
_dynamic = ord("d")
_guided = ord("g")
_runtime = ord("r")

# bounds rs fields
_r_start = 0
_r_stop = 1
_r_step = 2
_r_mod = 3
_r_div = 4
_r_off = 5
_rs_len = 6


class ForShared:
    """Shared state for `for` execution.

    Attributes:
        count (AtomicInt): Atomic counter used to assign dynamic schedules loop chunks.
    """
    count: AtomicInt

    @staticmethod
    def new(count: pyint) -> ForShared:
        """Create a shared for object.

        Args:
            count (pyint): Initial iteration value.

        Returns:
            ForShared: Initialized shared state.
        """
        obj: ForShared = ForShared.__new__(ForShared)
        obj.count = AtomicInt.new(count)
        return obj


class OrderedShared:
    """Shared state for ordered loop execution.

    Attributes:
        count (pyint): Last completed ordered iteration.
        it (pyint): Remaining iterations in the current ordered chunk.
        own (pyint): Thread that currently owns the ordered region.
        event (AtomicObject[Event]): Event used to wake waiting threads.
    """
    count: pyint
    it: pyint
    own: pyint
    event: AtomicObject[Event]

    @staticmethod
    def new(count: pyint) -> OrderedShared:
        """Create ordered synchronization state.

        Args:
            count (pyint): Initial ordered counter.

        Returns:
            OrderedShared: Initialized ordered state.
        """
        obj: OrderedShared = OrderedShared.__new__(OrderedShared)
        obj.count = count
        obj.own = -1
        obj.it = -1
        obj.chunk = -1
        obj.event = AtomicObject.new()
        obj.event.set(Event.new())
        return obj


class ForBounds:
    """Loop iteration bounds and scheduling state.

    Attributes:
        init (pyint): First iteration of the current chunk.
        end (pyint): End limit of the current chunk.
        it (pyint): Remaining iterations for collapsed ordered loops.
        step (pyint): Distance between assigned chunks.
        count (pyint): Internal scheduler counter.
        chunk (pyint): Chunk size for each assignment.
        its (pyint): Total flattened iteration count.
        collapse (pyint): Number of collapsed loops.
        rs (pyint_array): Internal range metadata buffer.
    """
    init: pyint
    end: pyint
    it: pyint
    step: pyint
    count: pyint
    chunk: pyint
    its: pyint
    collapse: pyint
    rs: pyint_array


class ForTask(Task):
    """Task context used by a `for` worksharing region.

    Attributes:
        bounds (ForBounds): Loop bounds assigned to the task.
        kind (Callable[[ForBounds], bool]): Scheduling function in use.
        shared_count (AtomicInt | None): Shared counter for dynamic schedules.
        ordered (OrderedShared | None): Ordered execution state.
    """
    bounds: ForBounds
    kind: Callable[[ForBounds], bool]
    shared_count: AtomicInt | None
    ordered: OrderedShared | None

    @staticmethod
    def new(ctx: TaskContext, bounds: ForBounds) -> ForTask:
        """Create a loop task context.

        Args:
            ctx (TaskContext): Current task context.
            bounds (ForBounds): Loop bounds data.

        Returns:
            ForTask: Initialized task.
        """
        obj: ForTask = ForTask.__new__(ForTask)
        obj._new_task(ctx.task.shared, ctx.icvs, ctx.task.barrier)
        obj.bounds = bounds
        obj.kind = for_next_static
        obj.ordered = None
        obj.shared_count = None
        return obj


def for_bounds(collapse: pyint) -> ForBounds:
    """Create a loop bounds container.

    Allocates and initializes the internal structure used to store loop
    iteration data, including support for collapsed nested loops.

    Args:
        collapse (pyint): Number of collapsed loops.

    Returns:
        ForBounds: Initialized bounds object.
    """
    bounds: ForBounds = ForBounds.__new__(ForBounds)
    bounds.collapse = collapse
    bounds.rs = new_pyint_array(collapse * _rs_len)
    return bounds


def set_bounds(bounds: ForBounds) -> None:
    """Compute internal loop iteration metadata.

    Prepares the flattened iteration space used by the runtime scheduler,
    including offsets, divisors, and total iteration count.

    Args:
        bounds (ForBounds): Loop bounds structure to update.
    """
    mod: pyint = 1
    i: pyint
    for i in range(bounds.collapse - 1, -1, -1):
        start: pyint = bounds.rs[_rs_len * i + _r_start]
        stop: pyint = bounds.rs[_rs_len * i + _r_stop]
        step: pyint = bounds.rs[_rs_len * i + _r_step]

        bounds.rs[_rs_len * i + _r_div] = ((stop - start + step - (1 if step > 0 else -1)) // step) * mod
        bounds.rs[_rs_len * i + _r_off] = mod
        mod = bounds.rs[_rs_len * i + _r_div]

    bounds.its = mod


def for_init(bounds: ForBounds, modifier: pyint, schedule: pyint, chunk: pyint, ordered: pyint) -> None:
    """Initialize a `for` worksharing region.

    Creates the task state for the loop, resolves scheduling policy,
    computes chunk sizes, and registers shared state required by the team.

    Args:
        bounds (ForBounds): Loop bounds information.
        modifier (pyint): Scheduling modifier.
        schedule (pyint): Scheduling kind.
        chunk (pyint): Requested chunk size.
        ordered (pyint): Whether ordered execution is enabled.
    """
    ctx = omp_ctx()
    task = ForTask.new(ctx, bounds)
    set_bounds(bounds)
    bounds.count = 0

    if schedule == _runtime:
        run_sched = ctx.icvs.run_sched
        chunk = run_sched.chunksize
        schedule = run_sched.type
        if schedule == _dynamic:
            task.kind = for_next_dynamic
        elif schedule == _guided:
            task.kind = for_next_guided

    if chunk < 0:
        chunk = cython.cast(cython.longlong, math.ceil(bounds.its / ctx.icvs.team_size)) if schedule == _static else 1
    if bounds.collapse == 1:
        bounds.count = bounds.rs[_r_start]
        if bounds.rs[_r_step] < 0:  # negative step
            chunk = -chunk

    bounds.step = chunk
    bounds.chunk = chunk

    if schedule == _static:
        bounds.step *= ctx.icvs.team_size - 1
        bounds.count += bounds.chunk * ctx.icvs.thread_num - bounds.step
    else:
        task.shared_count = cython.cast(
            ForShared,
            ctx.task.shared.get(ForShared) or ctx.task.shared.sync(ForShared.new(bounds.count)),
        ).count

    if ordered:
        ordered_count: pyint = bounds.rs[_r_start] if bounds.collapse == 1 else 0
        task.ordered = cython.cast(
            OrderedShared,
            ctx.task.shared.get(OrderedShared) or ctx.task.shared.sync(OrderedShared.new(ordered_count)),
        )

    ctx.push(task)


def for_next_runtime(bounds: ForBounds) -> bool:
    """Get the next loop chunk using runtime scheduling.

    Args:
        bounds (ForBounds): Loop bounds structure.

    Returns:
        bool: True if more iterations are available.
    """
    return cython.cast(ForTask, omp_ctx().task).kind(bounds)


def for_next_static(bounds: ForBounds) -> bool:
    """Get the next loop chunk using static scheduling.

    Args:
        bounds (ForBounds): Loop bounds structure.

    Returns:
        bool: True if more iterations are available.
    """
    bounds.count += bounds.step
    return fix_bounds(bounds)


def for_next_dynamic(bounds: ForBounds) -> bool:
    """Get the next loop chunk using dynamic scheduling.

    Args:
        bounds (ForBounds): Loop bounds structure.

    Returns:
        bool: True if more iterations are available.
    """
    task = cython.cast(ForTask, omp_ctx().task)
    shared_counter = cython.cast(AtomicInt, task.shared_count)
    bounds.count = shared_counter.fetch_add(bounds.step)
    return fix_bounds(bounds)


def for_next_guided(bounds: ForBounds) -> bool:
    """Get the next loop chunk using guided scheduling.

    Chunk sizes decrease over time while preserving a minimum size.

    Args:
        bounds (ForBounds): Loop bounds structure.

    Returns:
        bool: True if more iterations are available.
    """
    task = cython.cast(ForTask, omp_ctx().task)
    shared_count = cython.cast(AtomicInt, task.shared_count)
    step: pyint = 1 if bounds.collapse > 1 else bounds.rs[_r_step]
    stop: pyint = bounds.its if bounds.collapse > 1 else bounds.rs[_r_stop]
    num_threads: pyint = task.icvs.team_size
    chunk_size: pyint = bounds.chunk

    while True:
        start: pyint = shared_count.get()
        n: pyint = (stop - start) // step
        q: pyint = (n + num_threads - 1) // num_threads

        if q < chunk_size:  # noqa: PLR1730
            q = chunk_size
        if q <= n:
            stop = start + q * step

        if shared_count.compare_exchange_weak(start, stop):
            bounds.count = start
            bounds.chunk = q if bounds.chunk > 0 else -q
            break

    return fix_bounds(bounds)


def for_end(nowait: bool) -> None:
    """Finish a `for` worksharing region.

    Applies the implicit barrier unless `nowait` is enabled and restores
    the previous task context.

    Args:
        nowait (bool): If True, skip the barrier.
    """
    if not nowait:
        barrier()
    omp_ctx().pop()


def fix_bounds(bounds: ForBounds) -> bool:
    """Update visible loop bounds for the current chunk.

    Converts the internal counter into the public iteration range used by
    generated loop code.

    Args:
        bounds (ForBounds): Loop bounds structure.

    Returns:
        bool: True if more iterations are available.
    """
    stop: pyint = bounds.rs[_r_stop]
    count: pyint = bounds.count
    bounds.count += bounds.chunk

    if bounds.collapse > 1:
        if count >= bounds.its:
            return False

        bounds.it = bounds.chunk
        bounds.init = bounds.chunk
        bounds.rs[_r_mod] = count // bounds.rs[_r_off] * bounds.rs[_r_step]

        i: pyint = _rs_len
        for _ in range(1, bounds.collapse):
            bounds.rs[i + _r_mod] = count % bounds.rs[i + _r_div] // bounds.rs[i + _r_off] * bounds.rs[i + _r_step]
            i += _rs_len

    elif bounds.chunk > 0:
        if count >= stop:
            return False
        bounds.init = count
        if bounds.count > stop:  # noqa: PLR1730
            bounds.end = stop
        else:
            bounds.end = bounds.count
    else:
        if count <= stop:
            return False
        bounds.init = count
        if bounds.count < stop:  # noqa: PLR1730
            bounds.end = stop
        else:
            bounds.end = bounds.count

    return True


if cython.compiled:
    # Export the for_next function manually because cpdef prevents using a function pointer, which is required to
    # implement runtime scheduling.
    globals()["for_next_static"] = lambda bounds: for_next_static(bounds)  # noqa: PLW0108
    globals()["for_next_dynamic"] = lambda bounds: for_next_dynamic(bounds)  # noqa: PLW0108
    globals()["for_next_guided"] = lambda bounds: for_next_guided(bounds)  # noqa: PLW0108


def ordered_start() -> None:
    """Enter an ordered loop region.

    Blocks the current thread until it is allowed to execute the next
    ordered section of the loop.
    """
    ctx = omp_ctx()
    if not instanceof(ctx.task, ForTask):
        print(  # noqa: T201
            "error: 'ordered' directive used outside of a loop construct\n"
            "The 'ordered' directive must only appear inside a for-loop region.\n"
            "Aborting program.",
            file=sys.stderr,
        )
        os._exit(1)
    task = cython.cast(ForTask, ctx.task)
    ordered = task.ordered
    if ordered is None:
        print(  # noqa: T201
            "error: 'ordered' directive used without matching 'ordered' clause\n"
            "The 'ordered' directive requires the loop to be declared with an 'ordered' clause.\n"
            "Aborting program.",
            file=sys.stderr,
        )
        os._exit(1)

    if ordered.own == ctx.icvs.thread_num:
        return

    expected: pyint = task.bounds.count - task.bounds.chunk
    while ordered.own != -1 or ordered.count != expected:
        cython.cast(Event, ordered.event.get()).wait()

    ordered.it = task.bounds.chunk if task.bounds.chunk > 0 else -task.bounds.chunk
    ordered.own = ctx.icvs.thread_num


def ordered_next() -> None:
    """Advance ordered loop execution.

    Releases ownership of the ordered region when the current ordered
    chunk has finished.
    """
    task = cython.cast(ForTask, omp_ctx().task)
    ordered = cython.cast(OrderedShared, task.ordered)

    if ordered.it > 0:
        ordered.it -= 1

    if ordered.it == 0:
        ordered.count = task.bounds.count
        ordered.own = -1
        cython.cast(Event, ordered.event.exchange(Event.new())).notify()


class SectionsShared:
    """Shared state for `sections` execution.

    Attributes:
        count (AtomicInt): Atomic counter used to assign sections.
    """
    count: AtomicInt

    @staticmethod
    def new(n: pyint) -> SectionsShared:
        """Create shared sections state.

        Args:
            n (pyint): Number of sections.

        Returns:
            SectionsShared: Initialized shared state.
        """
        obj: SectionsShared = SectionsShared.__new__(SectionsShared)
        obj.count = AtomicInt.new(n)
        return obj


class SectionsTask(Task):
    """Task context used by a `sections` region.

    Attributes:
        shared (SectionsShared): Shared section scheduler state.
    """
    shared: SectionsShared

    @staticmethod
    def new(ctx: TaskContext, shared: SectionsShared) -> SectionsTask:
        """Create a sections task context.

        Args:
            ctx (TaskContext): Current task context.
            shared (SectionsShared): Shared sections state.

        Returns:
            SectionsTask: Initialized task.
        """
        obj: SectionsTask = SectionsTask.__new__(SectionsTask)
        obj._new_task(ctx.task.shared, ctx.icvs, ctx.task.barrier)
        obj.shared = shared
        return obj


def sections_init(sections: pyint) -> None:
    """Initialize a `sections` worksharing region.

    Args:
        sections (pyint): Number of sections available.
    """
    ctx = omp_ctx()
    task = SectionsTask.new(
        ctx,
        cython.cast(
            SectionsShared,
            ctx.task.shared.get(SectionsShared) or ctx.task.shared.sync(SectionsShared.new(sections)),
        ),
    )

    ctx.push(task)


def sections_next() -> pyint:
    """Get the next available section index.

    Returns:
        pyint: Section identifier, or `0` when no sections remain.
    """
    n: pyint = cython.cast(SectionsTask, omp_ctx().task).shared.count.fetch_sub(1)
    if n < 0:
        return 0
    return n


def sections_end(nowait: bool) -> None:
    """Finish a `sections` region.

    Applies the implicit barrier unless `nowait` is enabled.

    Args:
        nowait (bool): If True, skip the barrier.
    """
    if not nowait:
        barrier()
    omp_ctx().pop()


class SingleCopyPrivate:
    """Linked storage used for `copyprivate` variables.

    Attributes:
        v0 (object): Stored value slot 0.
        v1 (object): Stored value slot 1.
        v2 (object): Stored value slot 2.
        v3 (object): Stored value slot 3.
        v4 (object): Stored value slot 4.
        v5 (object): Stored value slot 5.
        v6 (object): Stored value slot 6.
        v7 (object): Stored value slot 7.
        next (SingleCopyPrivate | None): Next storage block.
    """
    v0: object
    v1: object
    v2: object
    v3: object
    v4: object
    v5: object
    v6: object
    v7: object
    next: SingleCopyPrivate | None

    @staticmethod
    def new(nvars: pyint) -> SingleCopyPrivate:
        """Create storage for `copyprivate` variables.

        Args:
            nvars (pyint): Number of variables to store.

        Returns:
            SingleCopyPrivate: Allocated storage chain.
        """
        obj: SingleCopyPrivate = SingleCopyPrivate.__new__(SingleCopyPrivate)
        if (nvars - 8) > 0:
            obj.next = SingleCopyPrivate.new(nvars - 8)
        return obj


class SingleShared:
    """Shared state for a `single` region.

    Attributes:
        count (AtomicInt): Selects the first thread entering the region.
        event (Event | None): Notification event for `copyprivate` data.
        copyprivate (SingleCopyPrivate | None): Shared copied values.
    """
    count: AtomicInt
    event: Event | None
    copyprivate: SingleCopyPrivate | None

    @staticmethod
    def new(copyprivate: bool) -> SingleShared:
        """Create shared `single` state.

        Args:
            copyprivate (bool): Whether copyprivate support is needed.

        Returns:
            SingleShared: Initialized shared state.
        """
        obj: SingleShared = SingleShared.__new__(SingleShared)
        obj.count = AtomicInt.new(1)
        obj.event = Event.new() if copyprivate else None
        obj.copyprivate = None
        return obj


class SingleTask(Task):
    """Task context used by a `single` region.

    Attributes:
        shared (SingleShared): Shared single-region state.
    """
    shared: SingleShared

    @staticmethod
    def new(ctx: TaskContext, shared: SingleShared) -> SingleTask:
        """Create a single task context.

        Args:
            ctx (TaskContext): Current task context.
            shared (SingleShared): Shared single state.

        Returns:
            SingleTask: Initialized task.
        """
        obj: SingleTask = SingleTask.__new__(SingleTask)
        obj._new_task(ctx.task.shared, ctx.icvs, ctx.task.barrier)
        obj.shared = shared
        return obj


def single_init(copyprivate: bool) -> bool:
    """Initialize a `single` region.

    Selects one thread to execute the single block and optionally prepares
    storage for `copyprivate` values.

    Args:
        copyprivate (bool): Whether copyprivate support is required.

    Returns:
        bool: True for the selected thread, False for others.
    """
    ctx = omp_ctx()
    task = SingleTask.new(
        ctx,
        cython.cast(
            SingleShared,
            ctx.task.shared.get(SingleShared) or ctx.task.shared.sync(SingleShared.new(copyprivate)),
        ),
    )

    ctx.push(task)
    return task.shared.count.fetch_sub(1) > 0


def single_end(nowait: bool) -> None:
    """Finish a `single` region.

    Applies the implicit barrier unless `nowait` is enabled.

    Args:
        nowait (bool): If True, skip the barrier.
    """
    if not nowait:
        barrier()
    omp_ctx().pop()


def single_copy_get(nvars: pyint) -> SingleCopyPrivate:
    """Create storage for `copyprivate` variables.

    Args:
        nvars (pyint): Number of variables to store.

    Returns:
        SingleCopyPrivate: Storage container.
    """
    shared = cython.cast(SingleTask, omp_ctx().task).shared
    shared.copyprivate = SingleCopyPrivate.new(nvars)
    return shared.copyprivate


def single_copy_notify() -> None:
    """Notify waiting threads that `copyprivate` data is ready."""
    shared = cython.cast(SingleTask, omp_ctx().task).shared
    cython.cast(Event, shared.event).notify()


def single_copy_wait() -> SingleCopyPrivate:
    """Wait for `copyprivate` data and return it.

    Returns:
        SingleCopyPrivate: Shared copied values.
    """
    shared = cython.cast(SingleTask, omp_ctx().task).shared
    cython.cast(Event, shared.event).wait()
    return cython.cast(SingleCopyPrivate, shared.copyprivate)
