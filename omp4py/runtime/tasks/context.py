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
from omp4py.runtime.icvs import Data, defaults
from omp4py.runtime.lowlevel import threadlocal
from omp4py.runtime.tasks.task import Barrier, SharedContext, Task

if typing.TYPE_CHECKING:
    from omp4py.runtime.tasks.threadprivate import TPrivRef

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["TaskContext", "omp_ctx"]
# END_CYTHON_IGNORE


class TaskContext:
    """Container for per-thread runtime state.

    Attributes:
        icvs (Data): Internal control variables for the runtime.
        task (Task): The currently executing task.
        tpvars (list[TPrivRef]): The `threadprivate` variables stored in the runtime state.
        all_tpvars (list[list[TPrivRef]] | None): Collection of thread-private
            variables for all threads, preserved beyond thread team lifetime.
    """
    icvs: Data
    task: Task
    tpvars: list[TPrivRef]
    all_tpvars: list[list[TPrivRef]] | None

    @staticmethod
    def new(task: Task, tpvars: list[TPrivRef]) -> TaskContext:
        """Create a new `TaskContext`.

        Args:
            task (Task): The first executing task.
            tpvars (list[TPrivRef]): The `threadprivate` variables.

        Returns:
            TaskContext: Container for per-thread runtime state.
        """
        obj: TaskContext = TaskContext.__new__(TaskContext)
        obj.task = task
        obj.icvs = task.icvs
        obj.tpvars = tpvars
        obj.all_tpvars = None
        return obj

    def push(self, task: Task) -> None:
        """Switch to a new task, saving the current one.

        The given task becomes the currently executing task. The previous
        task is stored so that execution can be resumed later via `pop()`.

        Args:
            task (Task): The task to switch to.
        """
        task._return_to = task # noqa: SLF001
        self.task = task
        self.icvs = task.icvs

    def pop(self) -> None:
        """Restore the previously executing task.

        If a previous task was saved by `push()`, it is restored as the
        currently executing task along with its associated runtime state.
        """
        if self.task._return_to is not None: # noqa: SLF001
            self.task = self.task._return_to # noqa: SLF001
            self.icvs = self.task.icvs


class ImplicitTask(Task):
    """Implicit task created for the initial thread and non-OpenMP created threads.

    This task represents the default execution context when no explicit task
    has been created by the OpenMP runtime.
    """

    @staticmethod
    def new() -> ImplicitTask:
        """Create a new implicit task instance.

        Returns:
            ImplicitTask: A newly initialized implicit task with a shared
            context and default internal control variables (ICVs).
        """
        obj: ImplicitTask = ImplicitTask.__new__(ImplicitTask)
        obj._new_task(SharedContext.new(), defaults.copy(), Barrier.new(1))
        return obj


def context_init() -> None:
    """Initialize the thread-local task context.

    This function is used as the initialization hook for the underlying
    thread-local storage. It creates a new `TaskContext` instance,
    initializes it with default internal control variables, and stores it
    in thread-local storage.
    """
    ctx = TaskContext.new(ImplicitTask.new(), [])
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
