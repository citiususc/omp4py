import typing

from omp4py.runtime.basics import array, lock, atomic
from omp4py.runtime.common import controlvars, threadshared
from omp4py.runtime.basics.types import *

__all__ = ['Task', 'ParallelTask', 'TeamsTask', 'ForTask', 'SectionsTask']


class Task:
    parent: typing.Optional['Task']
    parallel: typing.Optional['ParallelTask']
    teams: typing.Optional['TeamsTask']
    cvars: controlvars.ControlVars


class ParallelTask(Task):
    context: threadshared.SharedContext
    queue: threadshared.TaskQueue
    lock_mutex: lock.Mutex
    lock_barrier: lock.Barrier

    @staticmethod
    def new(cvars: controlvars.ControlVars, shared: threadshared.SharedFactory) -> 'ParallelTask':
        task: ParallelTask = ParallelTask.__new__(ParallelTask)
        task.cvars = cvars
        task.context = shared.context()
        task.queue = shared.task_queue()
        task.lock_mutex = shared.lock_mutex
        task.lock_barrier = shared.lock_barrier
        return task


class TeamsTask:
    pass


class ForTask(Task):
    kind: typing.Callable[['ForTask', array.iview], None]
    collapse: pyint
    monotonic: bool
    first_chunk: bool
    iters: pyint
    chunk: pyint
    count: pyint
    step: pyint
    shared_counter: atomic.AtomicInt

    @staticmethod
    def new(cvars: controlvars.ControlVars) -> 'ForTask':
        task: ForTask = ForTask.__new__(ForTask)
        task.cvars = cvars
        return task


class SectionsTask:
    pass
