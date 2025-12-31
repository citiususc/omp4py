import typing
import itertools
# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
from omp4py.runtime.basics import array, lock, atomic
from omp4py.runtime.common import controlvars, threadshared
from omp4py.runtime.basics.types import *

# END_CYTHON_IMPORTS


__all__ = ['Task', 'ParallelTask', 'ParallelTaskID', 'TeamsTask', 'TeamsTaskID', 'ForTask', 'ForTaskID', 'SectionsTask',
           'SectionsTaskID', 'SingleTask', 'SingleTaskID', 'BarrierTask', 'BarrierTaskID', 'CustomTask', 'CustomTaskID']

_taskid = itertools.count()
ParallelTaskID: pyint = next(_taskid)
ForTaskID: pyint = next(_taskid)
TeamsTaskID: pyint = next(_taskid)
SectionsTaskID: pyint = next(_taskid)
SingleTaskID: pyint = next(_taskid)
BarrierTaskID: pyint = next(_taskid)
CustomTaskID: pyint = next(_taskid)


class Task:
    parent: typing.Optional['Task']
    parallel: typing.Optional['ParallelTask']
    teams: typing.Optional['TeamsTask']
    cvars: controlvars.ControlVars


class ParallelTask(Task):
    context: threadshared.SharedContext
    queue: threadshared.TaskQueue
    lock_mutex: lock.Mutex

    @staticmethod
    def new(cvars: controlvars.ControlVars, shared: threadshared.SharedFactory) -> 'ParallelTask':
        task: ParallelTask = ParallelTask.__new__(ParallelTask)
        task.cvars = cvars
        task.context = shared.context()
        task.queue = shared.task_queue()
        task.lock_mutex = shared.lock_mutex
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
    current_chunk: pyint
    shared_count: atomic.AtomicInt

    @staticmethod
    def new(cvars: controlvars.ControlVars) -> 'ForTask':
        task: ForTask = ForTask.__new__(ForTask)
        task.cvars = cvars
        return task


class SectionsTask(Task):
    pass


class SingleTask(Task):
    executed: atomic.AtomicFlag

    @staticmethod
    def new(cvars: controlvars.ControlVars, executed: atomic.AtomicFlag) -> 'SingleTask':
        task: SingleTask = SingleTask.__new__(SingleTask)
        task.cvars = cvars
        task.executed = executed
        return task


class BarrierTask(Task):
    parties: pyint
    count: atomic.AtomicInt
    context: threadshared.SharedContext

    @staticmethod
    def new(cvars: controlvars.ControlVars, parties: pyint, context: threadshared.SharedContext,
            count: atomic.AtomicInt) -> 'BarrierTask':
        task: BarrierTask = BarrierTask.__new__(BarrierTask)
        task.cvars = cvars
        task.parties = parties
        task.count = count
        task.context = context.__copy__()

        return task


class CustomTask(Task):
    f: typing.Callable[[], None]
    wait_event: lock.Event

    @staticmethod
    def new(cvars: controlvars.ControlVars, f: typing.Callable[[], None]) -> 'CustomTask':
        task: CustomTask = CustomTask.__new__(CustomTask)
        task.cvars = cvars
        task.f = f
        task.wait_event = lock.Event()

        return task
