import typing
import threading

from omp4py.runtime.common import threadshared, controlvars, thread, tasks
from omp4py.runtime.basics import array
from omp4py.runtime.basics.types import *


def omp_parallel(num: pyint, f: typing.Callable[[], None], cvars: controlvars.ControlVars, parent: tasks.Task,
                 shared: threadshared.SharedFactory) -> None:
    task: tasks.ParallelTask = tasks.ParallelTask.new(cvars.__copy__(), shared)
    task.cvars.dataenv = task.cvars.dataenv.__copy__()
    task.cvars.dataenv.thread_num = num
    thread.init(parent).set_task(task).set_parallel(task)
    # TODO barrier and explicit tasks
    f()
    task.lock_barrier.wait()


def parallel_run(f: typing.Callable[[], None], c_if: bool, c_message: str, c_nthreads: tuple[pyint, ...],
                 c_safesync: pyint, c_severity: str) -> None:
    parent: tasks.Task = thread.current().task
    cvars: controlvars.ControlVars = parent.cvars.__copy__()
    cvars.dataenv = cvars.dataenv.__copy__()

    if len(c_nthreads) > 0:
        cvars.dataenv.nthreads = array.int_from(list(c_nthreads))

    cvars.dataenv.team_size = cvars.dataenv.nthreads[0]
    if cvars.dataenv.team_size < 1:
        cvars.dataenv.team_size = 1
    if len(cvars.dataenv.nthreads) > 0:
        cvars.dataenv.nthreads = cvars.dataenv.nthreads[1:]
    cvars.dataenv.levels += 1

    shared: threadshared.SharedFactory = threadshared.SharedFactory(cvars)
    if not c_if:
        omp_parallel(0, f, cvars, parent, shared)
        return
    cvars.dataenv.active_levels += 1

    i: pyint
    for i in range(1, cvars.dataenv.team_size):
        threading.Thread(target=omp_parallel, args=(i, f, cvars, parent, shared)).start()
    omp_parallel(0, f, cvars, parent, shared)


def teams_run(*f):
    pass
