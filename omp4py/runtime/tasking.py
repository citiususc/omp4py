import typing

from omp4py.runtime.common import thread, tasks, threadshared
from omp4py.runtime.basics.casting import *


def task_submit(f: typing.Callable[[], None], c_if: bool) -> None:
    if not c_if:
        f()
        return
    task: tasks.CustomTask = tasks.CustomTask.__new__(tasks.CustomTask)
    task.f = f

    thread.current().parallel.queue.add(task)


def task_wait() -> None:
    queue: threadshared.TaskQueue = thread.current().parallel.queue

    while True:
        entry: tasks.CustomTask = cast(tasks.CustomTask, queue.take())
        if entry is None:
            return
        entry.f()
