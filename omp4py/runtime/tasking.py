import typing

from omp4py.runtime.common import thread, tasks, barrier, threadshared
from omp4py.runtime.basics.casting import *


def task_submit(f: typing.Callable[[], None], c_if: bool) -> None:
    if not c_if:
        f()
        return
    task: tasks.CustomTask = tasks.CustomTask.new(thread.cvars(), f)

    thread.current().parallel.queue.add(task)
    barrier.task_notify(thread.current().task)


def task_wait() -> None:
    queue: threadshared.TaskQueue = thread.current().task.parallel.queue
    queue.set_history()
    current: thread.OmpThread = thread.current()
    while True:
        entry: tasks.CustomTask = cast(tasks.CustomTask, queue.take())
        if entry is None:
            break
        current.set_task(entry)
        entry.f()
        current.pop_task()
        entry.wait_event.notify()

    while True:
        entry: tasks.CustomTask = cast(tasks.CustomTask, queue.history_take())
        if entry is None:
            break
        entry.wait_event.wait()
