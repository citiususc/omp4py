import typing
# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
from omp4py.runtime.common import thread, tasks, barrier, threadshared
# END_CYTHON_IMPORTS


from cython import cast


def task_submit(f: typing.Callable[[], None]) -> None:
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
