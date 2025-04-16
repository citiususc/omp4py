from omp4py.runtime.basics import lock, atomic
from omp4py.runtime.common import tasks, threadshared, thread
from omp4py.runtime.basics.types import *
from omp4py.runtime.basics.casting import *


class BarrierContext:
    ctx: threadshared.SharedContext
    count: atomic.AtomicInt

    @staticmethod
    def new() -> 'BarrierContext':
        bc:BarrierContext = BarrierContext.__new__(BarrierContext)
        bc.ctx = threadshared.SharedContext.new()
        bc.count = atomic.AtomicInt.new(0)

        return bc


class BarrierShared:
    notify_event: bool
    event: lock.Event

    @staticmethod
    def new() -> 'BarrierShared':
        bs:BarrierShared = BarrierShared.__new__(BarrierShared)
        bs.notify_event = False
        bs.event = lock.Event.new()

        return bs


def task_barrier(task: tasks.Task) -> None:
    parties: pyint = task.cvars.dataenv.team_size
    current: thread.OmpThread = thread.current()
    bc: BarrierContext = task.parallel.context.push(tasks.BarrierTaskID, BarrierContext.new())
    barrier: tasks.BarrierTask = tasks.BarrierTask.new(thread.cvars(), parties, bc.ctx, bc.count)
    queue: threadshared.TaskQueue = task.parallel.queue

    current.set_task(barrier)
    while True:
        while True:
            entry: tasks.CustomTask = cast(tasks.CustomTask, queue.take())
            if entry is None:
                break
            current.set_task(entry)
            entry.f()
            current.pop_task()
            entry.wait_event.notify()

        shared: BarrierShared = barrier.context.push(tasks.BarrierTaskID, BarrierShared.new())

        if barrier.count.add(1) == parties:
            barrier.context.move_last()
            shared = barrier.context.get(tasks.BarrierTaskID)
            shared.notify_event = False
            shared.event.notify()
        else:
            shared.event.wait()
            barrier.count.sub(1)

        if not shared.notify_event:
            break

    current.pop_task()


def task_notify(task: tasks.Task) -> None:
    bc: BarrierContext | None = cast(BarrierContext, task.parallel.context.get(tasks.BarrierTaskID))
    if bc is None:
        return

    bc.ctx.move_last()
    shared: BarrierShared = bc.ctx.get(tasks.BarrierTaskID)
    if shared is not None:
        shared.notify_event = True
        shared.event.notify()
