from itertools import count

from omp4py.runtime.basics.types import *
from omp4py.runtime.basics import array, math, atomic
from omp4py.runtime.common import controlvars, thread, tasks, enums
from omp4py.runtime.basics.casting import *

_static: pyint = cast(pyint, enums.omp_sched_static)
_dynamic: pyint = cast(pyint, enums.omp_sched_dynamic)
_guided: pyint = cast(pyint, enums.omp_sched_guided)
_auto: pyint = cast(pyint, enums.omp_sched_auto)
_runtime: pyint = _auto + 1


# i, i0, [start, stop, step, mod, div, off] [start, stop, step, mod, div, off]
# 0,  1, [    2,    3,    4,   5,   6,   6] [    8,    9,   10,  11,  12,  13]

def for_bounds(elems: list[pyint]) -> array.iview:
    n: pyint = len(elems) // 3
    b: array.iview = array.new_int(2 + 6 * n)
    i: pyint
    mod: pyint = 1
    for i in range(n - 1, -1, -1):
        start: pyint = elems[3 * i]
        stop: pyint = elems[3 * i + 1]
        step: pyint = elems[3 * i + 2]

        j: pyint
        for j in range(3):
            b[2 + i * 6 + j] = elems[3 * i + j]

        b[5 + i * 6] = 0
        b[6 + i * 6] = ((stop - start + step - (1 if step > 0 else -1)) // step) * mod
        b[7 + i * 6] = mod
        mod = b[6 + i * 6]

    b[1] = mod
    return b


def for_init(bounds: array.iview, kind: pyint, chunk: pyint, monotonic: bool, ordered: pyint, order: pyint) -> None:
    cvars: controlvars.ControlVars = thread.cvars()
    team_size: pyint = cvars.dataenv.team_size

    if kind == _auto:
        kind = _static
    elif kind == _runtime:
        kind = cvars.dataenv.run_sched.kind
        chunk = cvars.dataenv.run_sched.chunk
        monotonic = cvars.dataenv.run_sched.monotonic

    task: tasks.ForTask = tasks.ForTask.new(cvars)
    task.collapse = (len(bounds) - 2) // 6
    task.first_chunk = True
    task.monotonic = monotonic
    task.iters = bounds[1]

    if kind == _guided:
        task.kind = for_guided
    elif kind == _dynamic:
        task.kind = for_dynamic
    else:
        kind = _static
        task.kind = for_static

    if chunk < 1:
        if kind == _static:
            chunk = math.ceil(task.iters / team_size)
        else:
            chunk = 1

    if task.collapse == 1 and bounds[4] < 0:
        task.chunk = -chunk
    else:
        task.chunk = chunk
    task.current_chunk = chunk
    task.step = chunk * (team_size - 1)

    thread.current().set_task(task)
    if kind == _dynamic or kind == _guided:
        start: pyint = -1 if task.collapse > 1 else bounds[2] - (chunk if kind == _dynamic else 0)
        task.shared_count = task.parallel.context.push(tasks.ForTaskID, atomic.AtomicInt.new(start))
    else:
        thread_num: pyint = task.cvars.dataenv.thread_num
        task.count = task.chunk * thread_num - task.step
        if task.collapse == 1:
            task.count += bounds[2]


def for_next(bounds: array.iview) -> bool:
    task: tasks.ForTask = cast(tasks.ForTask, thread.current().task)

    task.kind(task, bounds)
    count: pyint = task.count
    task.count += task.current_chunk

    if task.collapse > 1:
        if task.count >= task.iters:
            return False

        bounds[0] = task.current_chunk
        bounds[1] = task.current_chunk

        if task.collapse > 1:
            bounds[5] = count // bounds[7] * bounds[4]

            i: pyint
            for i in range(1, task.collapse):
                bounds[2 + i * 6 + 3] = (count % bounds[2 + i * 6 + 4] //
                                         bounds[2 + i * 6 + 5] * bounds[2 + i * 6 + 2])
        else:
            bounds[0] = bounds[3] + bounds[5]
            bounds[1] = bounds[0] + task.current_chunk
    elif task.current_chunk > 0:
        if count >= bounds[3]:
            return False
        bounds[0] = count
        if task.count > bounds[3]:
            task.count = bounds[3]
        bounds[1] = task.count
    else:
        if count <= bounds[3]:
            return False
        bounds[0] = count
        if task.count < bounds[3]:
            task.count = bounds[3]
        bounds[1] = task.count

    return True


def for_static(task: tasks.ForTask, bounds: array.iview) -> None:
    task.count += task.step


def for_dynamic(task: tasks.ForTask, bounds: array.iview) -> None:
    task.count = task.shared_count.add(task.step)


def for_guided(task: tasks.ForTask, bounds: array.iview) -> None:
    step: pyint = 1 if task.collapse > 1 else bounds[4]
    num_threads: pyint = task.cvars.dataenv.team_size
    chunk_size: pyint = task.chunk
    start: pyint = task.shared_count.get()
    while True:
        stop: pyint = task.iters if task.collapse else bounds[3]
        n: pyint = (stop - start) // step
        q: pyint = (n + num_threads - 1) // num_threads

        if q < chunk_size:
            q = chunk_size
        if q <= n:
            stop = start + q * step

        if task.shared_count.compare_exchange_weak(start, stop):
            task.count = start
            task.current_chunk = q
            break
        start = task.count


def section(*f):
    pass
