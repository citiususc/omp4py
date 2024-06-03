import copy
import math
import threading
import queue
from typing import List, Tuple, Callable, TypeVar, Generator
import omp4py.api as api
from omp4py.context import OpenMpLevel, OrderedContext
from omp4py.core import _omp_context
from omp4py.context import AtomicInt
from omp4py.error import OmpError

var = threading.local()


# function to execute parallel directive
def parallel_run(f: Callable, if_=True, num_threads=None):
    active = bool(if_) and api.omp_get_active_level() < api.omp_get_max_active_levels()

    levels = _omp_context.levels()
    current_level = _omp_context.current_level()
    # by default the new level is the same as previous
    new_level = OpenMpLevel(
        level=current_level.level + 1,
        active_level=current_level.active_level,
        num_threads=current_level.num_threads,
        thread_num=current_level.thread_num,
        barrier=current_level.barrier,
        lock=current_level.lock
    )

    # if is not active, the function is executed only by the call thread
    if not active:
        _omp_context.levels().append(new_level)
        return f()

    if num_threads is None:
        num_threads = _omp_context.max_num_threads

    if num_threads < 1:
        raise OmpError("num_threads must be a positive integer")

    # update with the new active level info
    new_level.active_level += 1
    new_level.num_threads = num_threads
    new_level.thread_num = 0  # current threads will be master in new block
    new_level.barrier = threading.Barrier(num_threads)
    # copy previous levels to new threads to use in api functions
    thread_levels = [levels + [copy.copy(new_level)] for _ in range(num_threads - 1)]

    levels.append(new_level)
    # launch new threads
    for i in range(num_threads - 1):
        threading.Thread(target=omp_parallel, args=(f, i + 1, thread_levels[i])).start()
    # master thread
    omp_parallel(f, 0, None)


# new threads start function
def omp_parallel(f: Callable, thread_num: int, levels: List[OpenMpLevel] | None):
    try:
        if thread_num > 0:  # update only in new threads
            _omp_context.set_levels(levels)
            _omp_context.current_level().thread_num = thread_num
        f()
        taskwait()  # consume any pending task
        _omp_context.current_level().barrier.wait()  # implicit synchronization at the end of parallel block
    except:
        _omp_context.current_level().barrier.abort()
        raise


# for directive function to create parallel loops
def omp_range(*args: int | Tuple[int, ...], id: str = "", ordered: bool = False, nowait: bool = False,
              schedule: str = "", chunk_size: int = -1, lastprivate=False):
    if schedule == "static":
        f = omp_static_range
    elif schedule == "dynamic":
        f = omp_range_dynamic
    elif schedule == "guided":
        f = omp_range_guided
    elif schedule == "runtime":
        return omp_range(*args, id=id, ordered=ordered, nowait=nowait, schedule=_omp_context.schedule,
                         chunk_size=chunk_size if chunk_size != -1 else _omp_context.chunk_size)
    else:
        f = omp_static_range

    if not isinstance(chunk_size, int):
        raise OmpError("chunk_size must be an integer")

    if len(args) == 0:
        raise TypeError("range expected at least 1 argument, got 0")

    # if collate > 1, start, stop and step are tuples of collated ranges
    if isinstance(args[0], int):  # without collate or with collate == 1
        args = list(args)
        if len(args) < 3:
            if len(args) < 2:
                args.insert(0, 0)
        args.append(1)
        if not omp_range_check(*args):
            return []
        gen = f(id, chunk_size, nowait, args[0], args[1], args[2])
        if lastprivate:
            _omp_context.current_level().last_private = next(reversed(range(*args)))
    else:  # with collate > 1
        for arg in args:
            if None in arg:  # decorator insert None if found a range() without arguments in any collated range
                raise TypeError("range expected at least 1 argument, got 0")
        if not omp_range_check(*args):
            return []
        gen = omp_range_flattener(f, id, chunk_size, nowait, args[0], args[1], args[2])

        if lastprivate:
            _omp_context.current_level().last_private = tuple(
                next(collected_range_reversed(0, args[0], args[1], args[2])))

    if ordered:
        gen = check_ordered_loop(id, gen, args[0], args[1], args[2])

    return gen


# check if range is correct and has any iterations
def omp_range_check(start: int | Tuple[int, ...], stop: int | Tuple[int, ...], step: int | Tuple[int, ...]):
    if isinstance(start, int):
        if step == 0:
            raise OmpError("range() arg 3 must not be zero")
        if (stop > start and step > 0) or (stop < start and step < 0):
            return True
        return False
    else:
        return all([omp_range_check(start[i], stop[i], step[i]) for i in range(len(start))])


# static scheduler
def omp_static_range(id: str, chunk_size: int, nowait: bool, start: int, stop: int, step: int):
    thread_num = _omp_context.current_level().thread_num
    num_threads = _omp_context.current_level().num_threads
    if chunk_size < 1:  # create num_threads chunks of equal size
        its = math.floor(math.fabs(stop - start) / step)
        chunk_size = math.ceil(its / num_threads)

    local_step = step * chunk_size
    local_start = start + local_step * thread_num

    for i in range(local_start, stop, local_step * num_threads):
        if local_step > 0:
            for j in range(0, local_step, step):
                if i + j < stop:
                    yield i + j
        else:
            for j in range(0, -local_step, -step):
                if i - j > stop:
                    yield i - j

    if not nowait:  # synchronization at the end of for block
        _omp_context.current_level().barrier.wait()


# dynamic scheduler
def omp_range_dynamic(id: str, chunk_size: int, nowait: bool, start: int, stop: int, step: int):
    if chunk_size < 1:
        chunk_size = 1

    counter = with_shared_obj(id, lambda: AtomicInt(start))

    local_step = step * chunk_size
    loop = True
    while loop:
        i = counter.get_and_inc(local_step)
        if local_step > 0:
            for j in range(0, local_step, step):
                if i + j < stop:
                    yield i + j
                else:
                    loop = False
                    break
        else:
            for j in range(0, -local_step, -step):
                if i - j > stop:
                    yield i - j
                else:
                    loop = False
                    break
    if not nowait:
        _omp_context.current_level().barrier.wait()  # synchronization at the end of for block


# guided scheduler
def omp_range_guided(id: str, chunk_size: int, nowait: bool, start: int, stop: int, step: int):
    num_threads = _omp_context.current_level().num_threads
    if chunk_size < 1:
        chunk_size = 1

    counter = with_shared_obj(id, lambda: AtomicInt(start))
    start = counter.get()

    while True:
        if start == stop:
            break

        n = (stop - start) // step
        q = (n + num_threads - 1) // num_threads

        if q < chunk_size:
            q = chunk_size
        if q <= n:
            nend = start + q * step
        else:
            nend = stop

        tmp = counter.compare_and_swap(start, nend)
        if tmp == start:
            local_step = step * chunk_size
            i = start
            if local_step > 0:
                for j in range(0, local_step, step):
                    if i + j < stop:
                        yield i + j
                    else:
                        break
            else:
                for j in range(0, -local_step, -step):
                    if i - j > stop:
                        yield i - j
                    else:
                        break
            tmp = nend
        start = tmp

    if not nowait:
        _omp_context.current_level().barrier.wait()  # synchronization at the end of for block


S = TypeVar("S")


# share an object between all threads in the level
def with_shared_obj(id: str, new: Callable[[], S]) -> S:
    thread_num = _omp_context.current_level().thread_num
    num_threads = _omp_context.current_level().num_threads
    # We need a lock to modify the common object context
    with _omp_context.current_level().lock:
        # if entry id not exits, we create a new entry for each thread
        if id not in _omp_context.current_level().shared:
            obj = new()
            _omp_context.current_level().shared[id] = [[obj] for _ in range(num_threads)]
        else:
            entry = _omp_context.current_level().shared[id]
            # if entry id exits but my array is empty, we create a new entry for each thread
            # with nowait a thread can start again before other thead end the previous execution
            if len(entry[thread_num]) == 0:
                obj = new()
                for i in range(num_threads):
                    entry[i].append(obj)
        # retrieve and delete the shared object
        obj = _omp_context.current_level().shared[id][thread_num].pop()
        # remove the entry if all threads have consumed the shared object
        if not any(_omp_context.current_level().shared[id]):
            del _omp_context.current_level().shared[id]
    return obj


# transform multiple loops in a single loop with n*m*... iterations
def omp_range_flattener(f: Callable, id: str, chunk_size: int, nowait: bool, start: Tuple[int, ...],
                        stop: Tuple[int, ...], step: Tuple[int, ...]):
    iter_list = list()
    iter_total = 1
    n = len(start)
    # calculate the number of iterations per loop
    for i in range(n):
        iter = math.floor(math.fabs(stop[i] - start[i]) / step[i])
        iter_total *= iter
        iter_list.append(iter)

    iter_mults = [iter_total / iter_list[0]]
    # calculate the module of each inner loop
    for i in range(1, n):
        iter_mults.append(iter_list[-1] / iter_list[i])

    # transform the target of the flatted loop in multiple targets
    for i in f(id, chunk_size, nowait, 0, iter_total, 1):
        yield [int(i // iter_mults[ni]) % iter_list[ni] * step[ni] + start[ni] for ni in range(n)]


def check_ordered_loop(id: str, gen: Generator, start: int | Tuple[int, ...], stop: int | Tuple[int, ...],
                       step: int | Tuple[int, ...]) -> Generator:
    if isinstance(start, int):
        l = lambda: OrderedContext(collected_range(0, (start,), (stop,), (step,)))
    else:
        l = lambda: OrderedContext(collected_range(0, start, stop, step))

    ordered_gen = with_shared_obj(id + "-ordered", l)
    level = _omp_context.current_level()
    level.iter_order = ordered_gen

    if isinstance(start, int):
        for elem in gen:
            level.iter_elem = [elem]
            yield elem
    else:
        for elem in gen:
            level.iter_elem = elem
            yield elem

    level.iter_order = None


# generate the result of multiple collected range
def collected_range(i: int, *args: Tuple[int, ...]):
    for x in range(args[0][i], args[1][i], args[2][i]):
        if i + 1 < len(args[i]):
            for y in collected_range(i + 1, *args):
                yield [x] + y
        else:
            yield [x]


# generate the result of multiple collected range in reverse order
def collected_range_reversed(i: int, *args: Tuple[int, ...]):
    for x in reversed(range(args[0][i], args[1][i], args[2][i])):
        if i + 1 < len(args[i]):
            for y in collected_range_reversed(i + 1, *args):
                yield [x] + y
        else:
            yield [x]


# Check if this thread has executed the last iteration or last section
def lastprivate(args: int | Tuple[int, ...]):
    return _omp_context.current_level().last_private == args


# Open an ordered context
def ordered(id: str):
    level = _omp_context.current_level()

    if level.iter_order.id == id:
        pass
    elif level.iter_order.id is None:
        level.iter_order.id = id
    elif level.iter_order.id != id:
        level.iter_order.error = OmpError("only a single ordered clause can appear on a loop directive")
        if level.iter_order.condition._lock.locked():
            level.iter_order.condition.notify_all()
        else:
            with level.iter_order.condition:
                level.iter_order.condition.notify_all()
        raise level.iter_order.error

    # Blocks until current elem is the next elem in the order
    class Order:

        def __enter__(self):
            level.iter_order.condition.acquire()
            while not level.iter_order.check(level.iter_elem):
                if level.iter_order.error:
                    raise level.iter_order.error
                level.iter_order.condition.wait()

        def __exit__(self, exc_type, exc_value, traceback):
            level.iter_order.condition.notify_all()
            level.iter_order.condition.release()

    return Order()


# Open a sections context
def sections(id: str, nowait=False):
    execs = with_shared_obj(id, lambda: set())
    lock = threading.RLock()

    class Sections:

        def __enter__(self):
            return self

        # only for synchronization if not nowait
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not nowait:
                _omp_context.current_level().barrier.wait()

        # return True if section not executed
        def __call__(self, i: int, n: int):
            with lock:
                _omp_context.current_level().last_private = n - 1
                if i in execs:
                    return False
                execs.add(i)
                return True

    return Sections()


# copy a variable
def var_copy(value):
    return copy.copy(value)


# get the level lock
def level_lock():
    return _omp_context.current_level().lock


# perform a thread synchronization
def barrier():
    level = _omp_context.current_level()
    if level.barrier is not None:  # barrier is None out of a parallel region
        level.barrier.wait()


# submit a new task to the queue
def task_submit(f: Callable, if_: bool = True):
    if if_:
        _omp_context.current_level().task_queue.put(f)


# execute all pending task in queue
def taskwait():
    level = _omp_context.current_level()
    if level.task_queue.qsize() == 0:
        return
    while True:
        try:
            task = level.task_queue.get_nowait()
            task()
        except queue.Empty:
            break


# return True for the master thread
def master():
    return _omp_context.current_level().thread_num == 0


# create a single context
def single(id: str, nowait=False):
    class Single:

        def __init__(self):
            self.copyprivate: queue.Queue[Callable] = queue.Queue()
            self.executed: bool = False

        def __enter__(self):
            return self

        # only for synchronization if not nowait
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not nowait:
                _omp_context.current_level().barrier.wait()

        # execute the single if was not executed
        def __call__(self):
            if not self.executed:
                with level_lock():
                    if not self.executed:
                        self.executed = True
                        return True
            return False

        # register function to update varialbes
        def copy_to(self, f: Callable):
            self.copyprivate.put(f)

        # update variables with arguments
        def copy_from(self, *args):
            for i in range(_omp_context.current_level().num_threads - 1):
                self.copyprivate.get()(*args)

    return with_shared_obj(id, lambda: Single())


# copy threadprivate variables from master
def copyin(id: str, *args: str) -> List:
    values = with_shared_obj(id, lambda: list())
    if _omp_context.current_level().thread_num == 0:
        for i, arg in enumerate(args):
            values.append(getattr(var, arg))
    barrier()
    if _omp_context.current_level().thread_num > 0:
        for i, arg in enumerate(args):
            setattr(var, arg, values[i])
