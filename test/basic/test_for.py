import threading
import time

import pytest

from omp4py import *
from queue import Queue
from typing import Callable


def for_(q: Queue):
    with omp("parallel"):
        with omp("for"):
            for i in range(10):
                q.put(i)


def test_for():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_)(q)
    assert sorted(q.queue) == list(range(10))


################################################


def for_no_loop_error(q: Queue):
    with omp("parallel for"):
        q.put(0)


def test_for_no_loop_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_no_loop_error)()


################################################


def for_parallel(q: Queue):
    with omp("parallel for"):
        for i in range(10):
            q.put(i)


def test_for_parallel():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_parallel)(q)
    assert sorted(q.queue) == list(range(10))


################################################


def for_no_range_error(q: Queue):
    elems = list(range(10))
    with omp("parallel for"):
        for i in elems:
            q.put(i)


def test_for_no_range_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_no_range_error)()


################################################


def for_schedule_static(q: Queue):
    with omp("parallel for schedule(static)"):
        for i in range(11):
            q.put((i, omp_get_thread_num()))


def test_for_schedule_static():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_static)(q)
    assert sorted(q.queue) == list(zip(range(11), [0] * 6 + [1] * 5))


################################################


def for_schedule_static_1(q: Queue):
    with omp("parallel for schedule(static, 1)"):
        for i in range(11):
            q.put((i, omp_get_thread_num()))


def test_for_schedule_static_1():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_static_1)(q)
    assert sorted(q.queue) == list(zip(range(11), [0] + [1, 0] * 5))


################################################


def for_schedule_runtime_as_static(q: Queue):
    with omp("parallel for schedule(runtime)"):
        for i in range(11):
            q.put((i, omp_get_thread_num()))


def test_for_schedule_runtime_as_static():
    q = Queue()
    omp_set_num_threads(2)
    omp_set_schedule(omp_sched_static, -1)
    omp(for_schedule_runtime_as_static)(q)
    assert sorted(q.queue) == list(zip(range(11), [0] * 6 + [1] * 5))


################################################


def for_schedule_runtime_chunk_error(q: Queue):
    with omp("parallel for schedule(runtime, 1)"):
        for i in range(11):
            q.put((i, omp_get_thread_num()))


def test_for_schedule_runtime_chunk_error():
    q = Queue()
    omp_set_num_threads(2)
    omp_set_schedule(omp_sched_static, -1)
    with pytest.raises(OmpSyntaxError):
        omp(for_schedule_runtime_chunk_error)(q)


################################################


def for_schedule_auto_chunk_error(q: Queue):
    with omp("parallel for schedule(auto, 1)"):
        for i in range(11):
            q.put((i, omp_get_thread_num()))


def test_for_schedule_auto_chunk_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_schedule_auto_chunk_error)(q, "1")


################################################


def for_schedule_auto(q: Queue):
    with omp("parallel for schedule(auto)"):
        for i in range(11):
            q.put(omp_get_thread_num())


def test_for_schedule_auto():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_auto)(q)
    assert 0 < sorted(q.queue).count(0) < 11


################################################


def for_schedule_dynamic(q: Queue):
    with omp("parallel for schedule(dynamic)"):
        for i in range(11):
            q.put(omp_get_thread_num())
            while q.qsize() < 2:
                time.sleep(0.1)


def test_for_schedule_dynamic():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_dynamic)(q)
    assert 0 < sorted(q.queue).count(0) < 11


################################################


def for_schedule_dynamic_3(q: Queue):
    with omp("parallel for schedule(dynamic, 3)"):
        for i in range(12):
            q.put(omp_get_thread_num())
            while q.qsize() < 2:
                time.sleep(0.1)


def test_for_schedule_dynamic_3():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_dynamic_3)(q)
    assert "000" in "".join(map(str, sorted(q.queue)))


################################################


def for_schedule_guided(q: Queue):
    with omp("parallel for schedule(guided)"):
        for i in range(11):
            q.put(omp_get_thread_num())
            while q.qsize() < 2:
                time.sleep(0.1)


def test_for_schedule_guided():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_guided)(q)
    assert 0 < sorted(q.queue).count(0) < 11


################################################


def for_schedule_var(q: Queue, var):
    with omp("parallel for schedule(static, var)"):
        for i in range(10):
            q.put(omp_get_thread_num())


def test_for_schedule_var():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_schedule_var)(q, 1)
    assert len(q.queue) == 10


@pytest.mark.filterwarnings("ignore:Exception in thread")
def test_for_schedule_var_str_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpError):
        omp(for_schedule_var)(q, "1")


################################################


def for_schedule_as_var_error(q: Queue, var_schedule):
    with omp("parallel for schedule(var_schedule, 1)"):
        for i in range(10):
            q.put(omp_get_thread_num())


@pytest.mark.filterwarnings("ignore:Exception in thread")
def test_for_schedule_as_var_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_schedule_as_var_error)(q, "static")


################################################


def for_collapse_1(q: Queue):
    with omp("parallel for collapse(1)"):
        for i in range(10):
            q.put(i)


def test_for_collapse_1():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_collapse_1)(q)
    assert sorted(q.queue) == list(range(10))


################################################


def for_collapse_2(q: Queue):
    with omp("parallel for collapse(1) schedule(static,1)"):
        for i in range(2):
            for j in range(10):
                q.put((i, omp_get_thread_num()))


def test_for_collapse_2():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_collapse_2)(q)
    assert list(map(lambda x: x[0], sorted(q.queue))) == [0] * 10 + [1] * 10
    assert (0, 0) in list(q.queue)
    assert (1, 1) in list(q.queue)


################################################


def for_collapse_var_error(q: Queue, n: int):
    with omp("parallel for collapse(n) schedule(static,1)"):
        for i in range(2):
            for j in range(10):
                q.put((i, omp_get_thread_num()))


def test_for_collapse_var_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_collapse_var_error)(q, 2)


################################################


def for_collapse_nested_error(q: Queue):
    with omp("parallel for collapse(2) schedule(static,1)"):
        for i in range(2):
            for j in range(10):
                q.put((i, omp_get_thread_num()))
            bad = 1


def test_for_nested_var_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_collapse_var_error)(q)


################################################


def for_ordered(q: Queue):
    with omp("parallel for ordered"):
        for i in range(10):
            with omp("ordered"):
                q.put(i)


def test_for_ordered():
    q = Queue()
    omp_set_num_threads(2)
    omp(for_ordered)(q)
    assert list(q.queue) == list(range(10))


################################################


def for_ordered_multiple_error(q: Queue):
    with omp("parallel for ordered"):
        for i in range(10):
            with omp("ordered"):
                q.put((i, omp_get_thread_num()))
            print("")
            with omp("ordered"):
                print(i)


def test_for_ordered_multiple_error():
    q = Queue()
    omp_set_num_threads(1)
    with pytest.raises(OmpSyntaxError):
        omp(for_ordered_multiple_error)(q)


################################################


def hidden_ordered(i):
    with omp("ordered"):
        print(i)


def for_ordered_multiple_hidden_error(q: Queue, f: Callable):
    with omp("parallel for ordered"):
        for i in range(10):
            with omp("ordered"):
                q.put((i, omp_get_thread_num()))
            print("")
            f(i)


@pytest.mark.filterwarnings("ignore:Exception in thread")
def test_for_ordered_multiple_hidden_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpError):
        omp(for_ordered_multiple_hidden_error)(q, omp(hidden_ordered))


################################################


def nested_hidden_ordered(i):
    with omp("ordered"):
        print(i)


def for_ordered_multiple_nested_hidden_error(q: Queue, f: Callable):
    with omp("parallel for ordered"):
        for i in range(10):
            with omp("ordered"):
                q.put((i, omp_get_thread_num()))
                f(i)


@pytest.mark.filterwarnings("ignore:Exception in thread")
def test_for_ordered_multiple_nested_hidden_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpError):
        omp(for_ordered_multiple_nested_hidden_error)(q, omp(nested_hidden_ordered))


################################################


def for_nowait(q: Queue, barrier: threading.Barrier):
    with omp("parallel"):
        with omp("for nowait"):
            for i in range(10):
                q.put(i)
                if i == 0:
                    barrier.wait()
        if omp_get_thread_num() != 0:
            barrier.wait()


def test_for_nowait():
    q = Queue()
    barrier = threading.Barrier(2)
    omp_set_num_threads(2)
    omp(for_nowait)(q, barrier)
    assert sorted(q.queue) == list(range(10))


################################################


def for_reduction():
    x = 0
    with omp("parallel"):
        with omp("for reduction(+:x)"):
            for i in range(10):
                x += i
    return x


def test_for_reduction():
    omp_set_num_threads(2)
    assert omp(for_reduction)() == sum(range(10))


################################################


def for_private():
    x = 0
    with omp("parallel"):
        x = 1
        with omp("for private(x)"):
            for i in range(10):
                x = i
    return x


def test_for_private():
    omp_set_num_threads(2)
    assert omp(for_private)() == 1


################################################


def for_private_private():
    x = 0
    y = 0
    with omp("parallel private(x)"):
        x = 1
        with omp("for private(x)"):
            for i in range(10):
                x = i
        y = x
    return x, y


def test_for_private_private():
    omp_set_num_threads(2)
    assert omp(for_private_private)() == (0, 1)


################################################


def for_firstprivate():
    x = 0
    with omp("parallel"):
        x = 1
        with omp("for firstprivate(x)"):
            for i in range(10):
                x += i
    return x


def test_for_firstprivate():
    omp_set_num_threads(2)
    assert omp(for_firstprivate)() == 1


################################################


def for_firstprivate_firstprivate():
    x = 1
    y = 0
    with omp("parallel firstprivate(x)"):
        x += 1
        with omp("for firstprivate(x)"):
            for i in range(10):
                x += i
        y = x
    return x, y


def test_for_firstprivate_firstprivate():
    omp_set_num_threads(2)
    assert omp(for_firstprivate_firstprivate)() == (1, 2)


################################################


def for_reduction_private_error():
    x = 0
    with omp("parallel private(x)"):
        with omp("for reduction(+:x)"):
            for i in range(10):
                x += i
    return x


def test_for_reduction_private_error():
    omp_set_num_threads(2)

    with pytest.raises(OmpSyntaxError):
        omp(for_reduction_private_error)()


def for_var_dup_error():
    x = 0
    with omp("parallel"):
        with omp("for private(x) reduction(+:x)"):
            for i in range(10):
                x += i
    return x


def test_for_var_dup_error():
    omp_set_num_threads(2)

    with pytest.raises(OmpSyntaxError):
        omp(for_var_dup_error)()


################################################


def for_var_dup_error2():
    x = 0
    with omp("parallel for private(x) lastprivate(x)"):
        for i in range(10):
            x += i
    return x


def test_for_var_dup_error2():
    omp_set_num_threads(2)

    with pytest.raises(OmpSyntaxError):
        omp(for_var_dup_error2)()


################################################


def for_lastprivate():
    x = 0
    with omp("parallel for lastprivate(x)"):
        for i in range(10):
            x = i
    return x


def test_for_lastprivate():
    omp_set_num_threads(2)
    assert omp(for_lastprivate)() == 9


################################################


def for_lastprivate_var_error():
    x = 0
    with omp("parallel for lastprivate(y)"):
        for i in range(10):
            x = i
    return x


def test_for_lastprivate_var_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        assert omp(for_lastprivate_var_error)()


################################################


def for_collapse_lastprivate():
    x = 0
    with omp("parallel for lastprivate(x) collapse(2)"):
        for i in range(10):
            for j in range(10):
                x = (i, j)
    return x


def test_for_collapse_lastprivate():
    omp_set_num_threads(2)
    assert omp(for_collapse_lastprivate)() == (9, 9)


################################################


def for_break():
    x = 0
    with omp("parallel for"):
        for i in range(10):
            x = 1
            for j in range(10):
                break
    return x


def test_for_break():
    omp_set_num_threads(2)
    assert omp(for_break)() == 1


################################################


def for_break_error():
    with omp("parallel"):
        with omp("for"):
            for i in range(10):
                if i == 4:
                    break


def test_for_break_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(for_break_error)()

################################################
