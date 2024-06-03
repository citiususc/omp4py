import pytest

from omp4py import *
from queue import Queue


def parallel(q: Queue):
    with omp("parallel"):
        q.put(omp_get_thread_num())


def test_parallel():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel)(q)
    assert sorted(q.queue) == [0, 1]


################################################
def parallel_num_threads(q: Queue, n: int):
    with omp("parallel num_threads(n)"):
        q.put(omp_get_thread_num())


def test_parallel_num_threads():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_num_threads)(q, 3)
    assert sorted(q.queue) == [0, 1, 2]


def test_parallel_num_threads_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpError):
        omp(parallel_num_threads)(q, 0)


################################################
def parallel_if(q: Queue, e: bool):
    with omp("parallel if(e)"):
        q.put(omp_get_thread_num())


def test_parallel_if_true():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_if)(q, True)
    assert sorted(q.queue) == [0, 1]


def test_parallel_if_false():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_if)(q, False)
    assert sorted(q.queue) == [0]


################################################


def parallel_shared():
    x = 0
    with omp("parallel"):
        x = 1
    return x


def test_parallel_shared():
    omp_set_num_threads(2)
    x = omp(parallel_shared)()
    assert x == 1


################################################


def parallel_private():
    x = 0
    with omp("parallel private(x)"):
        x = 1
    return x


def test_parallel_private():
    omp_set_num_threads(2)
    x = omp(parallel_private)()
    assert x == 0


################################################


def parallel_firstprivate(q: Queue):
    x = 2
    with omp("parallel firstprivate(x)"):
        q.put(x)


def test_parallel_firstprivate():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_firstprivate)(q)
    assert sorted(q.queue) == [2, 2]


################################################


def parallel_var_dup_error():
    x = 0
    with omp("parallel private(x) shared(x)"):
        x = 1
    return x


def test_parallel_var_dup_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_var_dup_error)()


################################################


def parallel_shared_dup_error():
    x = 0
    with omp("parallel shared(x) shared(x)"):
        x = 1
    return x


def test_parallel_shared_dup_error():
    omp_set_num_threads(2)
    x = omp(parallel_shared_dup_error)()
    assert x == 1


################################################


def parallel_private_dup():
    x = 0
    with omp("parallel private(x) private(x)"):
        x = 1
    return x


def test_parallel_private_dup():
    omp_set_num_threads(2)
    x = omp(parallel_private_dup)()
    assert x == 0


################################################


def parallel_firstprivate_dup(q: Queue):
    x = 2
    with omp("parallel firstprivate(x) firstprivate(x)"):
        q.put(x)
    return x


def test_parallel_firstprivate_dup():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_firstprivate_dup)(q)
    assert sorted(q.queue) == [2, 2]


################################################


def parallel_shared_noexist_error():
    x = 0
    with omp("parallel shared(y)"):
        x = 1
    return x


def test_parallel_shared_noexist_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_shared_noexist_error)()


################################################


def parallel_firstprivate_noexist_error():
    x = 0
    with omp("parallel firstprivate(y)"):
        x = 1
    return x


def test_parallel_firstprivate_noexist_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_firstprivate_noexist_error)()


################################################


def parallel_private_noexist_no_error():
    x = 0
    with omp("parallel private(y)"):
        x = 1
    return x


def test_parallel_private_noexist_no_error():
    omp_set_num_threads(2)
    omp_set_num_threads(2)
    x = omp(parallel_private_noexist_no_error)()
    assert x == 1


################################################


def parallel_default_shared():
    x = 0
    with omp("parallel default(shared) shared(x)"):
        x = 1
    return x


def test_parallel_default_shared():
    omp_set_num_threads(2)
    x = omp(parallel_default_shared)()
    assert x == 1


################################################


def parallel_default_none():
    x = 0
    with omp("parallel default(none) shared(x)"):
        x = 1
    return x


def test_parallel_default_none():
    omp_set_num_threads(2)
    x = omp(parallel_default_none)()
    assert x == 1


################################################


def parallel_default_none2():
    x = 0
    with omp("parallel for default(none) lastprivate(x)"):
        for i in range(10):
            x = 1
    return x


def test_parallel_default_none2():
    omp_set_num_threads(2)
    x = omp(parallel_default_none2)()
    assert x == 1


################################################


def parallel_default_none3():
    x = 0
    with omp("parallel sections default(none) lastprivate(x)"):
        with omp("section"):
            x = 1
    return x


def test_parallel_default_none3():
    omp_set_num_threads(2)
    x = omp(parallel_default_none3)()
    assert x == 1


################################################


def parallel_default_error():
    x = 0
    with omp("parallel default(none)"):
        x = 1
    return x


def test_parallel_default_error():
    with pytest.raises(OmpSyntaxError):
        omp(parallel_default_error)()


################################################


def parallel_default_private_error():
    x = 0
    with omp("parallel default(private)"):
        x = 1
    return x


def test_parallel_default_private_error():
    with pytest.raises(OmpSyntaxError):
        omp(parallel_default_private_error)()


################################################


def parallel_default_no_error():
    x = 0
    with omp("parallel default(none)"):
        y = 1
    return x


def test_parallel_default_no_error():
    omp_set_num_threads(2)
    x = omp(parallel_default_no_error)()
    assert x == 0


################################################


def parallel_reduction():
    x = 0
    with omp("parallel reduction(+:x)"):
        x = 1
    return x


def test_parallel_reduction():
    omp_set_num_threads(2)
    x = omp(parallel_reduction)()
    assert x == 2


################################################


def parallel_reduction_bool():
    x = 0
    with omp("parallel reduction(||:x)"):
        x = True
    return x


def test_parallel_reduction_bool():
    omp_set_num_threads(2)
    x = omp(parallel_reduction_bool)()
    assert x


################################################


def parallel_reduction_multiple():
    x = 0
    y = 1
    with omp("parallel reduction(+:x) reduction(*:y)"):
        x = 1
        y = 2
    return x, y


def test_parallel_reduction_multiple():
    omp_set_num_threads(2)
    (x, y) = omp(parallel_reduction_multiple)()
    assert (x, y) == (2, 4)


################################################


def parallel_reduction_error():
    x = 0
    y = 1
    with omp("parallel reduction(+:x, *:y)"):
        x = 1
        y = 2
    return x, y


def test_parallel_reduction_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_reduction_error)()


################################################


def parallel_reduction_format_error():
    x = 0
    with omp("parallel reduction(x)"):
        x = 1
    return x


def test_parallel_reduction_format_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_reduction_format_error)()


################################################


def parallel_reduction_op_error():
    x = 0
    with omp("parallel reduction(sum:x)"):
        x = 1
    return x


def test_parallel_reduction_op_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_reduction_op_error)()


################################################


def parallel_reduction_dup():
    x = 0
    with omp("parallel reduction(+:x,x,x)"):
        x = 1
    return x


def test_parallel_reduction_dup():
    omp_set_num_threads(2)
    x = omp(parallel_reduction_dup)()
    assert x == 2


################################################


def parallel_reduction_dup_error():
    x = 0
    with omp("parallel reduction(+:x) reduction(*:x)"):
        x = 2
    return x


def test_parallel_reduction_dup_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_reduction_dup_error)()


################################################


def parallel_return():
    x = 0
    with omp("parallel"):
        def f():
            return 2

        async def f2():
            return None

        x = f()
    return x


def test_parallel_return():
    omp_set_num_threads(2)
    x = omp(parallel_return)()
    assert x == 2


################################################


def parallel_return_error():
    x = 0
    with omp("parallel"):
        x = 2
        return x


def test_parallel_return_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_return_error)()


################################################


def parallel_yield_error():
    x = 0
    with omp("parallel"):
        x = 2
        yield x


def test_parallel_yield_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_yield_error)()


################################################


def parallel_yieldfrom_error():
    x = 0
    with omp("parallel"):
        x = 2
        yield from range(10)


def test_parallel_yieldfrom_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(parallel_yieldfrom_error)()


################################################


def parallel_raise_error():
    with omp("parallel"):
        x = 0 / 0


@pytest.mark.filterwarnings("ignore:Exception in thread")
def test_parallel_raise_error():
    omp_set_num_threads(2)
    with pytest.raises(ArithmeticError):
        omp(parallel_raise_error)()


################################################


def parallel_copyin_set():
    omp("threadprivate(x)")

    with omp("parallel num_threads(1)"):
        x = 2


def parallel_copyin_get(q: Queue):
    omp("threadprivate(x)")

    with omp("parallel num_threads(2) copyin(x)"):
        q.put(x)


def test_parallel_copyin():
    q = Queue()
    omp_set_num_threads(2)
    omp(parallel_copyin_set)()
    omp(parallel_copyin_get)(q)
    assert sorted(q.queue) == [2, 2]

################################################
