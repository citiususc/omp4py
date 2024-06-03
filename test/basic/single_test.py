import pytest

from omp4py import *
from queue import Queue


def single(q: Queue):
    with omp("parallel"):
        with omp("single"):
            q.put(0)


def test_single():
    q = Queue()
    omp_set_num_threads(2)
    omp(single)(q)
    assert sorted(q.queue) == [0]


################################################


def single_private(q: Queue):
    x = 0
    with omp("parallel"):
        with omp("single private(x)"):
            x = 1
    q.put(x)


def test_single_private():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_private)(q)
    assert sorted(q.queue) == [0]


################################################


def single_firstprivate(q: Queue):
    x = 0
    with omp("parallel"):
        with omp("single firstprivate(x)"):
            x += 1
    q.put(x)


def test_single_firstprivate():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_firstprivate)(q)
    assert sorted(q.queue) == [0]


################################################


def single_private_private(q: Queue):
    x = 0
    with omp("parallel private(x)"):
        x = 1
        with omp("single private(x)"):
            x = 2
            q.put(x)
        q.put(x)
    q.put(x)


def test_single_private_private():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_private_private)(q)
    assert sorted(q.queue) == [0, 1, 1, 2]


################################################

def single_firstprivate_firstprivate(q: Queue):
    x = 0
    with omp("parallel firstprivate(x)"):
        x += 1
        with omp("single firstprivate(x)"):
            x += 1
            q.put(x)
        q.put(x)
    q.put(x)


def test_single_firstprivate_firstprivate():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_firstprivate_firstprivate)(q)
    assert sorted(q.queue) == [0, 1, 1, 2]


################################################


def single_var_dup_error():
    x = 0
    with omp("parallel"):
        with omp("single private(x) firstprivate(x)"):
            x = 1


def test_var_dup_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(single_var_dup_error)(q)


################################################


def single_copyprivate(q: Queue):
    x = 0
    with omp("parallel private(x)"):
        with omp("single copyprivate(x)"):
            x = 4
        q.put(x)


def test_single_copyprivate():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_copyprivate)(q)
    assert sorted(q.queue) == [4, 4]

################################################


def single_nowait(q: Queue):
    with omp("parallel"):
        with omp("single nowait"):
                while q.qsize() == 0:
                    time.sleep(0.1)
                q.put(0)
        q.put(1)


def test_single_nowait():
    q = Queue()
    omp_set_num_threads(2)
    omp(single_nowait)(q)
    assert list(q.queue) == [1, 0, 1]

################################################
