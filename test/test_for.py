import pytest

from omp4py import *
from queue import Queue


def simple_for(q: Queue):
    with omp("parallel"):
        with omp("for"):
            for i in range(10):
                q.put(i)


def test_simple_for():
    q = Queue()
    omp_set_num_threads(2)
    omp(simple_for)(q)
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