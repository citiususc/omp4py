import pytest

from omp4py import *
from queue import Queue


def barrier(q: Queue):
    x = 1
    with omp("parallel"):
        x = 2
        with omp("barrier"):
            pass
        q.put(x)


def test_barrier():
    q = Queue()
    omp_set_num_threads(2)
    omp(barrier)(q)
    assert sorted(q.queue) == [2, 2]


################################################


def barrier_no_with(q: Queue):
    x = 1
    with omp("parallel"):
        x = 2
        omp("barrier")
        q.put(x)


def test_barrier_no_with():
    q = Queue()
    omp_set_num_threads(2)
    omp(barrier_no_with)(q)
    assert sorted(q.queue) == [2, 2]


################################################


def barrier_no_with_error(q: Queue):
    x = 1
    with omp("parallel"):
        x = 2
        omp("barrier some")
        q.put(x)


def test_barrier_no_with_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(barrier_no_with_error)(q)


################################################


def barrier_body_error(q: Queue):
    x = 1
    with omp("parallel"):
        x = 2
        with omp("barrier"):
            q.put(x)


def test_barrier_body_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(barrier_body_error)(q)

################################################
