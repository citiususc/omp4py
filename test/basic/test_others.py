import pytest

from omp4py import *
from queue import Queue


def critical():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("critical"):
                x += 1
    return x


def test_critical():
    omp_set_num_threads(2)
    assert omp(critical)() == 10


################################################

def critical_named():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("critical"):
                with omp("critical(named)"):
                    x += 1
    return x


def test_critical_named():
    omp_set_num_threads(2)
    assert omp(critical_named)() == 10


################################################

def atomic():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("atomic"):
                x += 1
    return x


def _test_atomic(): #TODO
    omp_set_num_threads(2)
    assert omp(atomic)() == 10


################################################


def atomic_error():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("atomic"):
                x = x + 1
    return x


def _test_atomic_error():#TODO
    omp_set_num_threads(2)
    with pytest.raises(SyntaxError):
        omp(atomic_error)()


################################################


def atomic_error2():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("atomic"):
                x += x + 1
    return x


def _test_atomic_error2():#TODO
    omp_set_num_threads(2)
    with pytest.raises(SyntaxError):
        omp(atomic_error2)()


################################################


def master(q: Queue):
    with omp("parallel"):
        with omp("master"):
            q.put(omp_get_thread_num())


def test_master():
    q = Queue()
    omp_set_num_threads(2)
    omp(master)(q)
    assert list(q.queue) == [0]

################################################
