import pytest

from omp4py import *
from queue import Queue


def section(q: Queue):
    with omp("parallel"):
        with omp("sections"):
            with omp("section"):
                q.put(0)


def test_section():
    q = Queue()
    omp_set_num_threads(2)
    omp(section)(q)
    assert sorted(q.queue) == [0]


################################################


def section2(q: Queue):
    with omp("parallel"):
        with omp("sections"):
            with omp("section"):
                q.put(0)
            with omp("section"):
                q.put(1)


def test_section2():
    q = Queue()
    omp_set_num_threads(2)
    omp(section2)(q)
    assert sorted(q.queue) == [0, 1]


################################################

def section_parallel(q: Queue):
    with omp("parallel sections"):
        with omp("section"):
            q.put(0)


def test_section_paralle():
    q = Queue()
    omp_set_num_threads(2)
    omp(section_parallel)(q)
    assert sorted(q.queue) == [0]


################################################

def section_nested_error():
    with omp("parallel sections"):
        with omp("section"):
            with omp("section"):
                pass


def test_section_nested_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(section_nested_error)()


################################################

def section_only_error():
    with omp("parallel sections"):
        with omp("section"):
            pass
        x = 1


def test_section_only_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(section_only_error)()


################################################


def section_only_error2():
    with omp("parallel sections"):
        with omp("section"):
            pass
        with open("section"):
            pass


def test_section_only_error2():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(section_only_error2)()


################################################

def section_lastprivate():
    x = 1
    with omp("parallel sections lastprivate(x)"):
        with omp("section"):
            time.sleep(0.2)
            x = 3
        with omp("section"):
            x = 2
    return x


def test_section_lastprivate():
    omp_set_num_threads(2)
    assert omp(section_lastprivate)() == 2


################################################

def section_reduction():
    x = 1
    with omp("parallel sections reduction(+:x)"):
        with omp("section"):
            x = 3
        with omp("section"):
            x = 2
    return x


def test_section_reduction():
    omp_set_num_threads(2)
    assert omp(section_reduction)() == 6


################################################

def section_var_dup_error():
    x = 1
    with omp("parallel sections shared(x) reduction(+:x)"):
        with omp("section"):
            x = 3
        with omp("section"):
            x = 2
    return x


def test_section_var_dup_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(section_var_dup_error)()


################################################


def section_nowait(q: Queue):
    with omp("parallel"):
        with omp("sections nowait"):
            with omp("section"):
                while q.qsize() == 0:
                    time.sleep(0.1)
                q.put(0)
        q.put(1)


def test_section_nowait():
    q = Queue()
    omp_set_num_threads(2)
    omp(section_nowait)(q)
    assert list(q.queue) == [1, 0, 1]

################################################
