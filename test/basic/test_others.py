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


def atomic():
    x = 0
    with omp("parallel"):
        for i in range(5):
            with omp("atomic"):
                x += 1
    return x


def test_atomic():
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


def test_atomic_error():
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


def test_atomic_error2():
    omp_set_num_threads(2)
    with pytest.raises(SyntaxError):
        omp(atomic_error2)()


################################################


def master(q: Queue):
    x = 0
    with omp("parallel firstprivate(x)"):
        with omp("master"):
            x = omp_get_thread_num()
        omp("barrier")
        q.put(x)

    return x


def test_master():
    q = Queue()
    omp_set_num_threads(2)
    omp(master)(q)
    assert list(q.queue) == [0, 0]

################################################


################################################


def sample_func_missing_file():
    x = 0
    with omp("parallel num_threads(2) reduction(+:x)"):
        x += 1
    return x


def test_missing_module_file():
    import sys
    mod = sys.modules[sample_func_missing_file.__module__]
    old_file = getattr(mod, "__file__", None)
    try:
        if hasattr(mod, "__file__"):
            del mod.__file__
        assert not hasattr(mod, "__file__")
        compiled = omp(sample_func_missing_file)
        assert compiled() == 2
    finally:
        if old_file is not None:
            mod.__file__ = old_file

################################################
