import time

import pytest

from omp4py import *
from queue import Queue


def task(q: Queue):
    with omp("parallel"):
        with omp("single"):
            with omp("task"):
                q.put(0)


def test_task():
    q = Queue()
    omp_set_num_threads(2)
    omp(task)(q)
    assert sorted(q.queue) == [0]


################################################


def task_shared(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task shared(x)"):
                x = 2
    q.put(x)


def test_task_shared():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_shared)(q)
    assert sorted(q.queue) == [2]


################################################


def task_default_shared(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task default(shared)"):
                x = 2
    q.put(x)


def test_task_default_shared():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_default_shared)(q)
    assert sorted(q.queue) == [2]


################################################


def task_default_none(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task default(none)"):
                pass
    q.put(x)


def test_task_default_none():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_default_none)(q)
    assert sorted(q.queue) == [1]


################################################


def task_default_none_error():
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task default(none)"):
                x = 2


def test_task_default_none_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(task_default_none_error)()


################################################


def task_private_shared_error():
    x = 1
    with omp("parallel private(x)"):
        with omp("single"):
            with omp("task shared(x)"):
                x = 2


def test_task_private_shared_error():
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(task_private_shared_error)()


################################################


def task_private(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task private(x)"):
                x = 2
    q.put(x)


def test_task_private():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_private)(q)
    assert sorted(q.queue) == [1]


################################################


def task_firstprivate(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task firstprivate(x)"):
                x += 2
    q.put(x)


def test_task_firstprivate():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_firstprivate)(q)
    assert sorted(q.queue) == [1]


################################################


def task_if(q: Queue, enable: bool):
    x = 10
    with omp("parallel"):
        with omp("single"):
            q.put(omp_get_thread_num())
            with omp("task if(enable)"):
                q.put(omp_get_thread_num())


def test_task_if_false():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_if)(q, False)
    assert list(q.queue)[0] == list(q.queue)[1]


def test_task_if_true():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_if)(q, True)
    assert list(q.queue)[0] != 10


################################################


def task_taskwait(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task"):
                x += 2
        with omp("taskwait"):
            pass
        q.put(x)


def test_task_taskwait():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_taskwait)(q)
    assert sorted(q.queue) == [3, 3]


################################################


def task_taskwait_body_error(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task"):
                x += 2
        with omp("taskwait"):
            q.put(x)


def test_task_taskwait_body_error():
    q = Queue()
    omp_set_num_threads(2)
    with pytest.raises(OmpSyntaxError):
        omp(task_taskwait_body_error)(q)


################################################


def task_taskwait_no_with(q: Queue):
    x = 1
    with omp("parallel"):
        with omp("single"):
            with omp("task"):
                x += 2
        omp("taskwait")
        q.put(x)


def test_task_taskwait_no_with():
    q = Queue()
    omp_set_num_threads(2)
    omp(task_taskwait_no_with)(q)
    assert sorted(q.queue) == [3, 3]

################################################
