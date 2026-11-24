"""`master` and `masked`: blocks that only one thread of the team executes, without a barrier at the end.

`master` runs on thread 0 and `masked` on the thread of its `filter` (thread 0 by default).
"""

import threading

import pytest

from test.helpers import omp_function

# ---- master and masked -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("directive", ["master", "masked"])
def test_only_thread_zero_executes_the_block(directive):
    f = omp_function(
        f"""
        def f(out):
            with omp("parallel num_threads(4)"):
                with omp("{directive}"):
                    out.append(omp_get_thread_num())
        """,
    )
    out = []
    f(out)
    assert out == [0]


@pytest.mark.parametrize("thread", [0, 1, 3])
def test_masked_filter_selects_the_thread(thread):
    f = omp_function(
        f"""
        def f(out):
            with omp("parallel num_threads(4)"):
                with omp("masked filter({thread})"):
                    out.append(omp_get_thread_num())
        """,
    )
    out = []
    f(out)
    assert out == [thread]


def test_masked_filter_is_an_expression():
    f = omp_function(
        """
        def f(out, k):
            with omp("parallel num_threads(4)"):
                with omp("masked filter(k + 1)"):
                    out.append(omp_get_thread_num())
        """,
    )
    out = []
    f(out, 1)
    assert out == [2]


def test_masked_filter_that_is_not_a_thread_of_the_team_executes_nothing():
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(2)"):
                with omp("masked filter(5)"):
                    out.append(omp_get_thread_num())
        """,
    )
    out = []
    f(out)
    assert out == []


@pytest.mark.parametrize("directive", ["master", "masked"])
def test_the_other_threads_do_not_wait_for_the_block(directive):
    """Thread 0 waits inside the block until the other threads have gone past it: there is no barrier."""
    passed = threading.Barrier(3, timeout=10)
    f = omp_function(
        f"""
        def f(passed):
            with omp("parallel num_threads(3)"):
                with omp("{directive}"):
                    passed.wait()
                if omp_get_thread_num() != 0:
                    passed.wait()
        """,
    )
    f(passed)
