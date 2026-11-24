"""`single`: a block that any one thread of the team executes.

The other threads wait for it at the end unless the construct has `nowait`, and `copyprivate` sends the value
of its variables to the other threads.
"""

import threading

import pytest

from test.helpers import omp_function

# ---- single ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 5])
def test_the_block_is_executed_by_one_thread(threads):
    f = omp_function(
        f"""
        def f(out):
            with omp("parallel num_threads({threads})"):
                with omp("single"):
                    out.append(omp_get_thread_num())
        """,
    )
    out = []
    f(out)
    assert len(out) == 1
    assert 0 <= out[0] < threads


def test_every_time_that_the_team_gets_to_the_construct_one_thread_executes_it():
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                for i in range(6):
                    with omp("single"):
                        out.append(i)
        """,
    )
    out = []
    f(out)
    assert out == list(range(6))


def test_the_other_threads_wait_for_the_block_to_finish():
    f = omp_function(
        """
        from time import sleep

        def f(out):
            done = []
            with omp("parallel num_threads(3)"):
                with omp("single"):
                    for _ in range(3):
                        sleep(0.05)
                    done.append(True)
                out.append(bool(done))
        """,
    )
    out = []
    f(out)
    assert out == [True] * 3


def test_nowait_lets_the_other_threads_go_on():
    """The thread that executes the block waits for another one to have gone past the construct."""
    passed = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(passed, out):
            with omp("parallel num_threads(2)"):
                with omp("single nowait"):
                    out.append(omp_get_thread_num())
                    passed.wait()
                if len(out) == 1 and omp_get_thread_num() != out[0]:
                    passed.wait()
        """,
    )
    out = []
    f(passed, out)
    assert len(out) == 1


def test_private_variables_of_a_single():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(2)"):
                with omp("single private(x)"):
                    x = 1
            return x
        """,
    )
    assert f() == 0


def test_firstprivate_variables_of_a_single():
    f = omp_function(
        """
        def f(out):
            x = 7
            with omp("parallel num_threads(2)"):
                with omp("single firstprivate(x)"):
                    x += 1
                    out.append(x)
            return x
        """,
    )
    out = []
    assert f(out) == 7
    assert out == [8]


def test_copyprivate_sends_the_value_to_the_other_threads():
    f = omp_function(
        """
        def f(out):
            x = 0
            y = 0
            with omp("parallel private(x, y) num_threads(4)"):
                with omp("single copyprivate(x, y)"):
                    x = 4
                    y = omp_get_thread_num() + 100
                out.append((x, y - 100))
            return out
        """,
    )
    out = f([])
    assert {x for x, _ in out} == {4}
    assert len({thread for _, thread in out}) == 1


def test_a_variable_cannot_be_in_two_clauses():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                x = 0
                with omp("parallel"):
                    with omp("single private(x) firstprivate(x)"):
                        x = 1
            """,
        )


def test_copyprivate_cannot_be_used_with_nowait():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                x = 0
                with omp("parallel private(x)"):
                    with omp("single copyprivate(x) nowait"):
                        x = 1
            """,
        )
