"""The worksharing loop (`for`): every iteration of the loop is executed exactly once by some thread of the team,
whatever the iteration space and the number of threads."""

import pytest

from test.helpers import omp_function

SPACES = [range(0), range(1), range(10), range(5, 5), range(3, 40, 7), range(20, -1, -3), range(100)]
THREADS = [1, 2, 3, 7]

COMBINED = """
def f(out):
    with omp("parallel for num_threads($threads)"):
        for i in $space:
            out.append(i)
"""

SEPARATE = """
def f(out):
    with omp("parallel num_threads($threads)"):
        with omp("for"):
            for i in $space:
                out.append(i)
"""


@pytest.mark.parametrize("template", [COMBINED, SEPARATE], ids=["parallel for", "parallel + for"])
@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("space", SPACES, ids=repr)
def test_every_iteration_runs_once(template, threads, space):
    out = []
    omp_function(template, threads=threads, space=repr(space))(out)
    assert sorted(out) == sorted(space)


def test_bounds_are_expressions():
    f = omp_function("""
        def f(out, start, stop, step):
            with omp("parallel for num_threads(3)"):
                for i in range(start * 2, stop + 1, step - 1):
                    out.append(i)
    """)
    out = []
    f(out, 1, 20, 4)
    assert sorted(out) == sorted(range(2, 21, 3))


def test_for_without_parallel_runs_in_the_calling_thread():
    """A `for` that is not inside a parallel region binds to a team of one thread."""
    f = omp_function("""
        def f(out):
            with omp("for"):
                for i in range(6):
                    out.append((i, omp_get_thread_num(), omp_get_num_threads()))
    """)
    out = []
    f(out)
    assert out == [(i, 0, 1) for i in range(6)]


def test_loop_variable_is_private():
    f = omp_function("""
        def f():
            i = -1
            with omp("parallel for num_threads(3)"):
                for i in range(10):
                    pass
            return i
    """)
    assert f() == -1


def test_every_thread_of_the_team_takes_part():
    """With a static schedule and a chunk of one iteration, each of the threads gets iterations."""
    f = omp_function("""
        def f(out):
            with omp("parallel for schedule(static, 1) num_threads(4)"):
                for i in range(8):
                    out.append(omp_get_thread_num())
    """)
    out = []
    f(out)
    assert sorted(set(out)) == [0, 1, 2, 3]


def test_a_nested_loop_inside_the_body_is_not_shared():
    """Only the loop of the directive is split among the threads; loops inside its body run whole."""
    f = omp_function("""
        def f(out):
            with omp("parallel for num_threads(3)"):
                for i in range(4):
                    for j in range(3):
                        out.append((i, j))
    """)
    out = []
    f(out)
    assert sorted(out) == [(i, j) for i in range(4) for j in range(3)]
