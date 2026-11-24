"""The `collapse` clause: the iterations of several perfectly nested loops form one iteration space that is
distributed among the threads, in the order of the sequential execution."""

import pytest

from test.helpers import omp_function

NESTED_2 = """
def f(out):
    with omp("parallel for collapse(2) $schedule num_threads($threads)"):
        for i in $outer:
            for j in $inner:
                out.append((i, j, omp_get_thread_num()))
"""

NESTED_3 = """
def f(out):
    with omp("parallel for collapse(3) $schedule num_threads(3)"):
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    out.append((i, j, k, omp_get_thread_num()))
"""

SCHEDULES = ["", "schedule(static, 1)", "schedule(static, 5)", "schedule(dynamic, 3)", "schedule(guided)"]


def run(schedule: str, outer: range, inner: range, threads: int) -> list[tuple[int, int, int]]:
    out = []
    omp_function(NESTED_2, schedule=schedule, outer=repr(outer), inner=repr(inner), threads=threads)(out)
    return out


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("threads", [1, 2, 3])
@pytest.mark.parametrize(
    ("outer", "inner"),
    [(range(4), range(5)), (range(1), range(7)), (range(6), range(1)), (range(0), range(4)), (range(3), range(0))],
    ids=repr,
)
def test_every_pair_runs_once(schedule, threads, outer, inner):
    out = run(schedule, outer, inner, threads)
    assert sorted((i, j) for i, j, _ in out) == [(i, j) for i in outer for j in inner]


@pytest.mark.parametrize("threads", [2, 3])
def test_steps_of_the_loops(threads):
    out = run("", range(2, 20, 5), range(9, 0, -4), threads)
    assert sorted((i, j) for i, j, _ in out) == sorted((i, j) for i in range(2, 20, 5) for j in range(9, 0, -4))


def test_chunks_follow_the_order_of_the_sequential_loops():
    """With a static schedule, the collapsed space is cut into chunks in the order of the sequential execution."""
    out = run("schedule(static, 3)", range(4), range(5), 3)
    owners = {(i, j): t for i, j, t in out}
    for k, (i, j) in enumerate((i, j) for i in range(4) for j in range(5)):
        assert owners[i, j] == (k // 3) % 3


def test_three_loops():
    out = []
    omp_function(NESTED_3, schedule="schedule(dynamic, 5)")(out)
    assert sorted(t[:3] for t in out) == [(i, j, k) for i in range(3) for j in range(4) for k in range(2)]


def test_without_collapse_only_the_outer_loop_is_distributed():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for schedule(static, 1) num_threads(2)"):
                for i in range(2):
                    for j in range(10):
                        out.append((i, omp_get_thread_num()))
        """,
    )(out)
    assert {t for i, t in out if i == 0} == {0}
    assert {t for i, t in out if i == 1} == {1}


# ---- errors ------------------------------------------------------------------------------------------------------


def test_the_loops_must_be_perfectly_nested():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for collapse(2)"):
                    for i in range(2):
                        for j in range(10):
                            pass
                        bad = 1
            """,
        )


def test_there_must_be_as_many_loops_as_collapse_says():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for collapse(2)"):
                    for i in range(10):
                        pass
            """,
        )


def test_the_bounds_of_an_inner_loop_cannot_depend_on_an_outer_one():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for collapse(2)"):
                    for i in range(10):
                        for j in range(i):
                            pass
            """,
        )


def test_collapse_needs_a_number_written_in_the_directive():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f(n):
                with omp("parallel for collapse(n)"):
                    for i in range(2):
                        for j in range(10):
                            pass
            """,
        )
