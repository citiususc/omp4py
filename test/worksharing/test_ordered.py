"""The `ordered` clause: the `ordered` regions of a loop run in the order of the iterations, while the rest of each
iteration runs in parallel."""

import threading

import pytest

from test.helpers import omp_function

SCHEDULES = [
    "",
    "schedule(static)",
    "schedule(static, 1)",
    "schedule(static, 3)",
    "schedule(dynamic)",
    "schedule(dynamic, 3)",
    "schedule(guided)",
]
SPACES = [range(1), range(10), range(37), range(3, 40, 7), range(20, -1, -3)]

ORDERED = """
def f(out):
    with omp("parallel for ordered $schedule num_threads($threads)"):
        for i in $space:
            if $condition:
                with omp("ordered"):
                    out.append(i)
"""


def run(schedule="", space=range(10), threads=3, condition="True") -> list[int]:
    out = []
    omp_function(ORDERED, schedule=schedule, space=repr(space), threads=threads, condition=condition)(out)
    return out


@pytest.mark.parametrize("space", SPACES, ids=repr)
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_regions_run_in_the_order_of_the_iterations(schedule, threads, space):
    assert run(schedule, space, threads) == list(space)


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("condition", ["i % 2 == 0", "i % 7 == 0", "i < 5", "i > 50", "False"])
def test_iterations_that_do_not_reach_the_region_do_not_hold_the_others_back(schedule, condition):
    space = range(60)
    assert run(schedule, space, 4, condition) == [i for i in space if eval(condition)]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_region_sees_what_the_previous_ones_wrote(schedule):
    f = omp_function(
        f"""
        def f(acc):
            with omp("parallel for ordered {schedule} num_threads(4)"):
                for i in range(40):
                    with omp("ordered"):
                        acc.append(acc[-1] + i)
        """,
    )
    acc = [0]
    f(acc)
    assert acc == [sum(range(k)) for k in range(41)]


def test_for_directive_inside_a_parallel_region():
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                with omp("for ordered schedule(dynamic)"):
                    for i in range(20):
                        with omp("ordered"):
                            out.append(i)
        """,
    )
    out = []
    f(out)
    assert out == list(range(20))


def test_code_outside_the_region_runs_in_parallel():
    """Iterations 0 and 1 wait for each other before their regions, so their code outside runs at the same time."""
    barrier = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(out, barrier):
            with omp("parallel for ordered schedule(static, 1) num_threads(2)"):
                for i in range(6):
                    if i < 2:
                        barrier.wait()
                    with omp("ordered"):
                        out.append(i)
        """,
    )
    out = []
    f(out, barrier)
    assert out == list(range(6))


# ---- collapse ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("shape", [(4, 5), (1, 6), (7, 1), (3, 3)])
def test_collapsed_loops_run_in_the_order_of_the_sequential_loops(schedule, shape):
    rows, columns = shape
    out = []
    omp_function(
        f"""
        def f(out):
            with omp("parallel for ordered collapse(2) {schedule} num_threads(3)"):
                for i in range({rows}):
                    for j in range({columns}):
                        with omp("ordered"):
                            out.append((i, j))
        """,
    )(out)
    assert out == [(i, j) for i in range(rows) for j in range(columns)]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_collapsed_loops_with_regions_that_are_not_always_reached(schedule):
    out = []
    omp_function(
        f"""
        def f(out):
            with omp("parallel for ordered collapse(2) {schedule} num_threads(4)"):
                for i in range(6):
                    for j in range(5):
                        if (i + j) % 3 == 0:
                            with omp("ordered"):
                                out.append((i, j))
        """,
    )(out)
    assert out == [(i, j) for i in range(6) for j in range(5) if (i + j) % 3 == 0]
