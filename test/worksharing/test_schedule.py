"""The `schedule` clause: how the iterations are distributed among the threads.

The standard fixes the distribution only for the static schedule with a chunk size: the chunks are assigned to the
threads in order, round robin. For the other schedules the tests check that every iteration runs once, and the
properties that the standard guarantees (the dynamic schedule gives the chunks on demand, the monotonic modifier
keeps the chunks of a thread in order, `runtime` follows `omp_set_schedule`).
"""

import threading
from math import ceil

import pytest

from omp4py import omp_sched_dynamic, omp_sched_guided, omp_sched_static, omp_set_schedule

from test.helpers import omp_function, raises_in_the_team

SPACES = [range(1), range(10), range(11), range(3, 40, 7), range(20, -1, -3), range(100)]

RECORD = """
def f(out):
    with omp("parallel for $schedule num_threads($threads)"):
        for i in $space:
            out.append((i, omp_get_thread_num()))
"""


def run(schedule: str, space: range, threads: int) -> list[tuple[int, int]]:
    """Run the loop and return the (iteration, thread) pairs in the logical order of the iterations."""
    out = []
    omp_function(RECORD, schedule=schedule, space=repr(space), threads=threads)(out)
    return sorted(out, key=lambda pair: space.index(pair[0]))


@pytest.mark.parametrize("space", SPACES, ids=repr)
@pytest.mark.parametrize("threads", [1, 2, 3])
@pytest.mark.parametrize(
    "schedule",
    [
        "schedule(static)",
        "schedule(static, 1)",
        "schedule(static, 4)",
        "schedule(dynamic)",
        "schedule(dynamic, 3)",
        "schedule(guided)",
        "schedule(guided, 2)",
        "schedule(auto)",
        "schedule(monotonic: dynamic, 2)",
        "schedule(nonmonotonic: dynamic, 2)",
        "schedule(nonmonotonic: guided)",
    ],
)
def test_every_iteration_runs_once(schedule, threads, space):
    assert [i for i, _ in run(schedule, space, threads)] == list(space)


# ---- static ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("space", SPACES, ids=repr)
@pytest.mark.parametrize("chunk", [1, 2, 3, 7])
@pytest.mark.parametrize("threads", [2, 3])
def test_static_with_chunk_is_round_robin(chunk, threads, space):
    """Chunk number j (of `chunk` iterations) is executed by thread j % threads."""
    result = run(f"schedule(static, {chunk})", space, threads)
    assert [thread for _, thread in result] == [(k // chunk) % threads for k in range(len(space))]


@pytest.mark.parametrize("space", SPACES, ids=repr)
@pytest.mark.parametrize("threads", [1, 2, 3, 4])
def test_static_without_chunk_gives_each_thread_one_block(threads, space):
    """The iterations are divided into chunks of about the same size, at most one for each thread."""
    owners = [thread for _, thread in run("schedule(static)", space, threads)]
    blocks = [owner for k, owner in enumerate(owners) if k == 0 or owner != owners[k - 1]]
    assert len(blocks) == len(set(blocks)), "a thread got more than one block"
    for thread in set(owners):
        assert owners.count(thread) <= ceil(len(space) / threads)


def test_default_schedule_is_correct_when_not_given():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for num_threads(3)"):
                for i in range(20):
                    out.append(i)
        """,
    )(out)
    assert sorted(out) == list(range(20))


# ---- dynamic -----------------------------------------------------------------------------------------------------


def test_chunks_are_given_on_demand():
    """A thread that is blocked does not hold back the iterations that it would get with a static schedule.

    The thread that gets iteration 0 waits until the rest of the iterations have been executed by the other
    thread, so with a static schedule (which gives half of them to the blocked one) it would wait forever.
    (A guided schedule starts with a big chunk, so the blocked thread would hold many iterations.)
    """
    finished = threading.Event()
    done = []

    def run_body(i):
        if i == 0:
            return finished.wait(timeout=10)
        done.append(i)
        if len(done) == 15:
            finished.set()
        return True

    results = []
    f = omp_function(
        f"""
        def f(body, results):
            with omp("parallel for schedule(dynamic, 1) num_threads(2)"):
                for i in range(16):
                    results.append(body(i))
        """,
    )
    f(run_body, results)
    assert all(results), "the iterations were not given to the free thread"


# ---- monotonic ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["dynamic, 2", "guided", "static, 3", "static"])
def test_monotonic_modifier_keeps_the_order_of_each_thread(kind):
    result = run(f"schedule(monotonic: {kind})", range(60), 3)
    for thread in range(3):
        mine = [i for i, owner in result if owner == thread]
        assert mine == sorted(mine)


# ---- the chunk size and the kind ---------------------------------------------------------------------------------


def test_chunk_size_can_be_an_expression():
    f = omp_function(
        """
        def f(out, size):
            with omp("parallel for schedule(static, size * 2) num_threads(2)"):
                for i in range(12):
                    out.append((i, omp_get_thread_num()))
        """,
    )
    out = []
    f(out, 3)
    assert [t for _, t in sorted(out)] == [0] * 6 + [1] * 6


def test_chunk_size_must_be_an_integer():
    f = omp_function(
        """
        def f(size):
            with omp("parallel for schedule(static, size) num_threads(2)"):
                for i in range(4):
                    pass
        """,
    )
    with raises_in_the_team(TypeError):
        f("1")


def test_schedule_kind_must_be_written_in_the_directive():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f(kind):
                with omp("parallel for schedule(kind, 1)"):
                    for i in range(4):
                        pass
            """,
        )


def test_auto_schedule_does_not_take_a_chunk():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for schedule(auto, 1)"):
                    for i in range(4):
                        pass
            """,
        )


# ---- runtime -----------------------------------------------------------------------------------------------------


def test_runtime_schedule_follows_omp_set_schedule():
    omp_set_schedule(omp_sched_static, 2)
    result = run("schedule(runtime)", range(14), 3)
    assert [t for _, t in result] == [(k // 2) % 3 for k in range(14)]

    omp_set_schedule(omp_sched_dynamic, 2)
    assert [i for i, _ in run("schedule(runtime)", range(14), 3)] == list(range(14))

    omp_set_schedule(omp_sched_guided, 1)
    assert [i for i, _ in run("schedule(runtime)", range(14), 3)] == list(range(14))
