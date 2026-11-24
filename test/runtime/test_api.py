"""The runtime library routines of the standard that omp4py implements: the ones that query and set the internal
control variables (ICVs) of the thread team, the nesting levels and the loop schedule.

Most ICVs belong to the task that sets them: a thread that calls `omp_set_num_threads` changes what its own
parallel regions use, not what the other threads or the code that started the region see.
"""

import threading
import warnings

import pytest

from omp4py import (
    omp_get_active_level,
    omp_get_ancestor_thread_num,
    omp_get_dynamic,
    omp_get_level,
    omp_get_max_active_levels,
    omp_get_max_threads,
    omp_get_nested,
    omp_get_num_threads,
    omp_get_schedule,
    omp_get_team_size,
    omp_get_thread_limit,
    omp_get_thread_num,
    omp_in_parallel,
    omp_sched_dynamic,
    omp_sched_guided,
    omp_sched_monotonic,
    omp_sched_static,
    omp_set_dynamic,
    omp_set_max_active_levels,
    omp_set_nested,
    omp_set_num_threads,
    omp_set_schedule,
)
from test.helpers import omp_function


@pytest.fixture(autouse=True)
def restore_icvs():
    """Leave the ICVs of the calling thread and the device as they were."""
    threads, dynamic, levels, schedule = omp_get_max_threads(), omp_get_dynamic(), omp_get_max_active_levels(), omp_get_schedule()
    yield
    omp_set_num_threads(threads)
    omp_set_dynamic(dynamic)
    omp_set_max_active_levels(levels)
    omp_set_schedule(*schedule)


# ---- the team ----------------------------------------------------------------------------------------------------


def test_omp_set_num_threads_is_what_omp_get_max_threads_returns():
    omp_set_num_threads(5)
    assert omp_get_max_threads() == 5


def test_outside_of_a_region():
    assert (omp_get_thread_num(), omp_get_num_threads(), omp_in_parallel()) == (0, 1, False)


def test_omp_get_max_threads_inside_a_region_is_the_icv_not_the_team_size():
    omp_set_num_threads(4)
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(2)"):
                out.append((omp_get_num_threads(), omp_get_max_threads()))
        """,
    )
    out = []
    f(out)
    assert out == [(2, 4)] * 2


def test_the_thread_that_calls_omp_set_num_threads_is_the_only_one_that_changes():
    """The new value is the one of the thread that set it, so the code after the region does not see it."""
    omp_set_num_threads(2)
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                if omp_get_thread_num() == 1:
                    omp_set_num_threads(7)
                omp("barrier")
                out.append((omp_get_thread_num(), omp_get_max_threads()))
        """,
    )
    out = []
    f(out)
    assert sorted(out) == [(0, 2), (1, 7), (2, 2)]
    assert omp_get_max_threads() == 2


def test_a_thread_that_is_not_in_the_team_starts_with_the_initial_value_of_the_icvs():
    omp_set_num_threads(6)
    seen = []
    thread = threading.Thread(target=lambda: seen.append(omp_get_max_threads()))
    thread.start()
    thread.join()
    assert seen != [6]


def test_dynamic_adjustment_can_be_set_and_asked():
    omp_set_dynamic(True)
    assert omp_get_dynamic() is True
    omp_set_dynamic(False)
    assert omp_get_dynamic() is False


def test_dynamic_adjustment_belongs_to_the_task():
    omp_set_dynamic(False)
    seen = []
    thread = threading.Thread(target=lambda: (omp_set_dynamic(True), seen.append(omp_get_dynamic())))
    thread.start()
    thread.join()
    assert seen == [True]
    assert omp_get_dynamic() is False


def test_the_thread_limit_is_a_positive_number():
    assert omp_get_thread_limit() >= 1


# ---- nesting -----------------------------------------------------------------------------------------------------


def test_the_nesting_limit_can_be_set_and_asked():
    omp_set_max_active_levels(3)
    assert omp_get_max_active_levels() == 3


def test_a_negative_nesting_limit_is_ignored():
    omp_set_max_active_levels(2)
    omp_set_max_active_levels(-1)
    assert omp_get_max_active_levels() == 2


def test_the_deprecated_nested_routines_follow_the_nesting_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        omp_set_nested(True)
        assert omp_get_nested() is True
        assert omp_get_max_active_levels() > 1
        omp_set_nested(False)
        assert omp_get_nested() is False
        assert omp_get_max_active_levels() == 1


NESTED = """
def f(out):
    with omp("parallel num_threads(3)"):
        outer = omp_get_thread_num()
        with omp("parallel num_threads(2)"):
            out.append((outer, omp_get_thread_num(), omp_get_num_threads(), omp_get_level(), omp_get_active_level()))
"""


def test_nested_regions_are_active_when_the_limit_allows_it():
    omp_set_max_active_levels(2)
    out = []
    omp_function(NESTED)(out)
    assert sorted(out) == sorted((o, i, 2, 2, 2) for o in range(3) for i in range(2))


def test_nested_regions_have_one_thread_when_the_limit_is_reached():
    omp_set_max_active_levels(1)
    out = []
    omp_function(NESTED)(out)
    assert sorted(out) == [(o, 0, 1, 2, 1) for o in range(3)]


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_the_limit_counts_only_the_active_regions(limit):
    """A region that executes with one thread (`if(false)`) does not use one of the allowed levels."""
    omp_set_max_active_levels(limit)
    f = omp_function(
        """
        def f(out):
            with omp("parallel if(False)"):
                with omp("parallel num_threads(2)"):
                    out.append((omp_get_level(), omp_get_active_level(), omp_get_num_threads()))
        """,
    )
    out = []
    f(out)
    assert sorted(out) == [(2, 1, 2)] * 2


def test_levels_and_active_levels_outside_a_region():
    assert (omp_get_level(), omp_get_active_level()) == (0, 0)


def test_a_region_with_one_thread_is_a_level_but_not_an_active_level():
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(1)"):
                out.append((omp_get_level(), omp_get_active_level(), omp_in_parallel()))
        """,
    )
    out = []
    f(out)
    assert out == [(1, 0, False)]


def test_omp_get_ancestor_thread_num_and_omp_get_team_size():
    omp_set_max_active_levels(2)
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                outer = omp_get_thread_num()
                with omp("parallel num_threads(2)"):
                    inner = omp_get_thread_num()
                    out.append((
                        outer,
                        inner,
                        [omp_get_ancestor_thread_num(level) for level in (0, 1, 2)],
                        [omp_get_team_size(level) for level in (0, 1, 2)],
                    ))
        """,
    )
    out = []
    f(out)
    assert sorted(out) == sorted((o, i, [0, o, i], [1, 3, 2]) for o in range(3) for i in range(2))


@pytest.mark.parametrize("level", [-1, 1, 2])
def test_levels_that_do_not_exist(level):
    assert omp_get_ancestor_thread_num(level) == -1
    assert omp_get_team_size(level) == -1


def test_level_zero_is_the_initial_thread():
    assert omp_get_ancestor_thread_num(0) == 0
    assert omp_get_team_size(0) == 1


# ---- schedule ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [omp_sched_static, omp_sched_dynamic, omp_sched_guided])
@pytest.mark.parametrize("chunk", [1, 4, 100])
def test_omp_get_schedule_returns_what_omp_set_schedule_sets(kind, chunk):
    omp_set_schedule(kind, chunk)
    assert omp_get_schedule() == (kind, chunk)


def test_the_monotonic_modifier_is_kept():
    omp_set_schedule(omp_sched_dynamic | omp_sched_monotonic, 3)
    assert omp_get_schedule() == (omp_sched_dynamic | omp_sched_monotonic, 3)


def test_the_schedule_belongs_to_the_task():
    """A thread that sets the schedule does not change the one of the other threads."""
    omp_set_schedule(omp_sched_static, 2)
    seen = []

    def other():
        before = omp_get_schedule()
        omp_set_schedule(omp_sched_dynamic, 9)
        seen.append((before, omp_get_schedule()))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert omp_get_schedule() == (omp_sched_static, 2)
    assert seen[0][0] != (omp_sched_static, 2)
    assert seen[0][1] == (omp_sched_dynamic, 9)


def test_the_schedule_set_by_one_thread_of_a_team_is_not_seen_by_the_others():
    omp_set_schedule(omp_sched_static, 2)
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                if omp_get_thread_num() == 1:
                    omp_set_schedule(omp_sched_dynamic, 5)
                omp("barrier")
                out.append((omp_get_thread_num(), omp_get_schedule()))
        """,
    )
    out = []
    f(out)
    assert sorted(out) == [(0, (omp_sched_static, 2)), (1, (omp_sched_dynamic, 5)), (2, (omp_sched_static, 2))]
