"""`critical` and `barrier`: mutual exclusion and synchronization of the threads of a team.

All the critical regions without a name exclude each other, the ones with the same name exclude each other, and
the ones with different names do not. A barrier makes every thread wait until all of them have reached it.
"""

import threading

import pytest

from test.helpers import omp_function

# ---- critical ----------------------------------------------------------------------------------------------------

INCREMENT = """
def f(counter):
    with omp("parallel num_threads(4)"):
        for i in range(200):
            with omp("$region"):
                value = counter[0]
                sleep(0)
                counter[0] = value + 1
"""


@pytest.mark.parametrize("region", ["critical", "critical(update)"])
def test_critical_regions_exclude_each_other(region):
    """The update of the counter is not atomic: it only gives the right result if the regions do not overlap."""
    f = omp_function("from time import sleep\n" + INCREMENT, region=region)
    counter = [0]
    f(counter)
    assert counter[0] == 800


def test_critical_regions_with_different_names_do_not_exclude_each_other():
    """Each of the two threads waits inside its region for the other one to be inside its own region."""
    barrier = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(barrier):
            with omp("parallel num_threads(2)"):
                if omp_get_thread_num() == 0:
                    with omp("critical(first)"):
                        barrier.wait()
                else:
                    with omp("critical(second)"):
                        barrier.wait()
        """,
    )
    f(barrier)


def test_a_critical_region_without_a_name_does_not_exclude_the_ones_with_a_name():
    barrier = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(barrier):
            with omp("parallel num_threads(2)"):
                if omp_get_thread_num() == 0:
                    with omp("critical"):
                        barrier.wait()
                else:
                    with omp("critical(named)"):
                        barrier.wait()
        """,
    )
    f(barrier)


def test_critical_regions_can_be_nested_if_they_have_different_names():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(2)"):
                for i in range(5):
                    with omp("critical"):
                        with omp("critical(named)"):
                            x += 1
            return x
        """,
    )
    assert f() == 10


def test_critical_needs_a_block():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel"):
                    omp("critical")
            """,
        )


# ---- barrier -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ['omp("barrier")', 'with omp("barrier"):\n                    pass'], ids=["call", "with"])
def test_no_thread_goes_on_until_all_have_reached_the_barrier(form):
    f = omp_function(
        f"""
        def f(out):
            with omp("parallel num_threads(4)"):
                out.append(("before", omp_get_thread_num()))
                {form}
                out.append(("after", omp_get_thread_num()))
        """,
    )
    out = []
    f(out)
    assert [phase for phase, _ in out] == ["before"] * 4 + ["after"] * 4


def test_barriers_can_be_repeated():
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(3)"):
                for step in range(5):
                    out.append(step)
                    omp("barrier")
        """,
    )
    out = []
    f(out)
    assert out == [step for step in range(5) for _ in range(3)]


def test_what_a_thread_wrote_before_the_barrier_is_seen_after_it():
    f = omp_function(
        """
        def f(out):
            data = [0] * 4
            with omp("parallel num_threads(4)"):
                data[omp_get_thread_num()] = omp_get_thread_num() + 1
                omp("barrier")
                out.append(sum(data))
        """,
    )
    out = []
    f(out)
    assert out == [10] * 4


def test_a_barrier_outside_a_parallel_region_does_nothing():
    f = omp_function(
        """
        def f():
            omp("barrier")
            return "done"
        """,
    )
    assert f() == "done"
