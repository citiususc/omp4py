"""The clauses of the `for` construct that are not about the schedule: `nowait`, `private`, `firstprivate`,
`lastprivate` and `reduction`, and the loops that the construct does not accept."""

import threading

import pytest

from test.helpers import omp_function

# ---- nowait ------------------------------------------------------------------------------------------------------


def test_nowait_removes_the_barrier_at_the_end_of_the_loop():
    """The thread that does not run iteration 0 leaves the loop while thread 0 waits at the barrier.

    Without `nowait` the loop would end with a barrier that thread 0 never reaches, and the barrier of the test
    would break by timeout.
    """
    barrier = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(out, barrier):
            with omp("parallel num_threads(2)"):
                with omp("for nowait"):
                    for i in range(10):
                        out.append(i)
                        if i == 0:
                            barrier.wait()
                if omp_get_thread_num() != 0:
                    barrier.wait()
        """,
    )
    out = []
    f(out, barrier)
    assert sorted(out) == list(range(10))


def test_the_loop_ends_with_a_barrier():
    """Without `nowait`, no thread leaves the loop before every iteration is done."""
    f = omp_function(
        """
        def f(done, seen):
            with omp("parallel num_threads(3)"):
                with omp("for"):
                    for i in range(30):
                        done.append(i)
                seen.append(len(done))
        """,
    )
    done, seen = [], []
    f(done, seen)
    assert seen == [30, 30, 30]


def test_nowait_is_not_accepted_in_a_combined_directive():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for nowait"):
                    for i in range(4):
                        pass
            """,
        )


def test_nowait_is_accepted_when_the_loop_is_the_only_statement_of_a_parallel_region():
    """A `parallel` and a `for` in different directives are not a combined construct, even if one has only the other."""
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel num_threads(2)"):
                with omp("for nowait"):
                    for i in range(4):
                        out.append(i)
        """,
    )(out)
    assert sorted(out) == [0, 1, 2, 3]


def test_nowait_is_accepted_in_a_loop_nested_in_several_directives():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel num_threads(2)"):
                with omp("parallel num_threads(1)"):
                    with omp("for nowait"):
                        for i in range(2):
                            out.append(i)
        """,
    )(out)
    assert sorted(out) == [0, 0, 1, 1]


def test_nowait_is_not_accepted_when_the_parallel_is_in_the_same_directive_at_any_depth():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel num_threads(2)"):
                    with omp("parallel for nowait num_threads(1)"):
                        for i in range(2):
                            pass
            """,
        )


# ---- private, firstprivate ---------------------------------------------------------------------------------------


def test_private_copies_are_not_visible_after_the_loop():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(2)"):
                x = 1
                with omp("for private(x)"):
                    for i in range(10):
                        x = i
            return x
        """,
    )
    assert f() == 1


def test_private_in_the_loop_does_not_change_the_value_of_the_enclosing_construct():
    f = omp_function(
        """
        def f():
            x = 0
            y = 0
            with omp("parallel private(x) num_threads(2)"):
                x = 1
                with omp("for private(x)"):
                    for i in range(10):
                        x = i
                y = x
            return x, y
        """,
    )
    assert f() == (0, 1)


def test_firstprivate_starts_with_the_value_before_the_loop():
    f = omp_function(
        """
        def f(out):
            x = 100
            with omp("parallel for firstprivate(x) schedule(static, 1) num_threads(3)"):
                for i in range(6):
                    x += 1
                    out.append(x)
            return x
        """,
    )
    out = []
    assert f(out) == 100
    assert min(out) == 101
    assert sorted(out)[:3] == [101, 101, 101]


def test_firstprivate_in_the_loop_and_in_the_region():
    f = omp_function(
        """
        def f():
            x = 1
            y = 0
            with omp("parallel firstprivate(x) num_threads(2)"):
                x += 1
                with omp("for firstprivate(x)"):
                    for i in range(10):
                        x += i
                y = x
            return x, y
        """,
    )
    assert f() == (1, 2)


# ---- lastprivate -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["schedule(static)", "schedule(static, 1)", "schedule(dynamic, 3)", "schedule(guided)"])
@pytest.mark.parametrize("space", [range(10), range(3, 40, 7), range(20, -1, -3), range(1)], ids=repr)
def test_lastprivate_takes_the_value_of_the_last_iteration(schedule, space):
    f = omp_function(
        f"""
        def f():
            x = None
            with omp("parallel for lastprivate(x) {schedule} num_threads(3)"):
                for i in {space!r}:
                    x = i
            return x
        """,
    )
    assert f() == space[-1]


@pytest.mark.parametrize("schedule", ["schedule(static)", "schedule(dynamic, 3)"])
def test_lastprivate_with_collapse(schedule):
    f = omp_function(
        f"""
        def f():
            x = None
            with omp("parallel for lastprivate(x) collapse(2) {schedule} num_threads(3)"):
                for i in range(10):
                    for j in range(10):
                        x = (i, j)
            return x
        """,
    )
    assert f() == (9, 9)


def test_lastprivate_of_a_variable_that_the_loop_does_not_assign():
    """There is nothing to copy back, and the code that is generated for the end of the loop has to be valid."""
    f = omp_function(
        """
        def f():
            x = 7
            with omp("parallel for lastprivate(x) num_threads(2)"):
                for i in range(4):
                    pass
            return x
        """,
    )
    assert f() == 7


# ---- reduction ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 3, 7])
def test_reduction_in_a_for_directive(threads):
    f = omp_function(
        """
        def f():
            x = 5
            with omp("parallel num_threads($threads)"):
                with omp("for reduction(+:x)"):
                    for i in range(10):
                        x += i
            return x
        """,
        threads=threads,
    )
    assert f() == 5 + sum(range(10))


# ---- invalid code ------------------------------------------------------------------------------------------------


def test_reduction_of_a_variable_that_is_private_outside():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                x = 0
                with omp("parallel private(x)"):
                    with omp("for reduction(+:x)"):
                        for i in range(10):
                            x += i
            """,
        )


def test_a_variable_cannot_appear_in_two_data_clauses():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                x = 0
                with omp("parallel"):
                    with omp("for private(x) reduction(+:x)"):
                        for i in range(10):
                            x += i
            """,
        )


def test_lastprivate_of_an_unknown_variable():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for lastprivate(y)"):
                    for i in range(10):
                        pass
            """,
        )


def test_the_directive_needs_a_loop():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel for"):
                    x = 0
            """,
        )


def test_the_loop_must_go_over_a_range():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f(items):
                with omp("parallel for"):
                    for i in items:
                        pass
            """,
        )


def test_the_loop_cannot_leave_the_block_with_break():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel"):
                    with omp("for"):
                        for i in range(10):
                            if i == 4:
                                break
            """,
        )


def test_break_of_an_inner_loop_is_allowed():
    f = omp_function(
        """
        def f(out):
            with omp("parallel for num_threads(2)"):
                for i in range(10):
                    for j in range(10):
                        if j == 2:
                            break
                        out.append((i, j))
        """,
    )
    out = []
    f(out)
    assert sorted(out) == [(i, j) for i in range(10) for j in range(2)]
