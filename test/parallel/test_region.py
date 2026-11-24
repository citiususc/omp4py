"""The `parallel` construct: a team of threads executes the structured block, and the region ends with a barrier.

Covers the size of the team (`num_threads`, `omp_set_num_threads`, `if`), what the threads see, how exceptions
behave, and the code that cannot appear in the block.
"""

import pytest

from omp4py import omp_set_num_threads
from test.helpers import omp_function, raises_in_the_team

IDS = """
def f(out, n):
    with omp("parallel num_threads(n)"):
        out.append((omp_get_thread_num(), omp_get_num_threads()))
"""


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_num_threads_is_the_size_of_the_team(n):
    out = []
    omp_function(IDS)(out, n)
    assert sorted(out) == [(t, n) for t in range(n)]


def test_every_thread_executes_the_block_once():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel num_threads(4)"):
                out.append("hello")
        """,
    )(out)
    assert out == ["hello"] * 4


def test_the_region_ends_with_a_barrier():
    """When the region ends every thread has finished the block."""
    f = omp_function(
        """
        def f(out):
            with omp("parallel num_threads(4)"):
                out.append(omp_get_thread_num())
            return len(out)
        """,
    )
    assert f([]) == 4


def test_outside_a_region_there_is_only_one_thread():
    f = omp_function(
        """
        def f():
            before = (omp_get_thread_num(), omp_get_num_threads(), omp_in_parallel())
            with omp("parallel num_threads(3)"):
                pass
            return before, (omp_get_thread_num(), omp_get_num_threads(), omp_in_parallel())
        """,
    )
    assert f() == ((0, 1, False), (0, 1, False))


def test_without_the_clause_the_size_comes_from_omp_set_num_threads():
    omp_set_num_threads(3)
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel"):
                out.append(omp_get_thread_num())
        """,
    )(out)
    assert sorted(out) == [0, 1, 2]


def test_num_threads_clause_takes_precedence_over_omp_set_num_threads():
    omp_set_num_threads(2)
    out = []
    omp_function(IDS)(out, 4)
    assert sorted(t for t, _ in out) == [0, 1, 2, 3]


def test_num_threads_is_an_expression():
    out = []
    omp_function(
        """
        def f(out, a, b):
            with omp("parallel num_threads(a * b + 1)"):
                out.append(omp_get_thread_num())
        """,
    )(out, 2, 1)
    assert sorted(out) == [0, 1, 2]


# ---- if ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "size"), [(True, 3), (False, 1)])
def test_if_clause_runs_the_region_with_one_thread_when_it_is_false(condition, size):
    out = []
    omp_function(
        """
        def f(out, condition):
            with omp("parallel if(condition) num_threads(3)"):
                out.append((omp_get_thread_num(), omp_get_num_threads(), omp_in_parallel()))
        """,
    )(out, condition)
    assert sorted(out) == [(t, size, size > 1) for t in range(size)]


def test_if_clause_is_an_expression():
    out = []
    omp_function(
        """
        def f(out, n):
            with omp("parallel if(n > 100) num_threads(3)"):
                out.append(omp_get_num_threads())
        """,
    )(out, 10)
    assert out == [1]


# ---- proc_bind ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["master", "primary", "close", "spread"])
def test_proc_bind_policies_are_accepted(policy):
    out = []
    omp_function(
        f"""
        def f(out):
            with omp("parallel proc_bind({policy}) num_threads(2)"):
                out.append(omp_get_thread_num())
        """,
    )(out)
    assert sorted(out) == [0, 1]


def test_proc_bind_with_an_unknown_policy():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel proc_bind(near)"):
                    pass
            """,
        )


# ---- exceptions and code that cannot appear in the block ----------------------------------------------------------


def test_an_exception_in_the_block_reaches_the_code_after_the_region():
    f = omp_function(
        """
        def f():
            with omp("parallel num_threads(2)"):
                x = 0 / 0
        """,
    )
    with raises_in_the_team(ArithmeticError):
        f()


def test_functions_defined_in_the_block_can_return():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(2)"):
                def g():
                    return 2

                async def h():
                    return None

                x = g()
            return x
        """,
    )
    assert f() == 2


@pytest.mark.parametrize(
    "statement",
    ["return x", "yield x", "yield from range(3)"],
    ids=["return", "yield", "yield from"],
)
def test_the_block_cannot_leave_the_function(statement):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                x = 2
                with omp("parallel num_threads(2)"):
                    {statement}
            """,
        )


@pytest.mark.parametrize("statement", ["break", "continue"])
def test_the_block_cannot_jump_to_an_enclosing_loop(statement):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                for i in range(3):
                    with omp("parallel num_threads(2)"):
                        {statement}
            """,
        )


def test_loops_inside_the_block_can_use_break_and_continue():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel num_threads(2)"):
                for i in range(5):
                    if i == 1:
                        continue
                    if i == 3:
                        break
                    out.append(i)
        """,
    )(out)
    assert sorted(out) == [0, 0, 2, 2]


def test_directives_must_be_written_in_the_string():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f(text):
                with omp(text):
                    pass
            """,
        )
