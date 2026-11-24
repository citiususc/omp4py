"""Data-sharing attributes of a parallel region: `shared`, `private`, `firstprivate` and `default`.

A shared variable is the same for every thread, a private one is a different variable in each thread, and a
firstprivate one is a private variable that starts with the value that the original had. The private copies do not
exist after the region. `default` gives the attribute to the variables that no clause names.
"""

import pytest

from test.helpers import omp_function

# ---- shared ------------------------------------------------------------------------------------------------------


def test_variables_are_shared_by_default():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(3)"):
                x = 1
            return x
        """,
    )
    assert f() == 1


def test_shared_clause():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel shared(x) num_threads(3)"):
                x = 1
            return x
        """,
    )
    assert f() == 1


def test_shared_objects_are_the_same_for_every_thread():
    f = omp_function(
        """
        def f(items):
            with omp("parallel num_threads(4)"):
                items.append(omp_get_thread_num())
        """,
    )
    items = []
    f(items)
    assert sorted(items) == [0, 1, 2, 3]


# ---- private -----------------------------------------------------------------------------------------------------


def test_private_copies_do_not_exist_after_the_region():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel private(x) num_threads(3)"):
                x = 1
            return x
        """,
    )
    assert f() == 0


def test_every_thread_has_its_own_private_variable():
    """Each thread writes its number in its copy, waits for the others and finds that its copy was not changed."""
    f = omp_function(
        """
        def f(out):
            x = -1
            with omp("parallel private(x) num_threads(4)"):
                x = omp_get_thread_num()
                omp("barrier")
                out.append(x == omp_get_thread_num())
        """,
    )
    out = []
    f(out)
    assert out == [True] * 4


# ---- firstprivate ------------------------------------------------------------------------------------------------


def test_firstprivate_starts_with_the_value_of_the_original():
    f = omp_function(
        """
        def f(out):
            x = 2
            with omp("parallel firstprivate(x) num_threads(3)"):
                out.append(x)
        """,
    )
    out = []
    f(out)
    assert out == [2, 2, 2]


def test_firstprivate_changes_do_not_reach_the_original_or_the_other_threads():
    f = omp_function(
        """
        def f(out):
            x = 10
            with omp("parallel firstprivate(x) num_threads(4)"):
                x += omp_get_thread_num()
                omp("barrier")
                out.append(x - 10 == omp_get_thread_num())
            return x
        """,
    )
    out = []
    assert f(out) == 10
    assert out == [True] * 4


# ---- default -----------------------------------------------------------------------------------------------------


def test_default_shared():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel default(shared) shared(x) num_threads(2)"):
                x = 1
            return x
        """,
    )
    assert f() == 1


def test_default_none_needs_every_variable_in_a_clause():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel default(none) shared(x) num_threads(2)"):
                x = 1
            return x
        """,
    )
    assert f() == 1


def test_default_none_with_a_variable_that_is_not_in_a_clause():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                x = 0
                with omp("parallel default(none)"):
                    x = 1
            """,
        )


def test_default_none_allows_variables_that_only_exist_in_the_block():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel default(none) num_threads(2)"):
                y = 1
            return x
        """,
    )
    assert f() == 0


@pytest.mark.parametrize(
    "directive",
    [
        "parallel for default(none) lastprivate(x) num_threads(2)",
        "parallel for default(none) shared(x) num_threads(1)",
    ],
)
def test_default_none_in_a_combined_loop(directive):
    f = omp_function(
        f"""
        def f():
            x = 0
            with omp("{directive}"):
                for i in range(10):
                    x = 1
            return x
        """,
    )
    assert f() == 1


def test_default_none_in_a_combined_sections():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel sections default(none) lastprivate(x) num_threads(2)"):
                with omp("section"):
                    x = 1
            return x
        """,
    )
    assert f() == 1


def test_default_firstprivate():
    f = omp_function(
        """
        def f(out):
            x = 5
            y = 7
            with omp("parallel default(firstprivate) shared(out) num_threads(3)"):
                x += 1
                out.append((x, y))
            return x, y
        """,
    )
    out = []
    assert f(out) == (5, 7)
    assert out == [(6, 7)] * 3


def test_default_firstprivate_with_a_clause_that_names_a_variable():
    """The clauses that name a variable win over `default`: here `total` is reduced, not firstprivate."""
    f = omp_function(
        """
        def f():
            total = 0
            step = 2
            with omp("parallel reduction(+:total) default(firstprivate) num_threads(3)"):
                total += step
            return total
        """,
    )
    assert f() == 6


def test_default_firstprivate_in_a_combined_loop_with_a_reduction():
    f = omp_function(
        """
        def f(n):
            total = 0
            width = 0.5
            with omp("parallel for reduction(+:total) default(firstprivate) num_threads(3)"):
                for i in range(n):
                    total += i * width
            return total
        """,
    )
    assert f(10) == sum(range(10)) * 0.5


def test_default_private():
    f = omp_function(
        """
        def f():
            x = 5
            with omp("parallel default(private) num_threads(3)"):
                x = 1
            return x
        """,
    )
    assert f() == 5


# ---- invalid clauses ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clauses",
    ["private(x) shared(x)", "shared(x) shared(x)", "private(x) firstprivate(x)", "private(x, x)"],
)
def test_a_variable_cannot_appear_twice_in_data_clauses(clauses):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                x = 0
                with omp("parallel {clauses}"):
                    x = 1
            """,
        )


@pytest.mark.parametrize("clause", ["shared", "private", "firstprivate"])
def test_a_variable_in_a_clause_must_exist(clause):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                x = 0
                with omp("parallel {clause}(y)"):
                    x = 1
            """,
        )
