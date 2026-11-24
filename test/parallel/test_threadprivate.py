"""`threadprivate` variables: a global variable that has one copy for each thread, and that keeps its value from one
parallel region to the next. `copyin` starts the copies of a region with the value of the master thread."""

import pytest

from test.helpers import omp_function

def test_every_thread_has_its_own_copy():
    out = []
    omp_function(
        """
        x = 0

        def f(out):
            global x
            omp("threadprivate(x)")
            with omp("parallel num_threads(4)"):
                x = 10 + omp_get_thread_num()
                omp("barrier")
                out.append(x == 10 + omp_get_thread_num())
        """,
    )(out)
    assert out == [True] * 4


def test_the_copy_of_the_master_thread_is_the_one_seen_after_the_region():
    f = omp_function(
        """
        x = 0

        def f():
            global x
            omp("threadprivate(x)")
            with omp("parallel num_threads(2)"):
                if omp_get_thread_num() > 0:
                    x = 2
            return x
        """,
    )
    assert f() == 0


def test_the_copies_keep_their_values_between_regions():
    out = []
    omp_function(
        """
        x = 0

        def f(out):
            global x
            omp("threadprivate(x)")
            with omp("parallel num_threads(3)"):
                x = 100 + omp_get_thread_num()
            with omp("parallel num_threads(3)"):
                out.append(x == 100 + omp_get_thread_num())
        """,
    )(out)
    assert out == [True] * 3


def test_copyin_starts_the_copies_with_the_value_of_the_master():
    out = []
    omp_function(
        """
        x = 0

        def f(out):
            global x
            omp("threadprivate(x)")
            with omp("parallel num_threads(3)"):
                if omp_get_thread_num() == 0:
                    x = 2
            with omp("parallel num_threads(3) copyin(x)"):
                out.append(x)
        """,
    )(out)
    assert out == [2, 2, 2]


def test_without_copyin_the_other_threads_keep_their_own_value():
    out = []
    omp_function(
        """
        x = 0

        def f(out):
            global x
            omp("threadprivate(x)")
            with omp("parallel num_threads(3)"):
                if omp_get_thread_num() == 0:
                    x = 2
            with omp("parallel num_threads(3)"):
                out.append(x)
        """,
    )(out)
    assert sorted(out) == [0, 0, 2]


@pytest.mark.parametrize("clause", ["private(x)", "firstprivate(x)", "shared(x)", "reduction(+:x)"])
def test_a_threadprivate_variable_cannot_be_in_a_data_clause(clause):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            x = 0

            def f():
                global x
                omp("threadprivate(x)")
                with omp("parallel {clause}"):
                    pass
            """,
        )


def test_copyin_needs_a_threadprivate_variable():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            x = 0

            def f():
                global x
                with omp("parallel copyin(x)"):
                    pass
            """,
        )
