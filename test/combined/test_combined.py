"""Combined directives: `parallel for`, `parallel sections`, ... A combined directive means the same as the
directives written one inside the other, from the outermost to the innermost, and its clauses go to the
constructs that accept them. The order is the one of the nesting, so `for parallel` is not a directive.

The names of the OpenMP 6.0 combined directives (`parallel masked`, `masked parallel`) are accepted too.
"""

import pytest

from omp4py import omp_set_max_active_levels
from test.helpers import omp_function

# ---- parallel for ------------------------------------------------------------------------------------------------

LOOP_COMBINED = """
def f(out):
    with omp("parallel for $clauses"):
        for i in range(24):
            out.append((i, omp_get_thread_num(), omp_get_num_threads()))
"""

LOOP_SEPARATE = """
def f(out):
    with omp("parallel $parallel_clauses"):
        with omp("for $for_clauses"):
            for i in range(24):
                out.append((i, omp_get_thread_num(), omp_get_num_threads()))
"""

CLAUSES = [
    ("num_threads(3)", "num_threads(3)", ""),
    ("num_threads(4) schedule(static, 1)", "num_threads(4)", "schedule(static, 1)"),
    ("num_threads(3) schedule(static, 5)", "num_threads(3)", "schedule(static, 5)"),
    ("num_threads(2) schedule(static)", "num_threads(2)", "schedule(static)"),
    ("num_threads(1)", "num_threads(1)", ""),
    ("if(False) num_threads(3)", "if(False) num_threads(3)", ""),
]


@pytest.mark.parametrize(("combined", "parallel", "worksharing"), CLAUSES)
def test_parallel_for_is_a_parallel_region_with_a_for_inside(combined, parallel, worksharing):
    """Both forms distribute the iterations in the same way. (The static schedules are the only ones that are fixed.)"""
    one, two = [], []
    omp_function(LOOP_COMBINED, clauses=combined)(one)
    omp_function(LOOP_SEPARATE, parallel_clauses=parallel, for_clauses=worksharing)(two)
    assert sorted(one) == sorted(two)


def test_the_clauses_go_to_the_construct_that_accepts_them():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for num_threads(3) schedule(static, 2) if(True)"):
                for i in range(12):
                    out.append((i, omp_get_thread_num(), omp_get_num_threads()))
        """,
    )(out)
    assert sorted(out) == [(i, (i // 2) % 3, 3) for i in range(12)]


def test_data_clauses_of_parallel_for():
    f = omp_function(
        """
        def f():
            total = 0
            seen = 100
            last = -1
            scratch = 0
            with omp("parallel for reduction(+:total) firstprivate(seen) lastprivate(last) private(scratch) num_threads(3)"):
                for i in range(10):
                    scratch = i
                    seen += 1
                    last = scratch
                    total += scratch
            return total, seen, last, scratch
        """,
    )
    assert f() == (45, 100, 9, 0)


def test_collapse_and_ordered_of_parallel_for():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for collapse(2) ordered schedule(dynamic) num_threads(3)"):
                for i in range(3):
                    for j in range(4):
                        with omp("ordered"):
                            out.append((i, j))
        """,
    )(out)
    assert out == [(i, j) for i in range(3) for j in range(4)]


def test_the_if_clause_with_the_name_of_the_directive():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for if(parallel: False) num_threads(3)"):
                for i in range(6):
                    out.append(omp_get_num_threads())
        """,
    )(out)
    assert out == [1] * 6


# ---- parallel sections -------------------------------------------------------------------------------------------


def test_parallel_sections():
    out = []
    omp_function(
        """
        def f(out):
            x = 0
            y = 0
            with omp("parallel sections num_threads(3) reduction(+:x) lastprivate(y)"):
                with omp("section"):
                    x += 1
                    y = 1
                    out.append(("a", omp_get_num_threads()))
                with omp("section"):
                    x += 2
                    y = 2
                    out.append(("b", omp_get_num_threads()))
            out.append((x, y))
        """,
    )(out)
    assert out[-1] == (3, 2)
    assert sorted(out[:-1]) == [("a", 3), ("b", 3)]


# ---- OpenMP 6.0 names --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("directive", ["parallel single", "parallel masked"])
def test_parallel_with_a_block_for_one_thread(directive):
    out = []
    omp_function(
        f"""
        def f(out):
            with omp("{directive} num_threads(4)"):
                out.append(omp_get_num_threads())
        """,
    )(out)
    assert out == [4]


def test_masked_parallel_opens_a_nested_region_from_one_thread_only():
    omp_set_max_active_levels(2)
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel num_threads(4)"):
                with omp("masked parallel num_threads(2)"):
                    out.append((omp_get_ancestor_thread_num(1), omp_get_thread_num(), omp_get_team_size(2)))
        """,
    )(out)
    assert sorted(out) == [(0, 0, 2), (0, 1, 2)]


# ---- what is not a combined directive ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "directive",
    ["for parallel", "sections parallel", "parallel parallel", "parallel master", "parallel critical", "parallel barrier"],
)
def test_directives_that_do_not_exist(directive):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                with omp("{directive}"):
                    for i in range(3):
                        pass
            """,
        )


@pytest.mark.parametrize("clause", ["schedule(static)", "ordered", "collapse(2)"])
def test_a_clause_that_the_directive_does_not_have(clause):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                with omp("parallel sections {clause}"):
                    with omp("section"):
                        pass
            """,
        )
