"""The `reduction` clause: every thread works on a private copy that starts with the identity of the operator, and
the copies are combined with the original value when the construct ends. The result is the one of the sequential
code, whatever the number of threads.

The operators are `+ - * & | ^ and or max min`, and the ones that `declare reduction` defines.
"""

import pytest

from test.helpers import omp_function

THREADS = [1, 2, 3, 7]

# operator, initial value, statement that updates x with the iteration variable i, iteration space
OPERATORS = [
    ("+", "5", "x += i", range(1, 20)),
    ("-", "100", "x -= i", range(1, 20)),
    ("*", "2", "x *= i", range(1, 12)),
    ("&", "0b1111111111", "x &= ~(1 << (i % 9))", range(1, 20)),
    ("|", "0", "x |= 1 << (i % 9)", range(1, 20)),
    ("^", "0b1010", "x ^= i", range(1, 20)),
    ("and", "True", "x = x and i != 13", range(1, 20)),
    ("or", "False", "x = x or i == 13", range(1, 20)),
    ("max", "3", "x = max(x, i * 7 % 11)", range(1, 30)),
    ("min", "50", "x = min(x, 100 - i * 7 % 11)", range(1, 30)),
]


def sequential(initial: str, statement: str, space: range):
    scope = {}
    exec(f"x = {initial}\nfor i in {space!r}:\n    {statement}", scope)  # noqa: S102
    return scope["x"]


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize(("operator", "initial", "statement", "space"), OPERATORS, ids=[o[0] for o in OPERATORS])
def test_the_result_is_the_one_of_the_sequential_loop(operator, initial, statement, space, threads):
    f = omp_function(
        f"""
        def f():
            x = {initial}
            with omp("parallel for reduction({operator}:x) num_threads({threads})"):
                for i in {space!r}:
                    {statement}
            return x
        """,
    )
    assert f() == sequential(initial, statement, space)


@pytest.mark.parametrize("threads", THREADS)
def test_float_sum(threads):
    f = omp_function(
        f"""
        def f(n):
            x = 0.5
            with omp("parallel for reduction(+:x) num_threads({threads})"):
                for i in range(n):
                    x += i * 0.25
            return x
        """,
    )
    assert f(100) == pytest.approx(0.5 + sum(i * 0.25 for i in range(100)))


@pytest.mark.parametrize("threads", THREADS)
def test_complex_sum(threads):
    f = omp_function(
        f"""
        def f(n):
            x = 1j
            with omp("parallel for reduction(+:x) num_threads({threads})"):
                for i in range(n):
                    x += complex(i, 1)
            return x
        """,
    )
    assert f(20) == 1j + sum(complex(i, 1) for i in range(20))


# ---- where the clause is used ------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", THREADS)
def test_in_a_parallel_region(threads):
    """Every thread adds its contribution to the original value."""
    f = omp_function(
        f"""
        def f():
            x = 10
            with omp("parallel reduction(+:x) num_threads({threads})"):
                x += 1
            return x
        """,
    )
    assert f() == 10 + threads


def test_the_private_copy_starts_with_the_identity_and_not_with_the_original_value():
    f = omp_function(
        """
        def f(out):
            x = 100
            with omp("parallel reduction(+:x) num_threads(3)"):
                out.append(x)
            return x
        """,
    )
    out = []
    assert f(out) == 100
    assert out == [0, 0, 0]


def test_in_sections():
    f = omp_function(
        """
        def f():
            x = 1
            with omp("parallel sections reduction(+:x) num_threads(2)"):
                with omp("section"):
                    x += 3
                with omp("section"):
                    x += 2
            return x
        """,
    )
    assert f() == 6


def test_in_a_single():
    f = omp_function(
        """
        def f():
            x = 1
            with omp("parallel num_threads(3)"):
                with omp("single"):
                    x += 5
            return x
        """,
    )
    assert f() == 6


def test_several_variables_with_the_same_operator():
    f = omp_function(
        """
        def f():
            a = 0
            b = 100
            with omp("parallel for reduction(+:a, b) num_threads(3)"):
                for i in range(10):
                    a += i
                    b += 2 * i
            return a, b
        """,
    )
    assert f() == (45, 100 + 90)


def test_several_reduction_clauses_with_different_operators():
    f = omp_function(
        """
        def f():
            total = 0
            product = 1
            with omp("parallel for reduction(+:total) reduction(*:product) num_threads(3)"):
                for i in range(1, 8):
                    total += i
                    product *= i
            return total, product
        """,
    )
    assert f() == (28, 5040)


def test_a_reduction_variable_can_be_read_outside_the_loop_after_it():
    f = omp_function(
        """
        def f():
            x = 0
            with omp("parallel num_threads(3)"):
                with omp("for reduction(+:x)"):
                    for i in range(10):
                        x += i
                with omp("single"):
                    x += 100
            return x
        """,
    )
    assert f() == 145


# ---- invalid clauses ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clause",
    ["reduction(+:x, *:y)", "reduction(x)", "reduction(sum:x)", "reduction(+:x, x)", "reduction(+:x) reduction(*:x)"],
)
def test_invalid_reduction_clauses(clause):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                x = 0
                y = 1
                with omp("parallel {clause}"):
                    x = 1
            """,
        )


def test_a_variable_must_exist():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel reduction(+:y)"):
                    pass
            """,
        )


# ---- declare reduction -------------------------------------------------------------------------------------------


def test_declare_reduction_with_the_combiner_inside_the_parentheses():
    f = omp_function(
        """
        def f(n):
            omp("declare reduction(merge : list : omp_out += omp_in) initializer(omp_priv = [])")
            items: list = []
            with omp("parallel for reduction(merge:items) num_threads(4)"):
                for i in range(n):
                    items.append(i)
            return sorted(items)
        """,
    )
    assert f(20) == list(range(20))


def test_declare_reduction_with_the_combiner_as_a_clause():
    f = omp_function(
        """
        def f(words):
            omp("declare reduction(longest : object) "
                "combiner(omp_out = omp_in if len(omp_in) > len(omp_out) else omp_out) "
                "initializer(omp_priv = '')")
            best = ""
            with omp("parallel for reduction(longest:best) num_threads(3)"):
                for i in range(len(words)):
                    if len(words[i]) > len(best):
                        best = words[i]
            return best
        """,
    )
    assert f(["a", "abc", "ab", "abcd", "x"]) == "abcd"


def test_declare_reduction_with_a_tuple_to_keep_the_position_of_the_maximum():
    f = omp_function(
        """
        def f(values):
            omp("declare reduction(best : tuple) "
                "combiner(omp_out = omp_in if omp_in[0] > omp_out[0] else omp_out) "
                "initializer(omp_priv = (float('-inf'), -1))")
            top: tuple = (float("-inf"), -1)
            with omp("parallel for reduction(best:top) num_threads(4)"):
                for i in range(len(values)):
                    if values[i] > top[0]:
                        top = (values[i], i)
            return top
        """,
    )
    assert f([3, 9, 2, 7, 11, 1, 5]) == (11, 4)


def test_without_an_initializer_the_copy_is_an_empty_value_of_the_same_type():
    f = omp_function(
        """
        def f():
            omp("declare reduction(merge : list : omp_out += omp_in)")
            items: list = [0]
            with omp("parallel reduction(merge:items) num_threads(3)"):
                items.append(1)
            return sorted(items)
        """,
    )
    assert f() == [0, 1, 1, 1]


def test_a_declaration_is_used_by_the_functions_that_follow_it():
    f = omp_function(
        """
        @omp
        def declare():
            omp("declare reduction(merge : list : omp_out += omp_in) initializer(omp_priv = [])")

        def f():
            items: list = []
            with omp("parallel reduction(merge:items) num_threads(2)"):
                items.append(1)
            return items
        """,
    )
    assert f() == [1, 1]


def test_the_same_reduction_cannot_be_declared_twice():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                omp("declare reduction(merge : list : omp_out += omp_in)")
                omp("declare reduction(merge : list : omp_out += omp_in)")
            """,
        )


def test_a_reduction_that_was_not_declared():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                items = []
                with omp("parallel reduction(merge:items)"):
                    items.append(1)
            """,
        )
