"""The Python code inside the clauses: `if(...)`, `num_threads(...)`, `schedule(static, ...)`...

The grammar of the clauses does not parse that code: it only has to find where it ends, which is before the `)` that
closes the clause, and it hands it over to `ast.parse`. For that, the brackets of the code (`()`, `[]` and `{}`) are
tokens of the grammar and they have to be balanced, but they can be empty (`f()`) and they can follow anything: a
name, a number, a string, another bracket.
"""

import pytest

from test.helpers import omp_function

# ---- the code of the clauses -------------------------------------------------------------------------------------

CODE = {
    # calls and brackets that are empty
    "call without arguments": "int() + 2",
    "call of a name": "f()",
    "method without arguments": "(2).bit_length()",
    "chained calls": "f()().bit_length()",
    "empty list": "len([]) + 2",
    "empty dictionary": "len({}) + 2",
    "empty tuple": "len(()) + 2",
    "lambda without arguments": "(lambda: 2)()",
    # brackets that follow a string, a bracket or a space
    "subscript with a string": "d['a']",
    "subscript with double quotes": 'd["a"]',
    "subscript with blanks": "d[ 'a' ]",
    "subscript with a string, then more": "d['a'] + 0",
    "subscripts in a row": "d['a'][0] + d['b'][1] - 1",
    "subscript of a subscript": "d[e['a']]",
    "subscript of a call": "d[g(1)]",
    "subscript of a tuple": "d[(0, 1)]",
    "subscript with a slice": "len(d['c'][0:2]) + 0",
    "subscript with a slice of strings": "len(w['a':'z']) + 0",
    "list with a string": "len(['a', 'b'])",
    "list that is indexed": "len(['a', 'b'][0]) + 1",
    "dictionary with strings": "{'a': 2}['a']",
    "dictionary that is called": "{'a': 2}.get('a', 1)",
    "set": "len({'a', 'b'})",
    "comprehensions": "len([c for c in 'ab']) + len({c: 1 for c in 'ab'})",
    "brackets in a string": "len('[(])') - 2",
    "brackets and a string with brackets": "d['[a]'] + len(['}'])",
}


@pytest.mark.parametrize("name", CODE)
def test_the_python_code_of_a_clause_is_found(name):
    """The code is not run: the directive is processed, and that fails if the grammar does not find the code."""
    omp_function(f'def f(d, e, w, f, g):\n    with omp("""parallel num_threads({CODE[name]})"""):\n        pass\n')


@pytest.mark.parametrize(
    "code",
    ["1 ]", "] + 1", "d['a'}]", "d['a']]", "d[[1]", "f(", "f)(", "{ 1", "( 1 ]"],
)
def test_unbalanced_brackets_are_rejected(code):
    with pytest.raises(SyntaxError):
        omp_function(f'def f(d):\n    with omp("parallel num_threads({code})"):\n        pass\n')


def test_the_code_that_has_a_call_without_arguments_runs():
    out = []
    omp_function(
        """
        def f(out):
            n = lambda: 2
            with omp("parallel num_threads(n())"):
                out.append(omp_get_thread_num())
        """,
    )(out)
    assert sorted(out) == [0, 1]


def test_the_code_that_has_a_subscript_with_a_string_runs():
    out = []
    omp_function(
        """
        def f(out):
            d = {"threads": 2}
            with omp("parallel num_threads(d['threads'])"):
                out.append(omp_get_thread_num())
        """,
    )(out)
    assert sorted(out) == [0, 1]


@pytest.mark.parametrize("clause", ["private(x)", "firstprivate(x)", "shared(x)", "lastprivate(x)"])
def test_the_brackets_of_the_lists_of_names_still_work(clause):
    """`x[1:2]`, a section, is not Python code: it is part of the grammar of the lists of names (`py_section`)."""
    directive = f"parallel for {clause} num_threads(2)"
    omp_function(f'def f():\n    x = 0\n    with omp("{directive}"):\n        for i in range(2):\n            pass\n')
