"""The directive is a string literal of Python, and it can be written in many ways: with any kind of quotes, as a
raw string, split in several literals over several lines, with comments between the parts...

Whatever the way, the directive that omp4py reads is the same, and when it is not valid the error points at the
exact place of the source file where the wrong token is.
"""

from pathlib import Path

import pytest

from test.helpers import omp_function

DIRECTIVE = "parallel for schedule(static, 2) num_threads(2)"


def one_line(text: str, quote: str = '"', prefix: str = "") -> str:
    return f"{prefix}{quote}{text}{quote}"


def split(text: str, separator: str) -> str:
    """The directive written as several literals, one for each word, that Python joins."""
    words = text.split(" ")
    return separator.join(f'"{word} "' if k < len(words) - 1 else f'"{word}"' for k, word in enumerate(words))


STYLES = {
    "double quotes": lambda text: one_line(text),
    "single quotes": lambda text: one_line(text, "'"),
    "raw": lambda text: one_line(text, prefix="r"),
    "u prefix": lambda text: one_line(text, prefix="u"),
    "triple quotes": lambda text: one_line(text, '"""'),
    "triple single quotes": lambda text: one_line(text, "'''"),
    "triple quotes on several lines": lambda text: '"""\n    ' + text.replace(" ", "\n    ", 2) + '\n    """',
    "implicit concatenation": lambda text: split(text, " "),
    "concatenation on several lines": lambda text: "(" + split(text, "\n         ") + ")",
    "concatenation with comments": lambda text: "(" + split(text, "  # next\n         ") + ")",
    "continuation inside the string": lambda text: '"' + text.replace(" ", " \\\n    ", 2) + '"',
    "parentheses": lambda text: "(" + one_line(text) + ")",
    "trailing spaces": lambda text: one_line(text + "   "),
}

TEMPLATE = """
def f(out):
    with omp($directive):
        for i in range(12):
            out.append((i, omp_get_thread_num()))
"""


@pytest.mark.parametrize("style", STYLES)
def test_every_way_of_writing_the_string_gives_the_same_directive(style):
    out = []
    omp_function(TEMPLATE, directive=STYLES[style](DIRECTIVE))(out)
    assert sorted(out) == [(i, (i // 2) % 2) for i in range(12)]


# ---- the Python code of the directive -----------------------------------------------------------------------------
# The directive has no escape sequences, whatever the prefix of the literal is. The Python code that it can have (the
# condition of `if`, the number of threads, the chunk size...) is evaluated later, with its own rules, as it is written.

CODE = {
    "quotes of the other kind": ("""omp("parallel if(word == 'a') num_threads(2)")""", "a", [0, 1]),
    "false condition": ("""omp("parallel if(word == 'a') num_threads(2)")""", "b", [0]),
    "hash inside a string": ("""omp("parallel if(word == 'a#b') num_threads(2)")""", "a#b", [0, 1]),
    "hash inside a string, false": ("""omp("parallel if(word == 'a#b') num_threads(2)")""", "a", [0]),
    "f-string of the code": ("""omp("parallel if(f'{word}!' == 'a!') num_threads(2)")""", "a", [0, 1]),
    "call with a string": ("""omp("parallel num_threads(len('ab'))")""", "a", [0, 1]),
    "conditional expression": ("""omp("parallel num_threads(2 if word else 1)")""", "a", [0, 1]),
    "comprehension": ("""omp("parallel num_threads(sum(1 for _ in 'ab'))")""", "a", [0, 1]),
    "walrus": ("""omp("parallel num_threads((k := 2))")""", "a", [0, 1]),
    # strings that have the characters that the grammar of the clauses uses
    "closing parenthesis in a string": ("""omp("parallel if(word == ')') num_threads(2)")""", ")", [0, 1]),
    "unbalanced parentheses in a string": ("""omp("parallel if(word == '))((') num_threads(2)")""", "))((", [0, 1]),
    "comma in a string": ("""omp("parallel if(word == 'a,b') num_threads(2)")""", "a,b", [0, 1]),
    "colon in a string": ("""omp("parallel if(word == 'a:b') num_threads(2)")""", "a:b", [0, 1]),
    "brackets and braces in a string": ("""omp("parallel if(word == '[{a}]') num_threads(2)")""", "[{a}]", [0, 1]),
    "semicolon in a string": ("""omp("parallel if(word == 'a;b') num_threads(2)")""", "a;b", [0, 1]),
    "clause in a string": ("""omp("parallel if(word == 'num_threads(9)') num_threads(2)")""", "num_threads(9)", [0, 1]),
    "directive in a string": ("""omp("parallel if(word == 'parallel for') num_threads(2)")""", "parallel for", [0, 1]),
    "empty string": ("""omp("parallel if(word == '') num_threads(2)")""", "", [0, 1]),
    "blanks in a string": ("""omp("parallel if(word == '  a  ') num_threads(2)")""", "  a  ", [0, 1]),
    "non ascii string": ("""omp("parallel if(word == 'ñ😀') num_threads(2)")""", "ñ😀", [0, 1]),
    "triple quotes inside": ("""omp("parallel if(word == '''a''') num_threads(2)")""", "a", [0, 1]),
    "both kinds of quotes inside": ('''omp("""parallel if(word == "a'b") num_threads(2)""")''', "a'b", [0, 1]),
    "concatenation in the code": ("""omp("parallel if(word == 'a' 'b') num_threads(2)")""", "ab", [0, 1]),
    "f-string with a format": ("""omp("parallel if(f'{len(word):>3}' == '  1') num_threads(2)")""", "a", [0, 1]),
    "f-string with braces": ("""omp("parallel if(f'{{{word}}}' == '{a}') num_threads(2)")""", "a", [0, 1]),
    "bytes": ("""omp("parallel if(b'a' == bytes(word, 'ascii')) num_threads(2)")""", "a", [0, 1]),
    "raw bytes": ("""omp("parallel if(len(rb'a') == 1 and word) num_threads(2)")""", "a", [0, 1]),
    "triple quotes outside": ('''omp("""parallel if(word == "a") num_threads(2)""")''', "a", [0, 1]),
    "escape sequence of the code": (r"""omp("parallel if('\n' == chr(10) and word) num_threads(2)")""", "a", [0, 1]),
    "backslash of the code": (r"""omp("parallel if('\\' == chr(92) and word) num_threads(2)")""", "a", [0, 1]),
    "raw string of the code": (r"""omp("parallel if(r'\n' == chr(92) + 'n' and word) num_threads(2)")""", "a", [0, 1]),
    "hexadecimal of the code": (r"""omp("parallel if('\x61' == word) num_threads(2)")""", "a", [0, 1]),
    "same with a raw literal": (r"""omp(r"parallel if('\n' == chr(10) and word) num_threads(2)")""", "a", [0, 1]),
    "line continuation": ('omp("parallel \\\n         num_threads(2)")', "a", [0, 1]),
}


@pytest.mark.parametrize("name", CODE)
def test_the_python_code_of_the_directive_works(name):
    directive, word, expected = CODE[name]
    out = []
    omp_function(f"def f(out, word):\n    with {directive}:\n        out.append(omp_get_thread_num())\n")(out, word)
    assert sorted(out) == expected


def test_a_string_in_the_arguments_of_the_schedule():
    out = []
    omp_function(
        """
        def f(out):
            with omp("parallel for schedule(static, int('1') + len('a,b')) num_threads(2)"):
                for i in range(8):
                    out.append((i, omp_get_thread_num()))
        """,
    )(out)
    assert [t for _, t in sorted(out)] == [0] * 4 + [1] * 4


@pytest.mark.parametrize(
    "directive",
    [r'"parallel\tnum_threads(2)"', r'"parallel\nnum_threads(2)"', r'"parallel if(word == \"a\")"', r'r"parallel\tfor"'],
    ids=["tab", "line break", "escaped quotes", "raw"],
)
def test_an_escape_sequence_is_not_a_valid_directive(directive):
    """The OpenMP syntax has no escape sequences, so the directive is rejected instead of guessing what it means."""
    with pytest.raises(SyntaxError):
        omp_function(f"def f(word):\n    with omp({directive}):\n        pass\n")


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_directives_at_different_indentations(depth):
    opening = "".join("    " * level + "if True:\n" for level in range(1, depth))
    indent = "    " * depth
    source = f"""
def f(out):
{opening}{indent}with omp("parallel num_threads(2)"):
{indent}    out.append(omp_get_thread_num())
"""
    out = []
    omp_function(source)(out)
    assert sorted(out) == [0, 1]


# ---- where the error is ------------------------------------------------------------------------------------------


def error_of(directive: str) -> SyntaxError:
    with pytest.raises(SyntaxError) as info:
        omp_function(TEMPLATE, directive=directive)
    return info.value


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("typo", ["scheduel(static)", "num_thread(2)"])
def test_the_error_points_at_the_wrong_token(style, typo):
    error = error_of(STYLES[style](f"parallel for {typo}"))
    line = Path(error.filename).read_text(encoding="utf-8").splitlines()[error.lineno - 1]
    assert line[error.offset - 1 :].startswith(typo.split("(")[0]), (line, error.offset)


@pytest.mark.parametrize("style", STYLES)
def test_the_error_at_the_end_of_the_directive_points_after_the_last_token(style):
    error = error_of(STYLES[style]("parallel for schedule(static,"))
    line = Path(error.filename).read_text(encoding="utf-8").splitlines()[error.lineno - 1]
    assert error.offset - 1 <= len(line)
    assert line[: error.offset - 1].rstrip(" \"'\\)").endswith(",")


# ---- strings that are not valid ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argument",
    [
        "text",
        'f"parallel {text}"',
        'fr"parallel {text}"',
        't"parallel {text}"',
        'b"parallel"',
        'rb"parallel"',
        '"parallel" b"for"',
        "3",
        '"parallel " + text',
        '"parallel %s" % text',
        "None",
    ],
    ids=[
        "variable",
        "f-string",
        "raw f-string",
        "t-string",
        "bytes",
        "raw bytes",
        "mixed",
        "number",
        "sum",
        "format",
        "None",
    ],
)
def test_the_directive_must_be_a_string_literal(argument):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f(text):
                with omp({argument}):
                    pass
            """,
        )


def test_the_directive_cannot_be_given_with_a_keyword():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp(directive="parallel"):
                    pass
            """,
        )


def test_an_empty_directive():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp(""):
                    pass
            """,
        )


def test_the_directive_is_not_case_sensitive_for_its_words():
    out = []
    omp_function(
        """
        def f(out):
            with omp("PARALLEL Num_Threads(2)"):
                out.append(omp_get_thread_num())
        """,
    )(out)
    assert sorted(out) == [0, 1]
