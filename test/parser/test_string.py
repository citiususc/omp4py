"""The text of a directive that `omp4py` gives to the OpenMP parser.

The directive is a Python string literal, but lark has to see only the directive. `extract_directive` cleans the
literal: everything that is Python syntax (the quotes, the prefix `r` `u`, the line continuations, the comments
between the parts of a concatenation...) is replaced by blanks, so every token of the directive is in the same
line and column that it has in the source file, and the errors of the parser point at the right place of the file.
The escape sequences are not decoded, whatever the prefix of the literal is: the OpenMP syntax has none, and the
Python code that the directive can have (`if('\n' == c)`) is evaluated later, with its own rules, as it is written.
So the text is always the same length as the literal.

Every case is a module level variable `string_<name>` with a string literal, which is read from the source of this
file, so it is written exactly as it would be in the code that uses omp4py. To add a case write another variable.
A case that needs a special source (line endings, other code before the literal) goes to `SOURCES`.
"""

import ast
import pathlib
import re

import pytest

from omp4py.core.parser.parser import extract_directive

# ---- the ways of writing a literal -------------------------------------------------------------------------------

string_double_quotes = "parallel for"

string_single_quotes = 'parallel for'

string_raw = r"parallel for"

string_u_prefix = u"parallel for"

string_uppercase_prefixes = R"parallel " U"for"

string_raw_after_blanks = r"   parallel"

string_blanks_around = "   parallel   "

string_triple_double_quotes = """parallel for"""

string_triple_single_quotes = '''parallel for
   schedule(guided)'''

string_raw_triple_quotes = r"""parallel
   for
   num_threads(2)"""

string_triple_quotes_with_quotes_inside = """parallel for if(" a " in b)"""

string_same_line_concatenation = "parallel" " for" "  schedule(static)"

string_concatenation_without_blank = "parallel" "for"

string_mixed_prefixes = r"parallel " 'for ' u"schedule(static)"

string_concatenation_in_lines = "parallel" \
                                " for" \
                                " schedule(static)"

string_concatenation_in_parentheses = ("parallel "
                                       "for "
                                       "schedule(dynamic)")

string_concatenation_with_comments = ("parallel "  # the construct
                                      "for "  # the loop
                                      "schedule(dynamic)")

string_parentheses = ("parallel for")

string_nested_parentheses = (("parallel for"))

string_continuation_inside_the_string = "parallel  \
            for"

string_continuation_with_indentation = "parallel \
for \
schedule(static)"

string_continuation_in_triple_quotes = """parallel \
for \
schedule(static)"""

# ---- what the directive contains ---------------------------------------------------------------------------------

string_unicode = "parallel for if('ñ' in word)"

string_unicode_outside_the_basic_plane = "parallel for if('😀' in word) num_threads(2)"

string_tab = "parallel	for"

string_only_blanks = "   "

string_empty = ""

string_quotes_of_the_other_kind = "parallel for if('a' == b)"

string_quotes_of_the_other_kind_in_single = 'parallel for if("a" == b)'

string_hash_inside_a_string = "parallel if(word == 'a#b') num_threads(2)"

string_text_that_looks_like_a_prefix = "r parallel u"

string_text_that_looks_like_a_prefix_in_raw = r"r'parallel'"

# ---- escape sequences: they are not decoded, they are part of the Python code of the directive ------------------

string_escaped_double_quote = "parallel for if(\"a\" == b)"

string_escaped_single_quote = 'parallel for if(\'a\' == b)'

string_escaped_new_line = "parallel for if(a == '\n')"

string_escaped_tab = "parallel for if(a == '\t')"

string_escaped_backslash = "parallel for if(a == '\\')"

string_escaped_hexadecimal = "parallel for if(a == '\x61')"

string_escaped_octal = "parallel for if(a == '\141')"

string_escaped_unicode_32_bits = "parallel for if(a == '\U0001F600')"

string_escaped_unicode_name = "parallel for if(a == '\N{LATIN SMALL LETTER N WITH TILDE}')"

string_escaped_in_triple_quotes = """parallel for
   if(a == '\t' and b == \"q\")"""

# a raw string keeps its backslashes, and so does a string that is not raw
string_escaped_in_raw = r"parallel for if(a == '\n' and b == '\\')"

string_escaped_in_raw_uppercase = R"parallel for if(a == '\x32')"

string_concatenation_raw_and_not_raw = r"parallel for if(a == '\n') " "if(b == '\t')"


def read_cases() -> tuple[str, dict[str, ast.Constant]]:
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    cases: dict[str, ast.Constant] = {}
    for node in ast.parse(source).body:
        match node:
            case ast.Assign(targets=[ast.Name(name)], value=ast.Constant() as value) if name.startswith("string_"):
                cases[name.removeprefix("string_")] = value
    return source, cases


SOURCE, CASES = read_cases()
LINE = re.compile(r"\r\n|\r|\n")  # the line endings of Python, that does not count a form feed
ESCAPE = re.compile(r"\\(?!\n)")  # a backslash that is not a line continuation


def check(node: ast.Constant, source: str) -> None:
    """Check the two properties of the extracted text."""
    text = extract_directive(node, source, __file__)
    literal = ast.get_source_segment(source, node)
    assert literal is not None

    # The tokens are in the same line and column that they have in the literal, whose first character is (1, 1).
    assert len(text) == len(literal), (text, literal)
    literal_lines = LINE.split(literal)
    text_lines = LINE.split(text)
    assert len(text_lines) == len(literal_lines), (text, literal)
    for number, line in enumerate(text_lines):
        for column, char in enumerate(line):
            in_literal = literal_lines[number][column : column + 1]
            assert char.isspace() or in_literal == char, (
                f"{char!r} is at line {number + 1}, column {column + 1} of the text, but the literal has "
                f"{in_literal!r} there",
                text,
                literal,
            )

    # Nothing else than the directive is left: without blanks, the text is the value of the literal, and when it has
    # escape sequences, that are not decoded, what is written between the quotes.
    if not ESCAPE.search(literal):
        assert "".join(text.split()) == "".join(str(node.value).split()), (text, literal)


@pytest.mark.parametrize("name", CASES)
def test_the_directive_is_cleaned_and_its_tokens_keep_their_position(name):
    check(CASES[name], SOURCE)


@pytest.mark.parametrize(
    "name",
    [name for name in CASES if name.startswith("escaped_") and "concatenation" not in name],
)
def test_the_escape_sequences_are_not_decoded(name):
    """The text is what is written between the quotes, with the prefix and the quotes as blanks, whether the literal
    is raw or not: the Python code of the directive is evaluated later, and `'\\n'` is what the user wrote."""
    literal = ast.get_source_segment(SOURCE, CASES[name])
    assert literal is not None
    quotes = re.match(r"[ru]?(\"\"\"|'''|\"|')", literal, re.IGNORECASE)
    assert quotes is not None
    body = literal[len(quotes.group()) : -len(quotes.group(1))]
    assert extract_directive(CASES[name], SOURCE, __file__).strip(" ") == body


# ---- sources that are not a module level assignment --------------------------------------------------------------

SOURCES = {
    "windows line endings": 'x = ("parallel "\r\n     "for "\r\n     "schedule(static)")\r\n',
    "code before the literal in the line": 'a = 1; x = "parallel for"\n',
    "characters of several bytes before the literal": 'a = "ñññ"; x = "parallel for"\n',
    "characters outside the basic plane before the literal": 'a = "😀😀"; x = "parallel for"\n',
    "literal in a call": 'x = print("😀", "parallel for", end="")\n',
    "literal after a tab": 'if True:\n\tx = "parallel for"\n',
    "literal with a form feed": 'x = "parallel\x0cfor"\n',
}


def find_literal(source: str) -> ast.Constant:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "parallel" in node.value:
            return node
    raise AssertionError("no literal")


@pytest.mark.parametrize("name", SOURCES)
def test_the_directive_is_cleaned_in_any_source(name):
    source = SOURCES[name]
    check(find_literal(source), source)
