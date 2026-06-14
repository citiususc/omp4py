from __future__ import annotations

import pytest

from omp4py.core.parser.parser import preprocesor

CASES = [
    # Empty strings
    ('""',       "", 1, 1),
    ("''",       "", 1, 1),
    ("''''''",   "", 3, 3),
    ('""""""',   "", 3, 3),
    ('r""""""',  "", 4, 3),
    ('rb""""""', "", 5, 3),

    # String prefixes
    ('r"rub"',  "rub", 2, 1),
    ('R"RUB"',  "RUB", 2, 1),
    ('b""',  "", 2, 1),
    ('B""',  "", 2, 1),
    ('u""',  "", 2, 1),
    ('U""',  "", 2, 1),
    ('br""', "", 3, 1),
    ('Br""', "", 3, 1),
    ('bR""', "", 3, 1),
    ('BR""', "", 3, 1),
    ('rb""', "", 3, 1),
    ('rB""', "", 3, 1),
    ('Rb""', "", 3, 1),
    ('RB""', "", 3, 1),

    # ANY: plain content, no escapes
    ('"hello"',       "hello",       1, 1),
    ("'hello'",       "hello",       1, 1),
    ('"""hello"""',   "hello",       3, 3),
    ("'''hello'''",   "hello",       3, 3),
    ('"hello world"', "hello world", 1, 1),

    # SINGLE_SCAPE_SEQ: recognized single-char escapes
    (r'"\n"',  " "*2, 1, 1),
    (r'"\t"',  " "*2, 1, 1),
    (r'"\r"',  " "*2, 1, 1),
    (r'"\\"',  " "*2, 1, 1),
    (r'"\'"',  " "*2, 1, 1),
    (r'"\""',  " "*2, 1, 1),
    (r'"\a"',  " "*2, 1, 1),
    (r'"\b"',  " "*2, 1, 1),
    (r'"\f"',  " "*2, 1, 1),
    (r'"\v"',  " "*2, 1, 1),
    ('"\\\n"', " "*2, 1, 1),

    # OCTAL_SCAPE
    (r'"\0"',   " "*2, 1, 1), # 1 digit
    (r'"\07"',  " "*3, 1, 1), # 2 digits
    (r'"\077"', " "*4, 1, 1), # 3 digits

    # HEX_SCAPE
    (r'"\x00"',  " "*4, 1, 1),
    (r'"\x41"',  " "*4, 1, 1),
    (r'"\xff"',  " "*4, 1, 1),
    (r'"\xFF"',  " "*4, 1, 1),

    # UNICODE_SCAPE: \uXXXX (4 hex digits)
    (r'"\u0041"', " "*6, 1, 1),  # 'A'
    (r'"\u00ff"', " "*6, 1, 1),
    (r'"\uFFFF"', " "*6, 1, 1),

    # UNICODE_SCAPE: \UXXXXXXXX (8 hex digits)
    (r'"\U00000041"', " "*10, 1, 1),
    (r'"\U0001F600"', " "*10, 1, 1),

    # NAMED_UNICODE_SCAPE
    (r'"\N{LATIN SMALL LETTER A}"',     " "*24, 1, 1),
    (r'"\N{snowman}"',                  " "*11, 1, 1),
    (r'"\N{Greek Small Letter Alpha}"', " "*28, 1, 1),

    # UNRECOGNIZED_SCAPE_SEQ: unknown escapes pass through
    (r'"\p"', r"\p", 1, 1),
    (r'"\q"', r"\q", 1, 1),
    (r'"\j"', r"\j", 1, 1),

    # NEWLINE in triple-quoted strings
    ('"""line1\nline2"""',   "line1\nline2",   3, 3),
    ("'''line1\nline2'''",   "line1\nline2",   3, 3),
    ('"""line1\n\nline2"""', "line1\n\nline2", 3, 3),

    # Quote characters allowed inside triple-quoted strings
    ('"""she said "hi" """',     'she said "hi" ',     3, 3),
    ("'''it's fine'''",          "it's fine",          3, 3),
    ('"""one " two "" three"""', 'one " two "" three', 3, 3),
    ("'''one ' two '' three'''", "one ' two '' three", 3, 3),

    # Opposite quote delimiter inside single-quoted strings
    ('"it\'s"',      "it's",     1, 1),
    ("'say \"hi\"'", 'say "hi"', 1, 1),

    # Mixed content: ANY + escapes
    (r'"hello\nworld"', "hello  world", 1, 1),
    (r'"col:\x41end"',  "col:    end",  1, 1),
    (r'"a\tb\tc"',      "a  b  c",      1, 1),

    # Multiple escape sequences in a row
    (r'"\n\t\r"',   " " * 6, 1, 1),
    (r'"\x41\x42"', " " * 8, 1, 1),

    # Triple-quoted with escapes
    ('"""\\n"""',     " " * 2, 3, 3),
    ('"""\\x41"""',   " " * 4, 3, 3),
    ('b"""\\x41"""',  " " * 4, 4, 3),
    ('rb"""\\x41"""', " " * 4, 5, 3),
]

@pytest.mark.no_isolate
@pytest.mark.parametrize("input,expected,expected_begin,expected_end", CASES)
def test_string_preprocessor(
    input: str,
    expected: str,
    expected_begin: int,
    expected_end: int,
) -> None:
    out, begin, end = preprocesor.parse(input)
    assert out   == expected
    assert begin == expected_begin
    assert end   == expected_end
