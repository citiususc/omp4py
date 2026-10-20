from __future__ import annotations

import pytest

from omp4py.core.parser.parser import preprocesor

CASES = [
    # Empty strings
    ('""',       " "*2),
    ("''",       " "*2),
    ("''''''",   " "*6),
    ('""""""',   " "*6),
    ('r""""""',  " "*7),
    ('rb""""""', " "*8),

    # String prefixes
    ('r"rub"',  "  rub "),
    ('R"RUB"',  "  RUB "),
    ('b""',  " "*3),
    ('B""',  " "*3),
    ('u""',  " "*3),
    ('U""',  " "*3),
    ('br""', " "*4),
    ('Br""', " "*4),
    ('bR""', " "*4),
    ('BR""', " "*4),
    ('rb""', " "*4),
    ('rB""', " "*4),
    ('Rb""', " "*4),
    ('RB""', " "*4),

    # ANY: plain content, no escapes
    ('"hello"',       " hello "),
    ("'hello'",       " hello "),
    ('"""hello"""',   "   hello   "),
    ("'''hello'''",   "   hello   "),
    ('"hello world"', " hello world "),

    # SINGLE_SCAPE_SEQ: recognized single-char escapes
    (r'"\n"',  " "*4),
    (r'"\t"',  " "*4),
    (r'"\r"',  " "*4),
    (r'"\\"',  " "*4),
    (r'"\'"',  " "*4),
    (r'"\""',  " "*4),
    (r'"\a"',  " "*4),
    (r'"\b"',  " "*4),
    (r'"\f"',  " "*4),
    (r'"\v"',  " "*4),
    ('"\\\n"', " "*4),

    # OCTAL_SCAPE
    (r'"\0"',   " "*4),
    (r'"\07"',  " "*5),
    (r'"\077"', " "*6),

    # HEX_SCAPE
    (r'"\x00"',  " "*6),
    (r'"\x41"',  " "*6),
    (r'"\xff"',  " "*6),
    (r'"\xFF"',  " "*6),

    # UNICODE_SCAPE: \uXXXX (4 hex digits)
    (r'"\u0041"', " "*8),
    (r'"\u00ff"', " "*8),
    (r'"\uFFFF"', " "*8),

    # UNICODE_SCAPE: \UXXXXXXXX (8 hex digits)
    (r'"\U00000041"', " "*12),
    (r'"\U0001F600"', " "*12),

    # NAMED_UNICODE_SCAPE
    (r'"\N{LATIN SMALL LETTER A}"',     " "*26),
    (r'"\N{snowman}"',                  " "*13),
    (r'"\N{Greek Small Letter Alpha}"', " "*30),

    # UNRECOGNIZED_SCAPE_SEQ: unknown escapes pass through
    (r'"\p"', r" \p "),
    (r'"\q"', r" \q "),
    (r'"\j"', r" \j "),

    # NEWLINE in triple-quoted strings
    ('"""line1\nline2"""',   "   line1\nline2   "),
    ("'''line1\nline2'''",   "   line1\nline2   "),
    ('"""line1\n\nline2"""', "   line1\n\nline2   "),

    # Quote characters allowed inside triple-quoted strings
    ('"""she said "hi" """',     '   she said "hi"    '),
    ("'''it's fine'''",          "   it's fine   "),
    ('"""one " two "" three"""', '   one " two "" three   '),
    ("'''one ' two '' three'''", "   one ' two '' three   "),

    # Opposite quote delimiter inside single-quoted strings
    ('"it\'s"',      " it's "),
    ("'say \"hi\"'", ' say "hi" '),

    # Mixed content: ANY + escapes
    (r'"hello\nworld"', " hello  world "),
    (r'"col:\x41end"',  " col:    end "),
    (r'"a\tb\tc"',      " a  b  c "),

    # Multiple escape sequences in a row
    (r'"\n\t\r"',   " " * 8),
    (r'"\x41\x42"', " " * 10),

    # Triple-quoted with escapes
    ('"""\\n"""',     " " * 8),
    ('"""\\x41"""',   " " * 10),
    ('b"""\\x41"""',  " " * 11),
    ('rb"""\\x41"""', " " * 12),
]

@pytest.mark.no_isolate
@pytest.mark.parametrize("input,expected", CASES)
def test_string_preprocessor(input: str, expected: str) -> None:
    content = preprocesor.parse(input)
    assert len(content) == len(input)
    assert content == expected
