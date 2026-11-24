"""The grammar of the directives and clauses, `openmp.lark`, and the parser that is generated from it.

Any change of the grammar can add a conflict that makes some input be read in a way that nobody wanted, so it is
built with the strictest options of lark. After changing it, the parser has to be generated again (`poe lark`).
"""

from pathlib import Path

import pytest

from omp4py.core.parser import openmp_lark
from omp4py.core.parser import parser as omp_parser

GRAMMAR = (Path(omp_parser.__file__).parent / "openmp.lark").read_text(encoding="utf-8")


def test_the_grammar_has_no_conflicts_of_any_kind():
    """`strict` makes lark raise on any shift/reduce or reduce/reduce conflict and on terminals that match the same
    text; `debug` keeps the information to show them."""
    lark = pytest.importorskip("lark")
    pytest.importorskip("interegular")  # strict needs it to compare the regular expressions of the terminals
    lark.Lark(
        GRAMMAR,
        parser="lalr",
        lexer="contextual",
        strict=True,
        debug=True,
        propagate_positions=True,
        maybe_placeholders=True,
    )


def test_the_generated_parser_is_the_one_of_the_grammar():
    """`openmp_lark.py` is generated from `openmp.lark`: it has to be generated again after a change."""
    lark = pytest.importorskip("lark")
    fresh = lark.Lark(GRAMMAR, parser="lalr", lexer="contextual", propagate_positions=True, maybe_placeholders=True)
    generated = openmp_lark.Lark_StandAlone(propagate_positions=True)
    assert {(t.name, t.pattern.value) for t in fresh.terminals} == {(t.name, t.pattern.value) for t in generated.terminals}
    assert len(fresh.rules) == len(generated.rules)
