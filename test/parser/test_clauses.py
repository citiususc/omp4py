from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from omp4py.core.parser import tree
from omp4py.core.parser.parser import _parse
from omp4py.core.parser.source_view import SourceView


def parse(code: str) -> tree.Directive:
    sv = SourceView(
        tree.Span(0, 0, 0, 0),
        "<string>",
        code.splitlines(),
        code,
    )
    return _parse(code, sv)


def find_clause[T](directive: tree.Directive, clause_type: type[T]) -> T | None:
    for attr in vars(directive.construct).values():
        if isinstance(attr, clause_type):
            return attr
        # It's fine to only get the first one because we will only test with one clause
        if isinstance(attr, list) and attr and isinstance(attr[0], clause_type):
            return attr[0]
    return None


################################################################################


INVALID_CLAUSES = [
    # Invalid syntax
    "",
    "test",
    "(",
    "for (",
    "for )",
    "for ,",
    "for test",
    "for for",
    "for , private(test)",

    "for private(test),",
    "for private(test",
    "for private(test abc)",
    "for private(test,)",
    "for private(test) abc",

    # Missing arguments
    "for private()",
    "for reduction()",
    "for schedule()",

    # Invalid identifiers
    "for private(test!)",
    "for private(a+b)",
    "for private(10)",
    "for private(39+(10+42)*1)",

    # Invalid integers
    "for collapse()",
    "for collapse(-1)",
    "for collapse(1.5)",
    "for collapse(1+1)",
    "for collapse(n)",

    "for collapse(4__2)",
    "for collapse(0b123)",
    "for collapse(0b0__0)",
    "for collapse(0o8)",
    "for collapse(0o911)",
    "for collapse(0o__1)",
    "for collapse(0xF__F)",
    "for collapse(0_x1F)",

    # Correct syntax but collapse(0) is not allowed
    "for collapse(0)",
    "for collapse(00_000_0)",
    "for collapse(0O0)",

    # Expressions
    # They are also present in if() and final(), but as they are the same rule,
    # so testing one is enough.
    "parallel num_threads()",
    "parallel num_threads(   )",
    "parallel num_threads(((((",
    "parallel num_threads(((()",
    "parallel num_threads(1+)",

    # schedule
    "for schedule(test)",
    "for schedule()",
    "for schedule(static,)",
    "for schedule(static 10)",
    "for schedule(static, )",
    "for schedule(static, 1+)",

    # shared / firstprivate / lastprivate / copyin / copyprivate
    "parallel shared()",
    "for firstprivate()",
    "for lastprivate()",
    "parallel copyin()",
    "single copyprivate()",
    "parallel shared(10)",
    "for firstprivate(x + y)",

    # reduction
    "for reduction(+)",
    "for reduction(max)",
    "for reduction(**:test)",
    "for reduction(*:test,)",
    "for reduction(:test)",
    "for reduction(+test)",
    "for reduction(+:)",
    "for reduction(and:)",
    "for reduction(+:10)",

    # default
    "parallel default()",
    "parallel default(private)",
    "parallel default(true)",
    "parallel default(Shared)",
    "parallel default(None)",

    # flag clauses with spurious arguments
    "for nowait(x)",
    "for ordered(x)",
    "task untied(x)",
    "task mergeable(x)",
]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", INVALID_CLAUSES)
def test_clause_invalid(source: str) -> None:
    with pytest.raises(SyntaxError):
        parse(source)


################################################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,targets", [
    # private
    ("for private(test)", tree.Private, ["test"]),
    ("for private(test, ççç, äîóòæ)", tree.Private, ["test", "ççç", "äîóòæ"]),
    ("for private(private, reduction, schedule, num_threads)", tree.Private, ["private", "reduction", "schedule", "num_threads"]),

    # shared
    ("parallel shared(x)",                        tree.Shared, ["x"]),
    ("parallel shared(x, y, z)",                  tree.Shared, ["x", "y", "z"]),
    ("parallel shared(shared, private, default)", tree.Shared, ["shared", "private", "default"]),

    # firstprivate
    ("for firstprivate(x)",    tree.FirstPrivate, ["x"]),
    ("for firstprivate(x, y)", tree.FirstPrivate, ["x", "y"]),

    # lastprivate
    ("for lastprivate(x)",    tree.LastPrivate, ["x"]),
    ("for lastprivate(x, y)", tree.LastPrivate, ["x", "y"]),

    # copyin
    ("parallel copyin(x)",    tree.CopyIn, ["x"]),
    ("parallel copyin(x, y)", tree.CopyIn, ["x", "y"]),

    # copyprivate
    ("single copyprivate(x)",    tree.CopyPrivate, ["x"]),
    ("single copyprivate(x, y)", tree.CopyPrivate, ["x", "y"]),
])
def test_data_scopes(source: str, clause_type: type[tree.DataScope], targets: list[str]) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert clause.str_targets == targets


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,op,targets", [
    ("for reduction(+:test)",             "+",   ["test"]),
    ("for reduction(+:test, var)",        "+",   ["test", "var"]),
    ("for reduction(+:test, ççç, äîóòæ)", "+",   ["test", "ççç", "äîóòæ"]),
    ("for reduction(+:private, reduction, schedule, num_threads)", "+", ["private", "reduction", "schedule", "num_threads"]),
    ("for reduction(-:test)",             "-",   ["test"]),
    ("for reduction(*:test)",             "*",   ["test"]),
    ("for reduction(&:test)",             "&",   ["test"]),
    ("for reduction(|:test)",             "|",   ["test"]),
    ("for reduction(^:test)",             "^",   ["test"]),
    ("for reduction(and:test)",           "and", ["test"]),
    ("for reduction(or:test)",            "or",  ["test"]),
    ("for reduction(max:test)",           "max", ["test"]),
    ("for reduction(min:test)",           "min", ["test"]),
    ("for reduction(  +  :  x  )",        "+",   ["x"]),
])
def test_reduction(source: str, op: str, targets: list[str]) -> None:
    directive = parse(source)
    assert isinstance(directive.construct, tree.For)
    assert len(directive.construct.reduction) == 1
    clause = directive.construct.reduction[0]
    assert clause.op.value == op
    assert clause.str_targets == targets


sk = tree.ScheduleType.Kind
@pytest.mark.no_isolate
@pytest.mark.parametrize("source,kind,chunk", [
    ("for schedule(static)",            sk.STATIC,  None),
    ("for schedule(dynamic)",           sk.DYNAMIC, None),
    ("for schedule(guided)",            sk.GUIDED,  None),
    ("for schedule(auto)",              sk.AUTO,    None),
    ("for schedule(runtime)",           sk.RUNTIME, None),
    ("for schedule(static,10)",         sk.STATIC,  "10"),
    ("for schedule(dynamic,4)",         sk.DYNAMIC, "4"),
    ("for schedule(guided,2)",          sk.GUIDED,  "2"),
    ("for schedule(auto,10+9*1/(1+1))", sk.AUTO,    "10+9*1/(1+1)"),
    ("for schedule(static,n)",          sk.STATIC,  "n"),
    ("for schedule(dynamic,n*2)",       sk.DYNAMIC, "n*2"),
    ("for schedule(dynamic,  n*2\t)",   sk.DYNAMIC, "  n*2\t"),
])
def test_schedule(source: str, kind: sk, chunk: str|None) -> None:
    directive = parse(source)
    assert isinstance(directive.construct, tree.For)
    clause = directive.construct.schedule
    assert clause is not None
    assert clause.type.kind == kind
    assert (clause.chunk.source if clause.chunk is not None else None) == chunk


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,expr", [
    ("parallel if(x > 0)",             tree.If, "x > 0"),
    ("parallel if(True)",              tree.If, "True"),
    ("parallel if(x > 0 and y < 10)",  tree.If, "x > 0 and y < 10"),
    ("parallel if(foo(x, y))",         tree.If, "foo(x, y)"),
    ("parallel if(   n    )",          tree.If, "   n    "),
    ("parallel if(    foo(x, y)\t\t)", tree.If, "    foo(x, y)\t\t"),

    # num_threads
    ("parallel num_threads(8)",                 tree.NumThreads, "8"),
    ("parallel num_threads((1 + sqrt(5))/2)",   tree.NumThreads, "(1 + sqrt(5))/2"),
    ("parallel num_threads(get_threads())",     tree.NumThreads, "get_threads()"),
    ("parallel num_threads(   n    )",          tree.NumThreads, "   n    "),
    ("parallel num_threads(    foo(x, y)\t\t)", tree.NumThreads, "    foo(x, y)\t\t"),

    # final
    ("task final(True)",  tree.Final, "True"),
    ("task final(x > 0)", tree.Final, "x > 0"),
])
def test_expr(source: str, clause_type: type[tree.Clause], expr: str) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert clause.expr.source == expr  # type: ignore[attr-defined]


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,num", [
    # collapse
    ("for collapse(1)", tree.Collapse, 1),
    ("for collapse(4_2)", tree.Collapse, 42),
    ("for collapse(0b_11_11)", tree.Collapse, 15),
    ("for collapse(0B101)", tree.Collapse, 5),
    ("for collapse(0o_7_5_5)", tree.Collapse, 493),
    ("for collapse(0x1Ff)", tree.Collapse, 511),
    ("for collapse(0X1_0)", tree.Collapse, 16),
])
def test_int(source: str, clause_type: type[tree.Clause], num: int) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert clause.num.value == num  # type: ignore[attr-defined]


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type", [
    ("for nowait",     tree.NoWait),
    ("for ordered",    tree.OrderedClause),
    ("task untied",    tree.Untied),
    ("task mergeable", tree.Mergeable),
])
def test_flags(source: str, clause_type: type[tree.Clause]) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,option", [
    # default
    ("parallel default(shared)", tree.Default, tree.Default.Type.SHARED),
    ("parallel default(none)",   tree.Default, tree.Default.Type.NONE),
])
def test_keyword_option(source: str, clause_type: type[tree.Clause], option: str) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert isinstance(clause, tree.Default) # Remove to add other clauses
    assert clause.type == option

