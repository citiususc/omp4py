from __future__ import annotations
from operator import sub
from inspect import getargs

import traceback
from typing import Any

import pytest

from omp4py.core.parser import tree
from omp4py.core.parser.parser import _parse
from omp4py.core.parser.source_view import SourceView


def parse(code: str, prefix:str='    with omp("', suffix:str='"):\n') -> tree.Directive:
    complete_code = prefix + code + suffix
    sv = SourceView(
        tree.Span(1, len(prefix), 1, len(prefix)+len(code)),
        "<string>",
        complete_code.splitlines(),
        complete_code,
    )
    return _parse(code, sv)


def find_clause[T](directive: tree.Directive, clause_type: type[T]) -> T | None:
    # It's fine to only get the first one because we will only test with one directive
    for attr in vars(list(directive.constructs.values())[0]).values():
        if isinstance(attr, clause_type):
            return attr
        if isinstance(attr, list) and attr and isinstance(attr[0], clause_type):
            return attr[0]
    return None


#### SYNTAX ERRORS #############################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "",
    "test",
    "(",
    "for (",
    "for )",
    "for ,",
    "for test",
    "for , private(test)",

    "for private(test),",
    "for private(test",
    "for private(test abc)",
    "for private(test,)",
    "for private(test) abc",
])
def test_clause_invalid_syntax(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)
    traceback.print_exception(e.value)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "for private(test!)",
    "for private(a+b)",
    "for private(10)",
    "for private(39+(10+42)*1)",
])
def test_clause_invalid_identifier(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "simd aligned(x: )",
    "simd aligned(x: -1)",
    "simd aligned(x: 1.5)",
    "simd aligned(x: 1+1)",
    "simd aligned(x: n)",

    "simd aligned(x: 4__2)",
    "simd aligned(x: 0b123)",
    "simd aligned(x: 0b0__0)",
    "simd aligned(x: 0o8)",
    "simd aligned(x: 0o911)",
    "simd aligned(x: 0o__1)",
    "simd aligned(x: 0xF__F)",
    "simd aligned(x: 0_x1F)",
])
def test_clause_invalid_integer(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "parallel num_threads()",
    "parallel num_threads(   )",
    "parallel num_threads(((((",
    "parallel num_threads(((()",
    "parallel num_threads(1+)",
])
def test_clause_invalid_expr(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)
    traceback.print_exception(e.value)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "declare reduce (+:test) combiner(test =)",
    "declare reduce (+:test) combiner(if variable:\n\tprint('hello')\nelse:\n\tprint('bye'))",
])
def test_clause_invalid_stmt(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)
    traceback.print_exception(e.value)

## Missing arguments
## Directive name to create orphan clauses
## schedule
#"for schedule(test)",
#"for schedule()",
#"for schedule(static,)",
#"for schedule(static 10)",
#"for schedule(static, )",
#"for schedule(static, 1+)",
## reduction
#"for reduction(+)",
#"for reduction(max)",
#"for reduction(**:test)",
#"for reduction(*:test,)",
#"for reduction(:test)",
#"for reduction(+test)",
#"for reduction(+:)",
#"for reduction(and:)",
#"for reduction(+:10)",
## default
#"parallel default()",
#"parallel default(private)",
#"parallel default(true)",
#"parallel default(Shared)",
#"parallel default(None)",


@pytest.mark.no_isolate
@pytest.mark.parametrize("source", [
    "metadirective when(for private(x))", # context_selector
    "declare variant(var) adjust_args(x)", # adjust_op_name
])
def test_missing_clause_modifiers(source: str) -> None:
    with pytest.raises(SyntaxError) as e:
        parse(source)
    traceback.print_exception(e.value)


#### SIMPLE CLAUSES ############################################################


# Testing: var_list
@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,targets,directive_name", [
    ("scan exclusive(test)",                  tree.Exclusive, ["test"], None),
    ("scan exclusive(test, ççç, äîóòæ)",      tree.Exclusive, ["test", "ççç", "äîóòæ"], None),
    ("scan exclusive(scan: exclusive, scan)", tree.Exclusive, ["exclusive", "scan"], "scan"),

    ("scan inclusive(äîóòæ)",                 tree.Inclusive, ["äîóòæ"], None),
    ("scan inclusive(scan: inclusive, scan)", tree.Inclusive, ["inclusive", "scan"], "scan"),

    ("for private(äîóòæ)",             tree.Private, ["äîóòæ"], None),
    ("for private(for: private, for)", tree.Private, ["private", "for"], "for"),

    ("parallel shared(äîóòæ)",                       tree.Shared, ["äîóòæ"], None),
    ("parallel shared(parallel : shared, parallel)", tree.Shared, ["shared", "parallel"], "parallel"),

    ("for firstprivate(äîóòæ)",                  tree.FirstPrivate, ["äîóòæ"], None),
    ("for firstprivate(for: firstprivate, for)", tree.FirstPrivate, ["firstprivate", "for"], "for"),

    ("for lastprivate(äîóòæ)",                 tree.LastPrivate, ["äîóòæ"], None),
    ("for lastprivate(for: lastprivate, for)", tree.LastPrivate, ["lastprivate", "for"], "for"),

    ("parallel copyin(äîóòæ)",                      tree.CopyIn, ["äîóòæ"], None),
    ("parallel copyin(parallel: copyin, parallel)", tree.CopyIn, ["copyin", "parallel"], "parallel"),

    ("single copyprivate(äîóòæ)",                       tree.CopyPrivate, ["äîóòæ"], None),
    ("single copyprivate(single: copyprivate, single)", tree.CopyPrivate, ["copyprivate", "single"], "single"),

    ("dispatch interop(äîóòæ)",                       tree.InteropClause, ["äîóòæ"], None),
    ("dispatch interop(dispatch: interop, dispatch)", tree.InteropClause, ["interop", "dispatch"], "dispatch"),

    ("dispatch is_device_ptr(äîóòæ)",                       tree.IsDevicePtr, ["äîóòæ"], None),
    ("dispatch is_device_ptr(dispatch: is_device_ptr, dispatch)", tree.IsDevicePtr, ["is_device_ptr", "dispatch"], "dispatch"),

    ("dispatch has_device_addr(äîóòæ)",                       tree.HasDeviceAddr, ["äîóòæ"], None),
    ("dispatch has_device_addr(dispatch: has_device_addr, dispatch)", tree.HasDeviceAddr, ["has_device_addr", "dispatch"], "dispatch"),

    ("declare simd uniform(äîóòæ)",                       tree.Uniform, ["äîóòæ"], None),
    ("declare simd uniform(declare simd: uniform, declare_simd)", tree.Uniform, ["uniform", "declare_simd"], "declare_simd"),
])
def test_data_scopes(source: str, clause_type: type[tree.DataScope], targets: list[str], directive_name: str|None) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert clause.str_targets == targets
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


# Testing: py_expr
@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,property,expr,directive_name", [
    ("parallel if(x > 0)",              tree.If, "expr", "x > 0", None),
    ("parallel if(True)",               tree.If, "expr", "True", None),
    ("parallel if(x > 0 and y < 10)",   tree.If, "expr", "x > 0 and y < 10", None),
    ("parallel if(foo(x, y))",          tree.If, "expr", "foo(x, y)", None),
    ("parallel if(n)",                  tree.If, "expr", "n", None),
    ("parallel if(foo(x, y))",          tree.If, "expr", "foo(x, y)", None),
    ("parallel if(foo(\n\tx,\n\ty))",   tree.If, "expr", "foo(\n\tx,\n\ty)", None),

    ("task final(True)",  tree.Final, "finalize", "True", None),
    ("task final(x > 0)", tree.Final, "finalize", "x > 0", None),

    (
        "declare_induction (+ : int) inductor(omp_var = 1+1) collector(omp_step * omp_idx + 1)",
        tree.Collector,
        "collector_expr",
        "omp_step * omp_idx + 1",
        None
    ),
    (
        "declare_induction (+ : int) inductor(declare_induction: omp_var = 1+1) collector(declare_induction: omp_step * omp_idx + 1)",
        tree.Collector,
        "collector_expr",
        "omp_step * omp_idx + 1",
        "declare_induction"
    ),

    ("scan init_complete(x > 0)", tree.InitComplete, "create_init_phase", "x > 0", None),
    ("scan init_complete(scan: x > 0)", tree.InitComplete, "create_init_phase", "x > 0", "scan"),

    ("allocate(x) align(2**10)", tree.Align, "alignment", "2**10", None),
    ("allocate(x) align(allocate: 2**10)", tree.Align, "alignment", "2**10", "allocate"),

    ("allocate(x) allocator(omp4py.Allocator())", tree.Allocator, "allocator", "omp4py.Allocator()", None),
    ("allocate(x) allocator(allocate: omp4py.Allocator())", tree.Allocator, "allocator", "omp4py.Allocator()", "allocate"),

    ("dispatch nocontext(True)", tree.NoContext, "dont_update_context", "True", None),
    ("dispatch nocontext(dispatch: x > 0)", tree.NoContext, "dont_update_context", "x > 0", "dispatch"),

    ("dispatch novariants(True)", tree.NoVariants, "dont_use_variant", "True", None),
    ("dispatch novariants(dispatch: x > 0)", tree.NoVariants, "dont_use_variant", "x > 0", "dispatch"),

    ("declare simd simdlen(x + y)", tree.Simdlen, "length", "x + y", None),
    ("declare simd simdlen(declare simd: x + y)", tree.Simdlen, "length", "x + y", "declare_simd"),

    ("declare simd inbranch(True)", tree.InBranch, "in_branch", "True", None),
    ("declare simd inbranch(declare simd: x > 0)", tree.InBranch, "in_branch", "x > 0", "declare_simd"),

    ("declare simd notinbranch(True)", tree.NotInBranch, "not_in_branch", "True", None),
    ("declare simd notinbranch(declare simd: x > 0)", tree.NotInBranch, "not_in_branch", "x > 0", "declare_simd"),

    ("declare target indirect", tree.Indirect, "invoked_by_fptr", None, None),
    ("declare target indirect(declare target: x > 0)", tree.Indirect, "invoked_by_fptr", "x > 0", "declare_target"),
])
def test_expr(source: str, clause_type: type[tree.Clause], property: str, expr: str|None, directive_name: str|None) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert hasattr(clause, property)
    if expr is None:
        assert getattr(clause, property) is None
    else:
        assert getattr(clause, property).source == expr
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


# Testing: py_stmt
@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,property,stmt,directive_name", [
    (
        "declare_reduction (+:int) combiner(omp_out = omp_in + omp_out)",
        tree.Combiner,
        "combiner_stmt",
        "omp_out = omp_in + omp_out",
        None
    ),
    (
        "declare_reduction (+:int) combiner(declare_reduction: omp_out = omp_in + omp_out)",
        tree.Combiner,
        "combiner_stmt",
        "omp_out = omp_in + omp_out",
        "declare_reduction"
    ),

    (
        "declare_reduction (+ : int : omp_out = omp_in + omp_out) initializer(omp_priv = omp_orig)",
        tree.Initializer,
        "initializer_stmt",
        "omp_priv = omp_orig",
        None
    ),
    (
        "declare_reduction (+ : int : omp_out = omp_in + omp_out) initializer(declare_reduction: omp_priv = omp_orig)",
        tree.Initializer,
        "initializer_stmt",
        "omp_priv = omp_orig",
        "declare_reduction"
    ),

    (
        "declare_induction (+ : int) inductor(omp_var = 1+1) collector(omp_step * omp_idx + 1)",
        tree.Inductor,
        "inductor_stmt",
        "omp_var = 1+1",
        None
    ),
    (
        "declare_induction (+ : int) inductor(declare_induction: omp_var = 1+1) collector(declare_induction: omp_step * omp_idx + 1)",
        tree.Inductor,
        "inductor_stmt",
        "omp_var = 1+1",
        "declare_induction"
    ),
])
def test_stmt(source: str, clause_type: type[tree.DataScope], property: str, stmt: str, directive_name: str|None) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert hasattr(clause, property)
    assert getattr(clause, property).source == stmt
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,property,num,directive_name", [
    # collapse
    ("simd aligned(x: 1)",        tree.Aligned, "alignment_modifier", 1, None),
    ("simd aligned(x: 4_2)",      tree.Aligned, "alignment_modifier", 42, None),
    ("simd aligned(x: 0b_11_11)", tree.Aligned, "alignment_modifier", 15, None),
    ("simd aligned(x: 0B101)",    tree.Aligned, "alignment_modifier", 5, None),
    ("simd aligned(x: 0o_7_5_5)", tree.Aligned, "alignment_modifier", 493, None),
    ("simd aligned(x: 0x1Ff)",    tree.Aligned, "alignment_modifier", 511, None),
    ("simd aligned(x: 0X1_0)",    tree.Aligned, "alignment_modifier", 16, None),
])
def test_integer(source: str, clause_type: type[tree.Clause], property: str, num: int, directive_name: str|None) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert hasattr(clause, property)
    assert getattr(clause, property).value == num
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,property,value,directive_name", [
    ("groupprivate device_type(host)", tree.DeviceType, "device_type_description", tree.DeviceType.Kind.HOST, None),
])
def test_keyword_arg(
    source: str,
    clause_type: type[tree.Clause],
    property: str,
    value: int,
    directive_name: str|None
) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert hasattr(clause, property)
    assert getattr(clause, property) == value
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,property,value,directive_name", [
    (
        "metadirective when(device={arch('nvptx')}: for private(abc))",
        tree.When,
        "directive",
        "for",
        None
    ),
    (
        "metadirective when(metadirective, device={arch('nvptx')}: for private(x))",
        tree.When,
        "directive",
        "for",
        "metadirective"
    ),

    (
        """
        metadirective
          when(device={arch('nvptx')}: for private(abc)),
          otherwise(metadirective: for)
        """,
        tree.Otherwise,
        "directive",
        "for",
        "metadirective"
    ),
])
def test_sub_directive(
    source: str,
    clause_type: type[tree.Clause],
    property: str,
    value: str|None,
    directive_name: str|None
) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None
    assert hasattr(clause, property)
    if value is not None:
        assert list(getattr(clause, property).constructs.keys())[0] == value
    else:
        assert len(getattr(clause, property).constructs) == 0
    if directive_name is None:
        assert clause.directive_name is None
    else:
        assert clause.directive_name is not None and clause.directive_name.string == directive_name


#### CLAUSES WITH MODIFIERS ####################################################

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,clause_type,fields", [
    (
        "declare variant(var) match(device={arch('nvptx')})",
        tree.Match,
        [("context_selector", tree.ContextSelector)]
    ),
    (
        "declare variant(var) match(device={arch('nvptx')}) adjust_args(nothing: x, y)",
        tree.AdjustArgs,
        [("adjust_op_name", tree.Name)]
    ),
    (
        "declare variant(var) match(device={arch('nvptx')}) append_args(interop(target, target, targetsync))",
        tree.AppendArgs,
        [("append_op", tree.InteropModifier)]
    ),
    (
        "declare simd linear(x, y, z)",
        tree.Linear,
        [("targets", list)]
    ),
    (
        "declare target enter(x, y, z)",
        tree.Enter,
        [("targets", list)]
    ),
    (
        "declare target enter(automap: x, y, z)",
        tree.Enter,
        [("targets", list), ("automap_name", tree.Name)]
    ),
])
def test_clause_with_modifiers(
    source: str,
    clause_type: type[tree.Clause],
    fields: list[tuple[str, type|None]],
) -> None:
    directive = parse(source)
    clause = find_clause(directive, clause_type)
    assert clause is not None

    for field_name, field_type in fields:
        assert hasattr(clause, field_name)
        if field_type is not None:
            assert isinstance(getattr(clause, field_name), field_type)
        else:
            assert getattr(clause, field_name) is None

