from __future__ import annotations

import inspect
import itertools
import types
import typing
from dataclasses import dataclass, field

import pytest

from omp4py.core.parser import tree, Parallel, Clause
from omp4py.core.parser.parser import _parse
from omp4py.core.parser.source_view import SourceView


#### HELPER FUNCTIONS ##########################################################


def parse(code: str) -> tree.Directive:
    sv = SourceView(
        tree.Span(0, 0, 0, 0),
        "<string>",
        code.splitlines(),
        code,
    )
    return _parse(code, sv)


@dataclass
class ClauseInfo:
    all: dict[str, type[tree.Clause]] = field(default_factory=dict)
    repeatable: dict[str, type[tree.Clause]] = field(default_factory=dict)
    optional: dict[str, type[tree.Clause]] = field(default_factory=dict)


    @staticmethod
    def from_type(cls: type[tree.Construct], exclude: set[str]|None=None) -> ClauseInfo:
        # Collect class variables to ignore them later
        exclude_vars = {
            name for name, hint in inspect.get_annotations(cls).items()
            if typing.get_origin(hint) is typing.ClassVar
        }

        if exclude is not None:
            exclude_vars.update(exclude)

        clause_info = ClauseInfo()
        for var_name, hint in typing.get_type_hints(cls).items():
            if var_name in exclude_vars:
                continue

            original_type = typing.get_origin(hint)
            type_args     = typing.get_args(hint)

            # If the type is something like list[Private] or list[Reduction],
            # it means that this clause is repeatable.
            if original_type is list and type_args and issubclass(type_args[0], tree.Clause):
                clause_info.all[var_name]        = type_args[0]
                clause_info.repeatable[var_name] = type_args[0]

            # Otherwise, if the type is NoWait|None or Schedule|None,
            # it means that this clause is not repeatable and optional
            elif original_type is types.UnionType or original_type is typing.Union:
                # Ignore the None types
                non_none = [t for t in type_args if t is not type(None)]
                # Check that the other type is actually a clause
                if len(non_none) == 1 and isinstance(non_none[0], type) and issubclass(non_none[0], tree.Clause):
                    clause_info.all[var_name]      = non_none[0]
                    clause_info.optional[var_name] = non_none[0]

        return clause_info

    @staticmethod
    def merge(info1: ClauseInfo, info2: ClauseInfo) -> ClauseInfo:
        return ClauseInfo(
            all        = {**info1.all,        **info2.all},
            repeatable = {**info1.repeatable, **info2.repeatable},
            optional   = {**info1.optional,   **info2.optional},
        )


    def repeatable_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.repeatable]

    def optional_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.optional]

    def all_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.all]


    def repeatable_joined(self, sep: str=" ") -> str:
        return sep.join(self.repeatable_examples())

    def optional_joined(self, sep: str=" ") -> str:
        return sep.join(self.optional_examples())

    def all_joined(self, sep: str=" ") -> str:
        return sep.join(self.all_examples())


@dataclass
class DirectiveSpec:
    name: str
    cls: type[tree.Construct]
    clauses: ClauseInfo = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clauses", ClauseInfo.from_type(self.cls))


#### CONSTANT DEFINITIONS ######################################################

CLAUSE_SEPARATORS = [",", ", ", ",\t", "\t", "\n"]
CLAUSES = {
    "collapse":      "collapse(2)",
    "first_private": "firstprivate(x)",
    "last_private":  "lastprivate(x)",
    "no_wait":       "nowait",
    "ordered":       "ordered",
    "private":       "private(x)",
    "reduction":     "reduction(+:x)",
    "schedule":      "schedule(static)",
    "shared":        "shared(x)",
    "num_threads":   "num_threads(4)",
    "if_":           "if(True)",
    "copyin":        "copyin(x)",
    "copyprivate":   "copyprivate(x)",
    "final":         "final(True)",
    "untied":        "untied",
    "mergeable":     "mergeable",
    "default":       "default(none)",
    "proc_bind":     "", # TODO: not in the grammar yet
}

DIRECTIVES = [
    DirectiveSpec("parallel", tree.Parallel),
    DirectiveSpec("for",      tree.For),
    DirectiveSpec("sections", tree.Sections),
    DirectiveSpec("single",   tree.Single),
    DirectiveSpec("task",     tree.Task),

    # No clauses
    DirectiveSpec("section",   tree.Section),
    DirectiveSpec("taskyield", tree.TaskYield),
    DirectiveSpec("master",    tree.Master),
    DirectiveSpec("barrier",   tree.Barrier),
    DirectiveSpec("taskwait",  tree.TaskWait),
    DirectiveSpec("ordered",   tree.Ordered),
    DirectiveSpec("critical",  tree.Critical), # TODO: test with identifier

    # TODO: test special constructs
    # DirectiveSpec("atomic",        tree.Atomic),
    # DirectiveSpec("flush",         tree.Flush),
    # DirectiveSpec("threadprivate", tree.ThreadPrivate),
]

PARALLEL_FOR_CLAUSES = ClauseInfo.merge(
    ClauseInfo.from_type(tree.Parallel),
    ClauseInfo.from_type(tree.For, exclude={"no_wait"}),
)
PARALLEL_SECTIONS_CLAUSES = ClauseInfo.merge(
    ClauseInfo.from_type(tree.Parallel),
    ClauseInfo.from_type(tree.Sections, exclude={"no_wait"}),
)


#### EMPTY DIRECTIVES ##########################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_empty_directive(directive: DirectiveSpec) -> None:
    ast = parse(directive.name)
    assert isinstance(ast.construct, directive.cls)

@pytest.mark.no_isolate
def test_empty_parallel_for() -> None:
    ast = parse("parallel for")
    assert isinstance(ast.construct, tree.ParallelFor)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.for_, tree.For)

@pytest.mark.no_isolate
def test_empty_parallel_sections() -> None:
    ast = parse("parallel sections")
    assert isinstance(ast.construct, tree.ParallelSections)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.sections, tree.Sections)


#### DIRECTIVE WITH ALL CLAUSES ################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_all_clauses(directive: DirectiveSpec) -> None:
    source = directive.name + " " + directive.clauses.all_joined()
    ast = parse(source)
    assert isinstance(ast.construct, directive.cls)

@pytest.mark.no_isolate
def test_all_clauses_parallel_for() -> None:
    source = "parallel for " + PARALLEL_FOR_CLAUSES.all_joined()
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelFor)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.for_, tree.For)

@pytest.mark.no_isolate
def test_all_clauses_parallel_sections() -> None:
    source = "parallel sections " + PARALLEL_SECTIONS_CLAUSES.all_joined()
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelSections)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.sections, tree.Sections)


#### TEST CLAUSES WITH DIFFERENT SEPARATORS ####################################


def _gen_separator_cases() -> list[tuple[str, type]]:
    cases = []
    for directive in DIRECTIVES:
        all_clauses = directive.clauses.all_examples()

        # Use only the first 2
        if len(all_clauses) <= 1:
            continue

        a, b = all_clauses[0], all_clauses[1]
        cases.extend((f"{directive.name} {a}{sep}{b}", directive.cls) for sep in CLAUSE_SEPARATORS)

    return cases

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,construct_cls", _gen_separator_cases())
def test_different_clause_separators(source: str, construct_cls: type) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, construct_cls)


#### COMBINED DIRECTIVES ####

def _gen_combined_separator_cases(directive: str, clauses: list[str]) -> list[str]:
    if len(clauses) <= 1:
        return []

    a, b, = clauses[0], clauses[1]
    return [f"{directive} {a}{sep}{b}" for sep in CLAUSE_SEPARATORS]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_combined_separator_cases("parallel for", PARALLEL_FOR_CLAUSES.all_examples()))
def test_different_clause_separators_parallel_for(source: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelFor)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.for_, tree.For)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_combined_separator_cases("parallel sections", PARALLEL_SECTIONS_CLAUSES.all_examples()))
def test_different_clause_separators_parallel_sections(source: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelSections)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.sections, tree.Sections)


#### TEST CHANGING THE CLAUSES' ORDER ##########################################


def _gen_ordering_cases(limit: int=10) -> list[tuple[str, type]]:
    cases = []
    for directive in DIRECTIVES:
        all_clauses = directive.clauses.all_examples()
        if len(all_clauses) <= 1:
            continue
        for i, perm in enumerate(itertools.permutations(all_clauses)):
            if i > limit:
                break
            cases.append((directive.name + " " + " ".join(perm), directive.cls))
    return cases

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,construct_cls", _gen_ordering_cases())
def test_clause_ordering(source: str, construct_cls: type) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, construct_cls)


#### COMBINED DIRECTIVES ####

def _gen_combined_ordering_cases(directive: str, clauses: list[str], limit: int=10) -> list[str]:
    if len(clauses) <= 1:
        return []
    cases = []
    for i, perm in enumerate(itertools.permutations(clauses)):
        if i > limit:
            break
        cases.append(directive + " " + " ".join(perm))
    return cases

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_combined_ordering_cases("parallel for", PARALLEL_FOR_CLAUSES.all_examples()))
def test_clause_ordering_parallel_for(source: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelFor)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.for_, tree.For)

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_combined_ordering_cases("parallel sections", PARALLEL_SECTIONS_CLAUSES.all_examples()))
def test_clause_ordering_parallel_sections(source: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelSections)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.sections, tree.Sections)


#### TEST REPEATED CLAUSES #####################################################


def _gen_repeatable_cases() -> list[tuple[str, type, str]]:
    return [
        (f"{directive.name} {CLAUSES[clause_key]} {CLAUSES[clause_key]}", directive.cls, clause_key)
        for directive in DIRECTIVES
        for clause_key in directive.clauses.repeatable
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,construct_cls,field_name", _gen_repeatable_cases())
def test_repeatable_clauses(source: str, construct_cls: type, field_name: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, construct_cls)
    # verify both instances were collected, not just the last
    assert len(getattr(ast.construct, field_name)) == 2


#### COMBINED DIRECTIVES ####

def _gen_combined_repeatable_cases(
    directive: str,
    clause_info: ClauseInfo
) -> list[tuple[str, str]]:
    return [
        (f"{directive} {CLAUSES[clause_key]} {CLAUSES[clause_key]}", clause_key)
        for clause_key in clause_info.repeatable
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,field_name", _gen_combined_repeatable_cases("parallel for", PARALLEL_FOR_CLAUSES))
def test_repeatable_clauses_parallel_for(source: str, field_name: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelFor)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.for_, tree.For)

    # verify both instances were collected, not just the last
    if hasattr(ast.construct.parallel, field_name):
        assert len(getattr(ast.construct.parallel, field_name)) == 2
    if hasattr(ast.construct.for_, field_name):
        assert len(getattr(ast.construct.for_, field_name)) == 2


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,field_name", _gen_combined_repeatable_cases("parallel sections", PARALLEL_SECTIONS_CLAUSES))
def test_repeatable_clauses_parallel_sections(source: str, field_name: str) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ParallelSections)
    assert isinstance(ast.construct.parallel, tree.Parallel)
    assert isinstance(ast.construct.sections, tree.Sections)

    # verify both instances were collected, not just the last
    if hasattr(ast.construct.parallel, field_name):
        assert len(getattr(ast.construct.parallel, field_name)) == 2
    if hasattr(ast.construct.sections, field_name):
        assert len(getattr(ast.construct.sections, field_name)) == 2

#### TEST NON-REPETEABLE CLAUSES RAISES AN ERROR ###############################


def _gen_non_repeatable_cases() -> list[str]:
    return [
        f"{directive.name} {clause} {clause}"
        for directive in DIRECTIVES
        for clause in directive.clauses.optional_examples()
    ] + [
        f"parallel for {clause} {clause}"
        for clause in PARALLEL_FOR_CLAUSES.optional_examples()
    ] + [
        f"parallel sections {clause} {clause}"
        for clause in PARALLEL_SECTIONS_CLAUSES.optional_examples()
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_non_repeatable_cases())
def test_non_repeatable_clauses(source: str) -> None:
    with pytest.raises(SyntaxError):
        parse(source)


#### TEST INVALID CLAUSES FOR THIS DIRECTIVE ###################################


def _gen_wrong_directive_cases() -> list[str]:
    all_clauses = set(CLAUSES)
    return [
        f"{directive.name} {CLAUSES[clause]}"
        for directive in DIRECTIVES
        for clause in all_clauses - set(directive.clauses.all)
        if clause in CLAUSES
    ] + [
        f"parallel for {CLAUSES[clause]}"
        for clause in all_clauses - set(PARALLEL_FOR_CLAUSES.all)
    ] + [
        f"parallel sections {CLAUSES[clause]}"
        for clause in all_clauses - set(PARALLEL_SECTIONS_CLAUSES.all)
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_wrong_directive_cases())
def test_wrong_directive_clauses(source: str) -> None:
    with pytest.raises(SyntaxError):
        parse(source)


#### SPECIAL DIRECTIVES ########################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,expected_type", [
    ("atomic",         None),
    ("atomic read",    tree.Atomic.Type.READ),
    ("atomic write",   tree.Atomic.Type.WRITE),
    ("atomic update",  tree.Atomic.Type.UPDATE),
    ("atomic capture", tree.Atomic.Type.CAPTURE),
])
def test_atomic(source: str, expected_type: tree.Atomic.Type|None) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.Atomic)
    if expected_type is not None:
        assert ast.construct.type == expected_type


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,targets", [
    ("flush",                      []),
    ("flush(áÁÀæç)",               ["áÁÀæç"]),
    ("flush(flush, private, for)", ["flush", "private", "for"]),
])
def test_flush(source: str, targets: list[str]) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.Flush)
    assert ast.construct.str_targets == targets


@pytest.mark.no_isolate
@pytest.mark.parametrize("source,targets", [
    ("threadprivate(áÁÀæç)",               ["áÁÀæç"]),
    ("threadprivate(flush, private, for)", ["flush", "private", "for"]),
])
def test_threadprivate(source: str, targets: list[str]) -> None:
    ast = parse(source)
    assert isinstance(ast.construct, tree.ThreadPrivate)
    assert ast.construct.str_targets == targets

