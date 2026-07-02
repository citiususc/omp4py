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
    required: dict[str, type[tree.Clause]] = field(default_factory=dict)
    repeatable: dict[str, type[tree.Clause]] = field(default_factory=dict)
    optional: dict[str, type[tree.Clause]] = field(default_factory=dict)


    @staticmethod
    def from_type(cls: type[tree.Construct], exclude: set[str]|None=None) -> ClauseInfo:
        if exclude is None:
            exclude = {"name", "span", "id", "directive_name"}

        clause_info = ClauseInfo()
        for var_name, hint in typing.get_type_hints(cls).items():
            if var_name in exclude:
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

            elif original_type is None and isinstance(hint, type) and issubclass(hint, tree.Clause):
                clause_info.all[var_name]      = hint
                clause_info.required[var_name] = hint

        return clause_info

    @staticmethod
    def merge(info1: ClauseInfo, info2: ClauseInfo) -> ClauseInfo:
        return ClauseInfo(
            all        = {**info1.all,        **info2.all},
            required   = {**info1.required,   **info2.required},
            repeatable = {**info1.repeatable, **info2.repeatable},
            optional   = {**info1.optional,   **info2.optional},
        )

    def required_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.required]

    def repeatable_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.repeatable]

    def optional_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.optional]

    def all_examples(self) -> list[str]:
        return [CLAUSES[c] for c in self.all]


    def required_joined(self, sep: str=" ") -> str:
        return sep.join(self.required_examples())

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
    "combiner":      "combiner(omp_out = omp_out + omp_in)",
    "initializer":   "initializer(omp_priv = 0)",
    "collector":     "collector(omp_step * omp_idx)",
    "inductor":      "inductor(omp_var = omp_var + omp_step)",

    # metadirective
    "when":          "when(device={arch('nvptx')}: for private(abc))",
    "otherwise":     "otherwise(for)",

    # scan
    "exclusive":     "exclusive(x)",
    "inclusive":     "inclusive(x)",
    "init_complete": "init_complete(True)",

    # groupprivate / declare_target
    "device_type":   "device_type(host)",
    "enter":         "enter(x)",
    "indirect":      "indirect(True)",
    "link":          "link(x)",
    "local":         "local(x)",

    # allocate
    "align":         "align(8)",
    "allocator":     "allocator(omp_default_mem_alloc)",

    # dispatch / declare_variant
    "interop":         "interop(x)",
    "is_device_ptr":   "is_device_ptr(x)",
    "has_device_addr": "has_device_addr(x)",
    "no_context":      "nocontext(True)",
    "no_variants":     "novariants(True)",
    "adjust_args":     "adjust_args(nothing: x)",
    "append_args":     "append_args(interop(target))",
    "match":           "match(device={arch('nvptx')})",

    # declare_simd
    "aligned":        "aligned(x)",
    "linear":         "linear(x)",
    "simdlen":        "simdlen(8)",
    "uniform":        "uniform(x)",
    "in_branch":      "inbranch(True)",
    "not_in_branch":  "notinbranch(True)",

    # requires
    "atomic_default_mem_order": "atomic_default_mem_order(seq_cst)",
    "dynamic_allocators":       "dynamic_allocators",
    "reverse_offload":          "reverse_offload",
    "unified_address":          "unified_address",
    "unified_shared_memory":    "unified_shared_memory",
    "self_maps":                "self_maps",
    "device_safesync":          "device_safesync",

    # assume  (note: field name is misspelled in tree.py itself)
    "absent":                "absent(parallel)",
    "contains":              "contains(parallel)",
    "holds":                 "holds(True)",
    "no_openmp":              "no_openmp",
    "no_openmop_contructs":   "no_openmp_constructs",
    "no_openmp_routines":     "no_openmp_routines",
    "no_parallelism":         "no_parallelism",

    # error
    "at":            "at(compilation)",
    "message":       'message("msg")',
    "severity":      "severity(warning)",

    # fuse / interchange / split / stripe / tile / unroll
    "apply":         "apply(unroll)",
    "looprange":     "looprange(1, 2)",
    "permutation":   "permutation(1, 2)",
    "counts":        "counts(1, 2)",
    "sizes":         "sizes(4)",
    "full":          "full",
    "partial":       "partial",

    # parallel
    "allocate":      "allocate(x)",
    "proc_bind":     "proc_bind(spread)",
    "safesync":      "safesync(4)",

    # teams
    "num_teams":     "num_teams(4)",
    "thread_limit":  "thread_limit(4)",

    # simd / for / distribute
    "non_temporal":  "nontemporal(x)",
    "order":         "order(concurrent)",
    "safelen":       "safelen(4)",
    "induction":     "induction(+:i)",
    "dist_schedule": "dist_schedule(static)",

    # masked
    "filter":        "filter(0)",

    # loop
    "bind":          "bind(thread)",

    # taskloop
    "grain_size":    "grain_size(4)",
    "num_tasks":     "num_tasks(4)",
    "no_group":      "nogroup",

    # taskgraph
    "graph_id":      "graph_id(1)",
    "graph_reset":   "graph_reset(True)",

    # target_data / target
    "use_device_ptr":  "use_device_ptr(x)",
    "use_device_addr": "use_device_addr(x)",
    "default_map":     "defaultmap(none)",
    "uses_allocators": "uses_allocators(omp_default_mem_alloc)",
    "map":             "map(to: x)",
    "device":          "device(0)",
    "depobj_update":   "update(x)",
    "do_across":       "doacross(sink: i = 0:10)",

    # target_update
    "from_":         "from(x)",
    "to":            "to(x)",

    # interop
    "destroy":       "destroy(x)",
    "init":          "init(x)",
    "use":           "use(x)",
    "interop":       "interop(x)",

    # critical / atomic / flush
    "hint":          "hint(0)",
    "mem_scope":     "memscope(device)",
    "read":          "read",
    "update":        "update",
    "write":         "write",
    "capture":       "capture",
    "compare":       "compare",
    "fail":          "fail(seq_cst)",
    "weak":          "weak",
    "acq_rel":       "acq_rel",
    "acquire":       "acquire",
    "relaxed":       "relaxed",
    "release":       "release",
    "seq_cst":       "seq_cst",

    # taskgroup
    "task_reduction": "task_reduction(+:x)",

    # task
    "affinity":      "affinity(x)",
    "depend":        "depend(in: x)",
    "detach":        "detach(x)",
    "in_reduction":  "in_reduction(+:x)",
    "priority":      "priority(1)",
    "replayable":    "replayable(True)",
    "thread_set":    "thread_set(omp_pool)",
    "transparent":   "transparent(True)",

    # ordered
    "simd":          "simd",
    "threads":       "threads",
}

# TODO: the commented directives require clauses of a group.
# This is enforced by the grammar (clause list cannot be empty)
# but cannot be represented in the dataclasses,
# because the type must be optional to handle the posibilities.
# These tests read from the classes, so they are not aware of this restrictions.
DIRECTIVES = [
    DirectiveSpec("threadprivate(x)", tree.ThreadPrivate),
    #DirectiveSpec("declare_reduction(+:int)", tree.DeclareReduction),
    #DirectiveSpec("declare_induction(+:int)", tree.DeclareInduction),
    #DirectiveSpec("scan", tree.Scan),
    #DirectiveSpec("declare_mapper(m: v: int)", tree.DeclareMapper),
    DirectiveSpec("groupprivate", tree.GroupPrivate),
    DirectiveSpec("allocate(x)", tree.Allocate),
    DirectiveSpec("metadirective", tree.Metadirective),
    DirectiveSpec("declare_variant(base:variant)", tree.DeclareVariant),
    DirectiveSpec("dispatch", tree.Dispatch),
    DirectiveSpec("declare_simd", tree.DeclareSimd),
    DirectiveSpec("declare_target", tree.DeclareTarget),
    #DirectiveSpec("requires", tree.Requires),
    #DirectiveSpec("assume", tree.Assume),
    DirectiveSpec("nothing", tree.Nothing),
    DirectiveSpec("error", tree.Error),
    #DirectiveSpec("fuse", tree.Fuse),
    DirectiveSpec("interchange", tree.Interchange),
    DirectiveSpec("reverse", tree.Reverse),
    #DirectiveSpec("stripe", tree.Stripe),
    #DirectiveSpec("tile", tree.Tile),
    #DirectiveSpec("split", tree.Split),
    DirectiveSpec("unroll", tree.Unroll),
    DirectiveSpec("parallel", tree.Parallel),
    DirectiveSpec("teams", tree.Teams),
    DirectiveSpec("simd", tree.Simd),
    DirectiveSpec("masked", tree.Masked),
    DirectiveSpec("single", tree.Single),
    DirectiveSpec("scope", tree.Scope),
    DirectiveSpec("sections", tree.Sections),
    DirectiveSpec("section", tree.Section),
    DirectiveSpec("workshare", tree.Workshare),
    DirectiveSpec("workdistribute", tree.Workdistribute),
    DirectiveSpec("for", tree.For),
    DirectiveSpec("distribute", tree.Distribute),
    DirectiveSpec("loop", tree.Loop),
    DirectiveSpec("task", tree.Task),
    DirectiveSpec("taskloop", tree.Taskloop),
    #DirectiveSpec("task_iteration", tree.TaskIteration),
    DirectiveSpec("taskyield", tree.Taskyield),
    DirectiveSpec("taskgraph", tree.Taskgraph),
    #DirectiveSpec("target_data", tree.TargetData),
    DirectiveSpec("target_enter_data", tree.TargetEnterData),
    DirectiveSpec("target_exit_data", tree.TargetExitData),
    DirectiveSpec("target", tree.Target),
    DirectiveSpec("target_update", tree.TargetUpdate),
    #DirectiveSpec("interop", tree.InteropConstruct),
    DirectiveSpec("critical", tree.Critical),
    DirectiveSpec("barrier", tree.Barrier),
    DirectiveSpec("taskgroup", tree.Taskgroup),
    DirectiveSpec("taskwait", tree.Taskwait),
    #DirectiveSpec("atomic", tree.Atomic),
    #DirectiveSpec("flush", tree.Flush),
    #DirectiveSpec("depobj(x) destroy(x)", tree.Depobj),
    DirectiveSpec("ordered", tree.Ordered),
    DirectiveSpec("cancel parallel", tree.Cancel),
    DirectiveSpec("cancellation_point parallel", tree.CancellationPoint),
]


#### EMPTY DIRECTIVES ##########################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_empty_directive(directive: DirectiveSpec) -> None:
    source = directive.name + " " + directive.clauses.required_joined()
    ast = parse(directive.name)
    assert isinstance(ast.constructs[directive.cls.id], directive.cls)


#### DIRECTIVE WITH ALL CLAUSES ################################################


@pytest.mark.no_isolate
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_all_clauses(directive: DirectiveSpec) -> None:
    source = directive.name + " " + directive.clauses.all_joined()
    ast = parse(source)
    assert isinstance(ast.constructs[directive.cls.id], directive.cls)


#### TEST CLAUSES WITH DIFFERENT SEPARATORS ####################################


def _gen_separator_cases() -> list[tuple[str, type]]:
    cases = []
    for directive in DIRECTIVES:
        clauses = directive.clauses.required_examples()

        # Use only the first 2
        if len(clauses) < 2:
            clauses_set = set(clauses).union(set(directive.clauses.all_examples()))
            if len(clauses_set) < 2:
                continue
            clauses = list(clauses_set)

        a, b = clauses[0], clauses[1]
        cases.extend((f"{directive.name} {a}{sep}{b}", directive.cls) for sep in CLAUSE_SEPARATORS)

    return cases

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,construct_cls", _gen_separator_cases())
def test_different_clause_separators(source: str, construct_cls: type[tree.Construct]) -> None:
    ast = parse(source)
    assert isinstance(ast.constructs[construct_cls.id], construct_cls)


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
    assert isinstance(ast.constructs[construct_cls.id], construct_cls)


#### TEST REPEATED CLAUSES #####################################################


def _gen_repeatable_cases(limit: int = 10) -> list[tuple[str, type, str]]:
    return [
        (f"{directive.name} {directive.clauses.required_joined()} {CLAUSES[clause_key]} {CLAUSES[clause_key]}", directive.cls, clause_key)
        for directive in DIRECTIVES
        for i, clause_key in enumerate(directive.clauses.repeatable)
        if i < limit
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source,construct_cls,field_name", _gen_repeatable_cases())
def test_repeatable_clauses(source: str, construct_cls: type, field_name: str) -> None:
    ast = parse(source)
    assert isinstance(ast.constructs[construct_cls.id], construct_cls)
    # verify both instances were collected, not just the last
    assert len(getattr(ast.constructs[construct_cls.id], field_name)) == 2


#### TEST NON-REPETEABLE CLAUSES RAISES AN ERROR ###############################


def _gen_non_repeatable_cases(limit: int = 10) -> list[str]:
    return [
        f"{directive.name} {directive.clauses.required_joined()} {clause} {clause}"
        for directive in DIRECTIVES
        for i, clause in enumerate(directive.clauses.optional_examples())
        if i < limit
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_non_repeatable_cases())
def test_non_repeatable_clauses(source: str) -> None:
    with pytest.raises(SyntaxError):
        parse(source)


#### TEST INVALID CLAUSES FOR THIS DIRECTIVE ###################################


def _gen_wrong_directive_cases(limit: int = 10) -> list[str]:
    all_clauses = set(CLAUSES)
    return [
        f"{directive.name} {CLAUSES[clause]}"
        for directive in DIRECTIVES
        for i, clause in enumerate(all_clauses - set(directive.clauses.all))
        if clause in CLAUSES and i < limit
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("source", _gen_wrong_directive_cases())
def test_wrong_directive_clauses(source: str) -> None:
    with pytest.raises(SyntaxError):
        parse(source)


#### EMPTY DIRECTIVES ##########################################################

COMBINED_DIRECTIVES = [
    DirectiveSpec("parallel", tree.Parallel),
    DirectiveSpec("teams", tree.Teams),
    DirectiveSpec("simd", tree.Simd),
    DirectiveSpec("masked", tree.Masked),
    DirectiveSpec("single", tree.Single),
    DirectiveSpec("sections", tree.Single),
    DirectiveSpec("workshare", tree.Workshare),
    DirectiveSpec("workdistribute", tree.Workdistribute),
    DirectiveSpec("for", tree.For),
    DirectiveSpec("distribute", tree.Distribute),
    DirectiveSpec("loop", tree.Loop),
    DirectiveSpec("task", tree.Task),
    DirectiveSpec("target_data", tree.TargetData),
    DirectiveSpec("target_enter_data", tree.TargetEnterData),
    DirectiveSpec("target_exit_data", tree.TargetExitData),
    DirectiveSpec("target", tree.Target),
    DirectiveSpec("target_update", tree.TargetUpdate),
]

def _gen_combined_directives(limit: int = 10) -> list[str]:
    return [
        f"{d1.name} {d2.name} {' '.join(set(d1.clauses.required_examples() + d2.clauses.required_examples()))}"
        for i, (d1, d2) in enumerate(itertools.combinations(COMBINED_DIRECTIVES, 2))
        if i < limit
    ]

@pytest.mark.no_isolate
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_combined_constructs(directive: DirectiveSpec) -> None:
    source = directive.name + " " + directive.clauses.required_joined()
    ast = parse(directive.name)
    assert isinstance(ast.constructs[directive.cls.id], directive.cls)

