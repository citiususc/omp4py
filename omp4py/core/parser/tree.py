"""Parser nodes for directives, clauses, and expressions.

An OpenMP directive consists of a construct and zero or more clauses. Each
keyword within the directive represents either the construct or a clause,
and together they form the directive's elements. Both constructs and clauses
can include modifiers, which are arguments that further define their behavior.
"""

from __future__ import annotations

import typing
from dataclasses import KW_ONLY, InitVar, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ast import alias, arg, expr, keyword, pattern, stmt, type_param


__all__ = [
    "AffinityPolicy",
    "Aligned",
    "Atomic",
    "Barrier",
    "Cancel",
    "CancellationPoint",
    "Capture",
    "Clause",
    "Collapse",
    "Combiner",
    "Construct",
    "ConstructType",
    "CopyIn",
    "CopyPrivate",
    "Critical",
    "DataScope",
    "DeclareReduction",
    "DeclareSimd",
    "DeclareTarget",
    "Default",
    "DefaultData",
    "DefaultMap",
    "Depend",
    "DependenceType",
    "Device",
    "Directive",
    "DirectiveName",
    "DistSchedule",
    "Distribute",
    "Filter",
    "Final",
    "FirstPrivate",
    "Flush",
    "For",
    "From",
    "GrainSize",
    "Hint",
    "If",
    "InBranch",
    "Initializer",
    "IsDevicePtr",
    "LastPrivate",
    "Linear",
    "LinearModifier",
    "Link",
    "Map",
    "MapModifier",
    "MapType",
    "Masked",
    "Master",
    "Mergeable",
    "Modifier",
    "Name",
    "NoGroup",
    "NoWait",
    "NotInBranch",
    "NumTasks",
    "NumTeams",
    "NumThreads",
    "OmpNode",
    "Ordered",
    "OrderedClause",
    "Parallel",
    "Priority",
    "Private",
    "ProcBind",
    "PyCode",
    "PyExpr",
    "PyInt",
    "PyName",
    "PySection",
    "PyStmt",
    "Read",
    "Reduction",
    "ReductionOp",
    "SafeLen",
    "Schedule",
    "ScheduleChunk",
    "ScheduleOrdering",
    "ScheduleType",
    "Section",
    "Sections",
    "SeqCst",
    "Shared",
    "Simd",
    "SimdClause",
    "Simdlen",
    "Single",
    "Span",
    "Target",
    "TargetData",
    "TargetEnterData",
    "TargetExitData",
    "TargetUpdate",
    "Task",
    "Taskgroup",
    "Taskloop",
    "Taskwait",
    "Taskyield",
    "Teams",
    "ThreadLimit",
    "ThreadPrivate",
    "Threads",
    "To",
    "Uniform",
    "Untied",
    "Update",
    "UseDevicePtr",
    "Write",
]


#######################################################################################################################
######################################################## Base #########################################################
#######################################################################################################################


@dataclass
class Span:
    """Represent a source code range using line and column positions.

    Line numbers are 1-based and column offsets are 0-based, matching
    Python AST location conventions.

    Attributes:
        lineno (int): Starting line number.
        offset (int): Starting column offset.
        end_lineno (int): Ending line number, or -1 if unavailable.
        end_offset (int): Ending column offset, or -1 if unavailable.
    """

    lineno: int
    offset: int
    end_lineno: int
    end_offset: int

    @staticmethod
    def from_ast(node: expr | stmt | arg | keyword | alias | pattern | type_param) -> Span:
        """Create a span from a Python AST node.

        Args:
            node (expr | stmt | arg | keyword | alias | pattern | type_param):
                AST node containing source location information.

        Returns:
            Span: Source range of the node. Missing end positions are set to -1.
        """
        return Span(
            node.lineno,
            node.col_offset,
            node.end_lineno if node.end_lineno is not None else -1,
            node.end_col_offset if node.end_col_offset is not None else -1,
        )

    def to_ast[T: expr | stmt | arg | keyword | alias | pattern | type_param](self, node: T) -> T:
        """Assign this span's source positions to a Python AST node.

        Args:
            node (T): AST node to update.

        Returns:
            T: The same node with its source location attributes updated.
        """
        node.lineno = self.lineno
        node.col_offset = self.offset
        node.end_lineno = self.end_lineno
        node.end_col_offset = self.end_offset
        return node


@dataclass
class OmpNode:
    """Represent a node in the OpenMP directive tree.

    Attributes:
        span (Span): Source range associated with the node.
    """

    span: Span

    @classmethod
    def id(cls) -> str:
        """Return the node name as a lowercase identifier.

        Returns:
            str: Lowercase identifier.
        """
        return cls.cid()

    @typing.final
    @classmethod
    def cid(cls) -> str:
        """Return the node name as a lowercase identifier.

        Returns:
            str: Lowercase identifier derived from the node class name.
        """
        return "".join(f"_{c}" if c.isupper() and i > 0 else c for i, c in enumerate(cls.__name__)).lower()


@dataclass
class Name(OmpNode):
    """Represent a name in an OpenMP directive.

    Attributes:
        string (str): Name text as it appears in the source code.
    """

    string: str

    def __str__(self) -> str:
        """Return the name text.

        Returns:
            str: The name's string representation.
        """
        return self.string


@dataclass
class Directive(OmpNode):
    """Represent a parsed OpenMP directive.

    Attributes:
        string (str): Original directive text passed to the parser.
        constructs (list[Construct]): Constructs forming the combined construct,
            ordered as they appear in the directive.
    """

    string: str
    constructs: list[Construct]


@dataclass
class Construct(OmpNode):
    """Represent an OpenMP construct.

    Attributes:
        name (Name): Name identifying the construct.
    """

    name: Name


@dataclass
class Clause(OmpNode):
    """Represent a clause associated with an OpenMP directive.

    Attributes:
        name (Name): Name identifying the clause.
        directive_name (DirectiveName | None): Directive name associated with
            the clause.
    """

    name: Name
    _: KW_ONLY
    directive_name: DirectiveName | None = field(default=None)


@dataclass
class DataScope(Clause):
    """Represent a data scope clause containing Python name references.

    Attributes:
        targets (list[PyName]): Python names referenced by the clause.
    """

    targets: list[PyName]

    @property
    def str_targets(self) -> list[str]:
        """Return the target names as strings.

        Returns:
            list[str]: String representations of the target names.
        """
        return [v.string for v in self.targets]


@dataclass
class Modifier(OmpNode):
    """Represent a modifier associated with an OpenMP directive or clause."""


@dataclass
class DirectiveName(Modifier):
    """Represent the name of an OpenMP directive.

    Attributes:
        value (Name): Directive name as represented in the source code.
    """

    value: Name

    def __str__(self) -> str:
        """Return the directive name.

        Returns:
            str: The directive name as a string.
        """
        return self.value.string


#######################################################################################################################
####################################################### Python ########################################################
#######################################################################################################################


@dataclass
class PyExpr(OmpNode):
    """Represent a Python expression in the AST.

    Attributes:
        value (expr): The AST node representing the expression.
    """

    value: expr


@dataclass
class PyInt(OmpNode):
    """Represent a Python integer.

    Attributes:
        value (int): A Python integer.

    """

    value: int

    def __int__(self) -> int:
        """Return the Python integer.

        Returns:
            int: The Python integer as an int.
        """
        return self.value


@dataclass
class PyName(Name):
    """Represent a Python name in an OpenMP directive.

    Attributes:
        sections (list[PyArraySection] | None): Optional array section.
    """

    sections: list[PySection] | None = None


@dataclass
class PySection(OmpNode):
    """Represent an array section.

    Attributes:
        lower (PyExpr | None): Lower bound of the section.
        length (PyExpr | None): Length of the section.
    """

    lower: PyExpr | None = None
    length: PyExpr | None = None


@dataclass
class PyStmt(OmpNode):
    """Represent a Python statement in the AST.

    Attributes:
        value (stmt): The AST node representing the statement.
    """

    value: stmt

@dataclass
class PyCode(OmpNode):
    """Represent a Python code in the AST.

    Attributes:
        value (list[stmt]): The list AST node representing the code statements.
    """

    value: list[stmt]



#######################################################################################################################
##################################################### Directives ######################################################
#######################################################################################################################


#### Parallel ####
@dataclass
class Parallel(Construct):
    """Represent an OpenMP parallel construct.

    Attributes:
        if_ (If | None): If clause.
        num_threads (NumThreads | None): NumThreads clause.
        default (Default | None): Default clause.
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        shared (list[Shared]): Shared clauses.
        copyin (list[CopyIn]): Copyin clauses.
        reduction (list[Reduction]): Reduction clauses.
        proc_bind (ProcBind | None): ProcBind clause.
    """

    if_: If | None = None
    num_threads: NumThreads | None = None
    default: Default | None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    copyin: list[CopyIn] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    proc_bind: ProcBind | None = None


@dataclass
class NumThreads(Clause):
    """Represent a num_threads clause.

    Attributes:
        nthreads (PyExpr): Expression specifying the number of threads.
    """

    nthreads: PyExpr


@dataclass
class ProcBind(Clause):
    """Represent a proc_bind clause.

    Attributes:
        affinity_policy (AffinityPolicy): Thread affinity policy.
    """

    affinity_policy: AffinityPolicy


@dataclass
class AffinityPolicy(OmpNode):
    """Represent a thread affinity policy.

    Attributes:
        choice (Choice): Selected thread affinity policy.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported thread affinity policies."""

        CLOSE = "close"
        PRIMARY = "primary"
        SPREAD = "spread"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the affinity policy from its string representation.

        Args:
            raw_choice (str): Affinity policy name.
        """
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class CopyIn(DataScope):
    """Represent a copyin clause."""


#### For ####
@dataclass
class For(Construct):
    """Represent an OpenMP for construct.

    Attributes:
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        last_private (list[LastPrivate]): Lastprivate clauses.
        linear (list[Linear]): Linear clauses.
        reduction (list[Reduction]): Reduction clauses.
        schedule (Schedule | None): Schedule clause.
        collapse (Collapse | None): Collapse clause.
        ordered (OrderedClause | None): Ordered clause.
        no_wait (NoWait | None): Nowait clause.
    """

    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    schedule: Schedule | None = None
    collapse: Collapse | None = None
    ordered: OrderedClause | None = None
    no_wait: NoWait | None = None


@dataclass
class Schedule(Clause):
    """Represent a schedule clause.

    Attributes:
        type (Kind): Scheduling policy type.
        schedule_ordering (ScheduleOrdering): Schedule ordering modifier.
        schedule_chunk (ScheduleChunk): Schedule chunk modifier.
    """

    type: ScheduleType
    chunk_size: PyExpr | None = None
    schedule_ordering: ScheduleOrdering | None = None
    schedule_chunk: ScheduleChunk | None = None


@dataclass
class ScheduleType(OmpNode):
    """Represent a scheduling policy type.

    Attributes:
        choice (Choice): Scheduling policy value.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported scheduling policies."""

        STATIC = "static"
        DYNAMIC = "dynamic"
        GUIDED = "guided"
        AUTO = "auto"
        RUNTIME = "runtime"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the schedule type from its string representation.

        Args:
            raw_choice (str): Schedule string type.
        """
        self.choice = self.Choice(raw_choice)


@dataclass
class ScheduleOrdering(Modifier):
    """Represent a schedule ordering modifier.

    Attributes:
        choice (Choice): Selected schedule modifier.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported schedule modifiers."""

        MONOTONIC = "monotonic"
        NON_MONOTONIC = "nonmonotonic"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the modifier from its string representation.

        Args:
            raw_choice (str): Modifier string value.
        """
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class ScheduleChunk(Modifier):
    """Represent a schedule chunk modifier.

    Attributes:
        choice (Choice): Selected schedule chunk modifier.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported schedule modifiers."""

        SIMD = "simd"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the modifier from its string representation.

        Args:
            raw_choice (str): Modifier string value.
        """
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class OrderedClause(Clause):
    """Represent an ordered clause.

    Attributes:
        n (PyInt | None): Optional parameter specifying the number of loops.
    """

    n: PyInt | None = None


#### Sections ####
@dataclass
class Sections(Construct):
    """Represent an OpenMP sections construct.

    Attributes:
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        last_private (list[LastPrivate]): Lastprivate clauses.
        reduction (list[Reduction]): Reduction clauses.
        no_wait (NoWait | None): Nowait clause.
    """

    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    no_wait: NoWait | None = None


#### Section ####
@dataclass
class Section(Construct):
    """Represent an OpenMP section construct."""


#### Single ####
@dataclass
class Single(Construct):
    """Represent an OpenMP single construct.

    Attributes:
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        copyprivate (list[CopyPrivate]): Copyprivate clauses.
        no_wait (NoWait | None): Nowait clause.
    """

    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    copyprivate: list[CopyPrivate] = field(default_factory=list)
    no_wait: NoWait | None = None


@dataclass
class CopyPrivate(DataScope):
    """Represent a copyprivate clause."""


#### Simd and DeclareSimd ####
@dataclass
class Simd(Construct):
    """Represent an OpenMP simd construct.

    Attributes:
        safelen (SafeLen | None): Safelen clause.
        simdlen (Simdlen | None): Simdlen clause.
        linear (list[Linear]): Linear clauses.
        aligned (list[Aligned]): Aligned clauses.
        private (list[Private]): Private clauses.
        last_private (list[LastPrivate]): Lastprivate clauses.
        reduction (list[Reduction]): Reduction clauses.
        collapse (Collapse | None): Collapse clause.
    """

    safelen: SafeLen | None = None
    simdlen: Simdlen | None = None
    linear: list[Linear] = field(default_factory=list)
    aligned: list[Aligned] = field(default_factory=list)
    private: list[Private] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    collapse: Collapse | None = None


@dataclass
class DeclareSimd(Construct):
    """Represent an OpenMP declare simd construct.

    Attributes:
        simdlen (Simdlen | None): Simdlen clause.
        linear (list[Linear]): Linear clauses.
        aligned (list[Aligned]): Aligned clauses.
        uniform (list[Uniform]): Uniform clauses.
        in_branch (InBranch | None): Inbranch clause.
        not_in_branch (NotInBranch | None): Notinbranch clause.
    """

    simdlen: Simdlen | None = None
    linear: list[Linear] = field(default_factory=list)
    aligned: list[Aligned] = field(default_factory=list)
    uniform: list[Uniform] = field(default_factory=list)
    in_branch: InBranch | None = None
    not_in_branch: NotInBranch | None = None


@dataclass
class SafeLen(Clause):
    """Represent an OpenMP safelen clause.

    Attributes:
        length (PyExpr): Maximum SIMD lane count.
    """

    length: PyExpr


@dataclass
class Simdlen(Clause):
    """Represent an OpenMP simdlen clause.

    Attributes:
        length (PyExpr): Preferred SIMD lane count.
    """

    length: PyExpr


@dataclass
class Linear(DataScope):
    """Represent an OpenMP linear clause."""

    linear_modifier: LinearModifier | None = None

    # TODO: how to represent REF/UVAL/VAL here?
@dataclass
class LinearModifier(Modifier):
    """Represent an OpenMP linear modifier.

    Attributes:
        choice (Choice): Linear modifier.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported linear modifiers."""

        REF = "ref"
        UVAL = "uval"
        VAL = "val"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the linear modifier from its string value."""
        self.choice = self.Choice(raw_choice.lower())

@dataclass
class Aligned(DataScope):
    """Represent an OpenMP aligned clause."""

    # TODO: how to represent REF/UVAL/VAL here?


@dataclass
class Uniform(DataScope):
    """Represent an OpenMP uniform clause."""

    # TODO: how to represent REF/UVAL/VAL here?


@dataclass
class InBranch(Clause):
    """Represent an OpenMP inbranch clause."""


@dataclass
class NotInBranch(Clause):
    """Represent an OpenMP notinbranch clause."""


#### Task and Taskloop ####
@dataclass
class Task(Construct):
    """Represent an OpenMP task construct.

    Attributes:
        if_ (If | None): If clause.
        final (Final | None): Final clause.
        untied (Untied | None): Untied clause.
        default (Default | None): Default data-sharing clause.
        mergeable (Mergeable | None): Mergeable clause.
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        shared (list[Shared]): Shared clauses.
        depend (list[Depend]): Depend clauses.
        priority (Priority | None): Priority clause.
    """

    if_: If | None = None
    final: Final | None = None
    untied: Untied | None = None
    default: Default | None = None
    mergeable: Mergeable | None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    depend: list[Depend] = field(default_factory=list)
    priority: Priority | None = None


@dataclass
class Taskloop(Construct):
    """Represent an OpenMP taskloop construct.

    Attributes:
        if_ (If | None): If clause.
        shared (list[Shared]): Shared clauses.
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        last_private (list[LastPrivate]): Lastprivate clauses.
        default (Default | None): Default data-sharing clause.
        grain_size (GrainSize | None): Grainsize clause.
        num_tasks (NumTasks | None): Num_tasks clause.
        collapse (Collapse | None): Collapse clause.
        final (Final | None): Final clause.
        priority (Priority | None): Priority clause.
        untied (Untied | None): Untied clause.
        mergeable (Mergeable | None): Mergeable clause.
        no_group (NoGroup | None): Nogroup clause.
    """

    if_: If | None = None
    shared: list[Shared] = field(default_factory=list)
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    default: Default | None = None
    grain_size: GrainSize | None = None
    num_tasks: NumTasks | None = None
    collapse: Collapse | None = None
    final: Final | None = None
    priority: Priority | None = None
    untied: Untied | None = None
    mergeable: Mergeable | None = None
    no_group: NoGroup | None = None


@dataclass
class Final(Clause):
    """Represent an OpenMP final clause.

    Attributes:
        finalize (PyExpr): Expression determining whether a task is final.
    """

    finalize: PyExpr


@dataclass
class Untied(Clause):
    """Represent an OpenMP untied clause.

    Attributes:
        can_change_threads (PyExpr | None): Expression controlling whether a task can resume on a different thread.
    """

    can_change_threads: PyExpr | None = None


@dataclass
class Mergeable(Clause):
    """Represent an OpenMP mergeable clause.

    Attributes:
        can_merge (PyExpr | None): Expression controlling whether a task can be merged.
    """

    can_merge: PyExpr | None = None


@dataclass
class Priority(Clause):
    """Represent an OpenMP priority clause.

    Attributes:
        value (PyExpr): Expression specifying the task priority.
    """

    value: PyExpr


@dataclass
class GrainSize(Clause):
    """Represent an OpenMP grainsize clause.

    Attributes:
        grain_size (PyExpr): Expression specifying the task grain size.
    """

    grain_size: PyExpr


@dataclass
class NumTasks(Clause):
    """Represent an OpenMP num_tasks clause.

    Attributes:
        num_tasks (PyExpr): Expression specifying the number of tasks.
    """

    num_tasks: PyExpr


@dataclass
class NoGroup(Clause):
    """Represent an OpenMP nogroup clause.

    Attributes:
        not_synchronize (PyExpr | None): Expression controlling taskgroup synchronization.
    """

    not_synchronize: PyExpr | None = None


#### Taskyield ####
@dataclass
class Taskyield(Construct):
    """Represent an OpenMP taskyield construct."""


#### TargetData ####
@dataclass
class TargetData(Construct):
    """Represent an OpenMP target data construct.

    Attributes:
        if_ (If | None): If clause.
        device (Device | None): Device clause.
        map (Map | None): Map clause.
        use_device_ptr (list[UseDevicePtr]): Use_device_ptr clauses.
    """

    if_: If | None = None
    device: Device | None = None
    map: Map | None = None
    use_device_ptr: list[UseDevicePtr] = field(default_factory=list)


@dataclass
class UseDevicePtr(DataScope):
    """Represent an OpenMP use_device_ptr clause."""

    # TODO: list


#### TargetEnterData ####
@dataclass
class TargetEnterData(Construct):
    """Represent an OpenMP target enter data construct.

    Attributes:
        if_ (If | None): If clause.
        device (Device | None): Device clause.
        map (Map | None): Map clause.
        depend (list[Depend]): Depend clauses.
        no_wait (NoWait | None): Nowait clause.
    """

    if_: If | None = None
    device: Device | None = None
    map: Map | None = None
    depend: list[Depend] = field(default_factory=list)
    no_wait: NoWait | None = None


#### TargetExitData ####
@dataclass
class TargetExitData(Construct):
    """Represent an OpenMP target exit data construct.

    Attributes:
        if_ (If | None): If clause.
        device (Device | None): Device clause.
        map (Map | None): Map clause.
        depend (list[Depend]): Depend clauses.
        no_wait (NoWait | None): Nowait clause.
    """

    if_: If | None = None
    device: Device | None = None
    map: Map | None = None
    depend: list[Depend] = field(default_factory=list)
    no_wait: NoWait | None = None


#### Target ####
@dataclass
class Target(Construct):
    """Represent an OpenMP target construct.

    Attributes:
        if_ (If | None): If clause.
        device (Device | None): Device clause.
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        map (Map | None): Map clause.
        is_device_ptr (list[IsDevicePtr]): Is_device_ptr clauses.
        default_map (DefaultMap | None): Defaultmap clause.
        no_wait (NoWait | None): Nowait clause.
        depend (list[Depend]): Depend clauses.
    """

    if_: If | None = None
    device: Device | None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    map: Map | None = None
    is_device_ptr: list[IsDevicePtr] = field(default_factory=list)
    default_map: DefaultMap | None = None
    no_wait: NoWait | None = None
    depend: list[Depend] = field(default_factory=list)


@dataclass
class IsDevicePtr(DataScope):
    """Represent an OpenMP is_device_ptr clause."""

    # TODO: list


@dataclass
class DefaultMap(Clause):
    """Represent an OpenMP defaultmap clause.

    Attributes:
        default_behavior (Name): Data-mapping behavior for transfer to and from the device.
        variable_category (Name): Category of variables affected.
    """

    default_behavior: DefaultBehavior
    variable_category: VariableCategory | None = None


@dataclass
class DefaultBehavior(OmpNode):
    """Represent the behavior for transfer to and from the device.

    Attributes:
        choice (Choice): Selected behavior for transfer.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported transfer behaviors."""

        SIMD = "tofrom"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the transfer behavior from its string representation.

        Args:
            raw_choice (str): Behavior string value.
        """
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class VariableCategory(Modifier):
    """Represent the category of variables affected by data-mapping behavior.

    Attributes:
        choice (Choice): Selected category for data-mapping.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported categories."""

        SIMD = "scalar"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the category of variable from its string representation.

        Args:
            raw_choice (str): Category string value.
        """
        self.choice = self.Choice(raw_choice.lower())


#### TargetUpdate and DeclareTarget ####
@dataclass
class TargetUpdate(Construct):
    """Represent an OpenMP target update construct.

    Attributes:
        if_ (If | None): If clause.
        device (Device | None): Device clause.
        no_wait (NoWait | None): Nowait clause.
        depend (list[Depend]): Depend clauses.
        to (list[To]): To clauses.
        from_ (list[From]): From clauses.
    """

    if_: If | None = None
    device: Device | None = None
    no_wait: NoWait | None = None
    depend: list[Depend] = field(default_factory=list)
    to: list[To] = field(default_factory=list)
    from_: list[From] = field(default_factory=list)


@dataclass
class DeclareTarget(Construct):
    """Represent an OpenMP declare target construct.

    Attributes:
        targets (list[PyName] | None): Target names.
        link (list[Link]): Link clauses.
        to (list[To]): To clauses.
    """

    targets: list[PyName] | None = None
    link: list[Link] = field(default_factory=list)
    to: list[To] = field(default_factory=list)

    @property
    def str_targets(self) -> list[str]:
        """Return target names as strings.

        Returns:
            list[str]: Target names.
        """
        return [v.string for v in self.targets or []]


@dataclass
class To(DataScope):
    """Represent an OpenMP to clause."""


@dataclass
class From(DataScope):
    """Represent an OpenMP from clause."""

    @classmethod
    def id(cls) -> str:
        """Return the clause identifier.

        Returns:
            str: The identifier ``from_``.
        """
        return "from_"


@dataclass
class Link(DataScope):
    """Represent an OpenMP link clause."""


#### Teams ####
@dataclass
class Teams(Construct):
    """Represent an OpenMP teams construct.

    Attributes:
        num_teams (NumTeams | None): Num_teams clause.
        thread_limit (ThreadLimit | None): Thread_limit clause.
        default (Default | None): Default data-sharing clause.
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        shared (list[Shared]): Shared clauses.
        reduction (list[Reduction]): Reduction clauses.
    """

    num_teams: NumTeams | None = None
    thread_limit: ThreadLimit | None = None
    default: Default | None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)


@dataclass
class NumTeams(Clause):
    """Represent an OpenMP num_teams clause.

    Attributes:
        upper_bound (PyExpr): Expression specifying the upper bound on the number of teams.
    """

    upper_bound: PyExpr


@dataclass
class ThreadLimit(Clause):
    """Represent an OpenMP thread_limit clause.

    Attributes:
        threads (PyExpr): Expression specifying the thread limit.
    """

    threads: PyExpr


#### Distribute ####
@dataclass
class Distribute(Construct):
    """Represent an OpenMP distribute construct.

    Attributes:
        private (list[Private]): Private clauses.
        first_private (list[FirstPrivate]): Firstprivate clauses.
        last_private (list[LastPrivate]): Lastprivate clauses.
        collapse (Collapse | None): Collapse clause.
        dist_schedule (DistSchedule | None): Dist_schedule clause.
    """

    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    collapse: Collapse | None = None
    dist_schedule: DistSchedule | None = None


@dataclass
class DistSchedule(Clause):
    """Represent an OpenMP dist_schedule clause.

    Attributes:
        type (DistScheduleType): Distribution schedule type.
        chunk_size (PyExpr | None): Expression specifying the chunk size.
    """

    type: DistScheduleType  # = static
    chunk_size: PyExpr | None = None


@dataclass
class DistScheduleType(OmpNode):
    """Represent a distributed scheduling policy type.

    Attributes:
        choice (Choice): Scheduling policy value.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported schedule policies."""

        STATIC = "static"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the schedule type from its string representation.

        Args:
            raw_choice (str): Schedule string type.
        """
        self.choice = self.Choice(raw_choice)


#### Master ####
@dataclass
class Master(Construct):
    """Represent an OpenMP master construct."""


#### Masked ####
@dataclass
class Masked(Construct):
    """Represent an OpenMP masked construct.

    Attributes:
        filter (Filter | None): Filter clause.
    """

    filter: Filter | None = None


@dataclass
class Filter(Clause):
    """Represent an OpenMP filter clause.

    Attributes:
        thread (PyExpr): Thread id.
    """

    thread: PyExpr


#### Critical ####
@dataclass
class Critical(Construct):
    """Represent an OpenMP critical construct.

    Attributes:
        critical_name (PyName | None): Optional name of the critical region.
        hint (Hint | None): Hint clause.
    """

    critical_name: PyName | None = None
    hint: Hint | None = None


@dataclass
class Hint(Clause):
    """Represent an OpenMP hint clause.

    Attributes:
        expr (PyExpr): Expression specifying the hint.
    """

    expr: PyExpr


#### Barrier ####
@dataclass
class Barrier(Construct):
    """Represent an OpenMP barrier construct."""


#### Taskwait ####
@dataclass
class Taskwait(Construct):
    """Represent an OpenMP taskwait construct."""


#### Taskgroup ####
@dataclass
class Taskgroup(Construct):
    """Represent an OpenMP taskgroup construct."""


#### Atomic ####
@dataclass
class Atomic(Construct):
    """Represent an OpenMP atomic construct.

    Attributes:
        read (Read | None): Read clause.
        write (Write | None): Write clause.
        update (Update | None): Update clause.
        capture (Capture | None): Capture clause.
        seq_cst (SeqCst): Seq_cst clause.
    """

    read: Read | None = None
    write: Write | None = None
    update: Update | None = None
    capture: Capture | None = None
    seq_cst: SeqCst | None = None


@dataclass
class Read(Clause):
    """Represent an OpenMP atomic read clause.

    Attributes:
        use_semantics (PyExpr | None): Optional memory-order semantics expression.
    """

    use_semantics: PyExpr | None = None


@dataclass
class Write(Clause):
    """Represent an OpenMP atomic write clause.

    Attributes:
        use_semantics (PyExpr | None): Optional memory-order semantics expression.
    """

    use_semantics: PyExpr | None = None


@dataclass
class Update(Clause):
    """Represent an OpenMP atomic update clause.

    Attributes:
        use_semantics (PyExpr | None): Optional memory-order semantics expression.
    """

    use_semantics: PyExpr | None = None


@dataclass
class Capture(Clause):
    """Represent an OpenMP atomic capture clause.

    Attributes:
        use_semantics (PyExpr | None): Optional memory-order semantics expression.
    """

    use_semantics: PyExpr | None = None


@dataclass
class SeqCst(Clause):
    """Represent an OpenMP seq_cst clause."""


#### Flush ####
@dataclass
class Flush(Construct):
    """Represent an OpenMP flush construct.

    Attributes:
        targets (list[PyName] | None): Names of the variables to flush.
    """

    targets: list[PyName] | None = None

    @property
    def str_targets(self) -> list[str]:
        """Return target names as strings.

        Returns:
            list[str]: Target names.
        """
        return [v.string for v in self.targets or []]


#### Ordered ####
@dataclass
class Ordered(Construct):
    """Represent an OpenMP ordered construct.

    Attributes:
        threads (Threads | None): Threads clause.
        simd_clause (SimdClause | None): SIMD clause.
        depend (list[Depend]): Depend clauses.
    """

    threads: Threads | None = None
    simd_clause: SimdClause | None = None
    depend: list[Depend] = field(default_factory=list)


#### Cancel and CancellationPoint ####
@dataclass
class Cancel(Construct):
    """Represent an OpenMP cancel construct.

    Attributes:
        construct_type (ConstructType): Type of construct to cancel.
        if_ (If | None): If clause.
    """

    construct_type: ConstructType
    if_: If | None = None


@dataclass
class CancellationPoint(Construct):
    """Represent an OpenMP cancellation point construct.

    Attributes:
        construct_type (ConstructType): Type of construct for the cancellation point.
    """

    construct_type: ConstructType


@dataclass
class ConstructType(Clause):
    """Represent the type of OpenMP construct for cancellation.

    Attributes:
        choice (Choice): Construct type.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported construct types."""

        FOR = "for"
        PARALLEL = "parallel"
        SECTIONS = "sections"
        TASKGROUP = "taskgroup"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the construct type from its string value."""
        self.choice = self.Choice(raw_choice.lower())


#### ThreadPrivate ####
@dataclass
class ThreadPrivate(Construct):
    """Represent an OpenMP threadprivate construct.

    Attributes:
        targets (list[PyName]): Names of the threadprivate variables.
    """
    targets: list[PyName]

    @property
    def str_targets(self) -> list[str]:
        """Return target names as strings.

        Returns:
            list[str]: Target names.
        """
        return [v.string for v in self.targets]


#### DeclareReduction ####
@dataclass
class DeclareReduction(Construct):
    """Represent an OpenMP declare reduction construct.

    Attributes:
        op (PyName): Reduction operator.
        ann_list (list[PyExpr]): Expressions associated with the declaration.
        combiner (Combiner): Combiner expression or statement.
        initializer (Initializer | None): Optional initializer.
    """

    op: PyName
    ann_list: list[PyExpr]
    combiner: Combiner
    initializer: Initializer | None = None


@dataclass
class Initializer(Clause):
    """Represent an OpenMP reduction initializer.

    Attributes:
        code (PyCode): Initializer Python code.
    """

    code: PyCode


@dataclass
class Combiner(Clause):
    """Represent an OpenMP reduction combiner.

    Attributes:
        code (PyCode): Combiner Python code.
    """

    code: PyCode


#######################################################################################################################
############################################ Common Clauses and Modifiers #############################################
#######################################################################################################################


@dataclass
class If(Clause):
    """Represent an OpenMP if-clause.

    Attributes:
        expr (PyExpr): Conditional expression.
    """

    expr: PyExpr

    @classmethod
    def id(cls) -> str:
        """Return the clause identifier.

        Returns:
            str: The identifier ``if_``.
        """
        return "if_"


@dataclass
class NoWait(Clause):
    """Represent an OpenMP nowait clause.

    Attributes:
        not_synchronize (PyExpr | None): Expression controlling whether waiting is enabled.
    """

    not_synchronize: PyExpr | None = None


@dataclass
class Collapse(Clause):
    """Represent an OpenMP collapse clause.

    Attributes:
        num (PyInt): Number of loops to collapse.
    """

    num: PyInt


@dataclass
class Threads(Clause):
    """Represent an OpenMP threads clause.

    Attributes:
        appy_to_threads (PyExpr | None): Expression associated with the threads clause.
    """

    appy_to_threads: PyExpr | None = None


@dataclass
class SimdClause(Clause):
    """Represent an OpenMP simd clause.

    Attributes:
        appy_to_simd (PyExpr | None): Expression associated with the SIMD clause.
    """

    appy_to_simd: PyExpr | None = None


@dataclass
class Device(Clause):
    """Represent an OpenMP device clause.

    Attributes:
        description (PyExpr): Expression specifying the target device.
    """

    description: PyExpr


@dataclass
class Depend(Clause):
    """Represent an OpenMP depend clause.

    Attributes:
        dependence_type (DependenceType): Type of dependence.
        items (list[PyName]): Names of the dependent items.
    """

    dependence_type: DependenceType
    items: list[PyName] = field(default_factory=list)


@dataclass
class DependenceType(Modifier):
    """Represent an OpenMP dependence type.

    Attributes:
        choice (Choice): Dependence type.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported dependence types."""

        IN = "in"
        OUT = "out"
        INOUT = "inout"
        SOURCE = "source"
        SINK = "sink"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the dependence type from its string value."""
        self.choice = self.Choice(raw_choice.lower())


#### DATA SHARING ####


@dataclass
class Default(Clause):
    """Represent an OpenMP default data-sharing clause.

    Attributes:
        data (DefaultData): Default data-sharing attribute.
    """

    data: DefaultData


@dataclass
class DefaultData(OmpNode):
    """Represent an OpenMP default data-sharing attribute.

    Attributes:
        choice (Choice): Default data-sharing attribute value.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported default data-sharing attribute."""

        FIRST_PRIVATE = "firstprivate"
        NONE = "none"
        PRIVATE = "private"
        FOR = "shared"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the data-sharing attribute from its string value."""
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class Shared(DataScope):
    """Represent an OpenMP shared clause."""


@dataclass
class Private(DataScope):
    """Represent an OpenMP private clause."""


@dataclass
class FirstPrivate(DataScope):
    """Represent an OpenMP firstprivate clause."""


@dataclass
class LastPrivate(DataScope):
    """Represent an OpenMP lastprivate clause."""


@dataclass
class Reduction(DataScope):
    """Represent an OpenMP reduction clause.

    Attributes:
        op (ReductionOp): Reduction operator.
    """

    op: ReductionOp


@dataclass
class ReductionOp(Modifier):
    """Represent an OpenMP reduction operator.

    Attributes:
        op (PyName): Name of the reduction operator.
    """

    op: PyName


@dataclass
class Map(DataScope):
    """Represent an OpenMP map clause.

    Attributes:
        map_type (MapType): Data-mapping type.
        map_modifier (Name | None): Optional map modifier.
    """

    map_type: MapType
    map_modifier: Name | None = None


@dataclass
class MapType(Modifier):
    """Represent an OpenMP map type.

    Attributes:
        choice (Choice): Data-mapping type.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported map types."""

        TO = "to"
        FROM = "from"
        TOFROM = "tofrom"
        ALLOC = "alloc"
        RELEASE = "release"
        DELETE = "delete"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the map type from its string value."""
        self.choice = self.Choice(raw_choice.lower())


@dataclass
class MapModifier(Modifier):
    """Represent an OpenMP map modifier.

    Attributes:
        choice (Choice): Map modifier.
    """

    raw_choice: InitVar[str]
    choice: Choice = field(init=False)

    class Choice(StrEnum):
        """Supported map modifiers."""

        ALWAYS = "always"

    def __post_init__(self, raw_choice: str) -> None:
        """Initialize the map modifier from its string value."""
        self.choice = self.Choice(raw_choice.lower())
