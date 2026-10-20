"""Parser nodes for directives, clauses, and expressions.

An OpenMP directive consists of a construct and zero or more clauses. Each
keyword within the directive represents either the construct or a clause,
and together they form the directive's elements. Both constructs and clauses
can include modifiers, which are arguments that further define their behavior.
"""

from __future__ import annotations
from omp4py.runtime.icvs import defaults
from Cython.Compiler.Options import CompilationOptions

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ast import alias, arg, expr, keyword, pattern, stmt, type_param

__all__ = [
    # Base
    "Name", "Directive", "Construct", "Clause", "DataScope", "Modifier", "DirectiveName",
    "PyExpr", "PyInt", "PyName", "PyStmt",

    # Constructs
    "Parallel", "For", "Sections", "Section", "Single",
    "Simd", "DeclareSimd",
    "Task", "Taskloop", "Taskyield",
    "TargetData", "TargetEnterData", "TargetExitData", "Target", "TargetUpdate", "DeclareTarget", "Teams", "Distribute",
    "Master", "Critical", "Barrier", "Taskwait", "Taskgroup", "Atomic", "Flush", "Ordered",
    "CancelDirectiveName", "Cancel", "CancellationPoint",
    "ThreadPrivate", "DeclareReduction",

    # Clauses
    "If", "NoWait", "Collapse",
    "Default", "Shared", "Private", "FirstPrivate", "LastPrivate", "Reduction", "Linear",
    "CopyIn", "CopyPrivate",
    "TaskDependenceKind", "Depend", "Map", "DefaultMap",

    "NumThreads", "ProcBind", "Schedule", "OrderedClause",
    "SafeLen", "Simdlen", "Aligned",
    "Uniform", "InBranch", "NotInBranch",
    "Final", "Untied", "Mergeable", "Priority",
    "GrainSize", "NumTasks", "NoGroup",
    "Device", "UseDevicePtr", "IsDevicePtr",
    "To", "From", "Link",
    "NumTeams", "ThreadLimit", "DistSchedule",
    "Read", "Write", "Update", "Capture", "SeqCst",
    "Threads", "SimdClause", "Initializer", "Combiner",
    "ScheduleType", "ReductionOp",
]


#######################################################################################################################
######################################################## Base #########################################################
#######################################################################################################################


@dataclass
class Span:
    lineno: int
    offset: int
    end_lineno: int
    end_offset: int

    @staticmethod
    def from_ast(node: expr | stmt | arg | keyword | alias | pattern | type_param) -> Span:
        return Span(
            node.lineno,
            node.col_offset,
            node.end_lineno if node.end_lineno is not None else -1,
            node.end_col_offset if node.end_col_offset is not None else -1,
        )

    def to_ast[T: expr | stmt | arg | keyword | alias | pattern | type_param](self, node: T) -> T:
        node.lineno = self.lineno
        node.col_offset = self.offset
        node.end_lineno = self.end_lineno
        node.end_col_offset = self.end_offset
        return node


@dataclass
class OmpNode:
    span: Span


@dataclass
class Name(OmpNode):
    string: str

    def __str__(self):
        return self.string


#### DIRECTIVES & CONSTRUCTS ####

@dataclass
class Directive(OmpNode):
    string: str
    # Dict[directive_name, Construct] to model combined constructs.
    # In OpenMP 6.0 there is 413 different combined constructs,
    # it doesn't make sense to create a class for each of them.
    constructs: dict[str, Construct]


@dataclass
class Construct(OmpNode):
    id: ClassVar[str] = "construct" # must be redefined
    name: Name


#### CLAUSES ####

@dataclass(kw_only=True)
class Clause(OmpNode):
    id: ClassVar[str] = "clause"  # must be redefined
    directive_name: DirectiveName|None = None
    name: Name


@dataclass
class DataScope(Clause):
    targets: list[PyName]

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets]


#### MODIFIERS ####

@dataclass
class Modifier(OmpNode):
    id: ClassVar[str] = "modifier"  # must be redefined


@dataclass
class DirectiveName(Modifier):
    id: ClassVar[str] = "directive_name"
    string: str

    def __str__(self):
        return self.string


#######################################################################################################################
##################################################### Constructs ######################################################
#######################################################################################################################

@dataclass
class Parallel(Construct):
    id: ClassVar[str] = "parallel"
    if_: If|None = None
    num_threads: NumThreads | None = None
    default: Default|None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    copyin: list[CopyIn] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    proc_bind: ProcBind|None = None


#### WORKSHARING CONTRUCTS ####

@dataclass
class For(Construct):
    id: ClassVar[str] = "for"
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    schedule: Schedule|None = None
    collapse: Collapse|None = None
    ordered: OrderedClause|None = None
    no_wait: NoWait|None = None

@dataclass
class Sections(Construct):
    id: ClassVar[str] = "sections"
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    no_wait: NoWait|None = None

@dataclass
class Section(Construct):
    id: ClassVar[str] = "section"

@dataclass
class Single(Construct):
    id: ClassVar[str] = "single"
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    copyprivate: list[CopyPrivate] = field(default_factory=list)
    no_wait: NoWait|None = None


#### SIMD CONTRUCTS ####

@dataclass
class Simd(Construct):
    id: ClassVar[str] = "simd"
    safelen: SafeLen|None = None
    simdlen: Simdlen|None = None
    aligned: list[Aligned] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    private: list[Private] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    collapse: Collapse|None = None

@dataclass
class DeclareSimd(Construct):
    id: ClassVar[str] = "declare_simd"
    proc_name: PyExpr|None = None

    linear: list[Linear] = field(default_factory=list)
    simdlen: Simdlen|None = None
    aligned: list[Aligned] = field(default_factory=list)
    uniform: list[Uniform] = field(default_factory=list)
    in_branch: InBranch|None = None
    not_in_branch: NotInBranch|None = None


#### TASKING CONTRUCTS ####

@dataclass
class Task(Construct):
    id: ClassVar[str] = "task"
    if_: If|None = None
    final: Final|None = None
    untied: Untied|None = None
    default: Default|None = None
    mergeable: Mergeable|None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    depend: list[Depend] = field(default_factory=list)
    priority: Priority|None = None

@dataclass
class Taskloop(Construct):
    id: ClassVar[str] = "taskloop"
    if_: If|None = None
    shared: list[Shared] = field(default_factory=list)
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    default: Default|None = None
    grain_size: GrainSize|None = None
    num_tasks: NumTasks|None = None
    collapse: Collapse|None = None
    final: Final|None = None
    priority: Priority|None = None
    untied: Untied|None = None
    mergeable: Mergeable|None = None
    no_group: NoGroup|None = None

@dataclass
class Taskyield(Construct):
    id: ClassVar[str] = "taskyield"


#### DEVICE CONTRUCTS ####

@dataclass(kw_only=True)
class TargetData(Construct):
    id: ClassVar[str] = "target_data"
    if_: If|None = None
    device: Device|None = None
    map: Map|None = None
    use_device_ptr: list[UseDevicePtr] = field(default_factory=list)

@dataclass
class TargetEnterData(Construct):
    id: ClassVar[str] = "target_enter_data"
    if_: If|None = None
    device: Device|None = None
    map: Map|None = None
    depend: list[Depend] = field(default_factory=list)
    no_wait: NoWait|None = None

@dataclass
class TargetExitData(Construct):
    id: ClassVar[str] = "target_exit_data"
    if_: If|None = None
    device: Device|None = None
    map: Map|None = None
    depend: list[Depend] = field(default_factory=list)
    no_wait: NoWait|None = None

@dataclass
class Target(Construct):
    id: ClassVar[str] = "target"
    if_: If|None = None
    device: Device|None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    is_device_ptr: list[IsDevicePtr] = field(default_factory=list)
    default_map: DefaultMap|None = None
    no_wait: NoWait|None = None
    depend: list[Depend] = field(default_factory=list)

@dataclass
class TargetUpdate(Construct):
    id: ClassVar[str] = "target_update"
    if_: If|None = None
    device: Device|None = None
    no_wait: NoWait|None = None
    depend: list[Depend] = field(default_factory=list)
    to: list[To] = field(default_factory=list)
    from_: list[From] = field(default_factory=list)

@dataclass
class DeclareTarget(Construct):
    id: ClassVar[str] = "declare_target"
    targets: list[PyName]|None = None

    link: list[Link] = field(default_factory=list)
    to: list[To] = field(default_factory=list)

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets or []]

@dataclass
class Teams(Construct):
    id: ClassVar[str] = "teams"
    num_teams: NumTeams|None = None
    thread_limit: ThreadLimit|None = None
    default: Default|None = None
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)

@dataclass
class Distribute(Construct):
    id: ClassVar[str] = "distribute"
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    collapse: Collapse|None = None
    dist_schedule: DistSchedule|None = None


#### MASTER AND SYNCH CONSTRUCTS ####

@dataclass
class Master(Construct):
    id: ClassVar[str] = "master"

@dataclass
class Critical(Construct):
    id: ClassVar[str] = "critical"
    critical_name: PyName|None = None
    hint: Hint|None = None

@dataclass
class Barrier(Construct):
    id: ClassVar[str] = "barrier"

@dataclass
class Taskwait(Construct):
    id: ClassVar[str] = "taskwait"

@dataclass
class Taskgroup(Construct):
    id: ClassVar[str] = "taskgroup"

@dataclass
class Atomic(Construct):
    id: ClassVar[str] = "atomic"
    read: Read|None = None
    write: Write|None = None
    update: Update|None = None
    capture: Capture|None = None
    seq_cst: list[SeqCst] = field(default_factory=list)

@dataclass
class Flush(Construct):
    id: ClassVar[str] = "flush"
    targets: list[PyName]|None = None

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets or []]

@dataclass
class Ordered(Construct):
    id: ClassVar[str] = "ordered"
    threads: Threads|None = None
    simd: SimdClause|None = None
    depend: list[Depend] = field(default_factory=list)


#### CANCELLATION CONSTRUCTS ####

class CancelDirectiveName(Enum):
    FOR = 0
    PARALLEL = 1
    SECTIONS = 2
    TASKGROUP = 3

    @staticmethod
    def from_name(name: Name) -> CancelDirectiveName:
        return {
            "for": CancelDirectiveName.FOR,
            "parallel": CancelDirectiveName.PARALLEL,
            "sections": CancelDirectiveName.SECTIONS,
            "taskgroup": CancelDirectiveName.TASKGROUP,
        }[name.string.lower()]

@dataclass(kw_only=True)
class Cancel(Construct):
    id: ClassVar[str] = "cancel"
    if_: If|None = None
    construct_type_name: Name
    construct_type: CancelDirectiveName = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "construct_type", CancelDirectiveName.from_name(self.construct_type_name))

@dataclass
class CancellationPoint(Construct):
    id: ClassVar[str] = "cancellationpoint"
    construct_type_name: Name
    construct_type: CancelDirectiveName = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "construct_type", CancelDirectiveName.from_name(self.construct_type_name))


#### OTHER CONSTRUCTS ####

@dataclass
class ThreadPrivate(Construct):
    id: ClassVar[str] = "threadprivate"
    targets: list[PyName] = field(default_factory=list)

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets]

@dataclass
class DeclareReduction(Construct):
    id: ClassVar[str] = "declare_reduction"
    op: ReductionOp
    combiner: Combiner
    ann_list: list[PyExpr]
    initializer: Initializer|None = None


#######################################################################################################################
####################################################### Clauses #######################################################
#######################################################################################################################

@dataclass
class If(Clause):
    id: ClassVar[str] = "if_"
    expr: PyExpr

@dataclass
class NoWait(Clause):
    id: ClassVar[str] = "no_wait"
    dont_synchronize: PyExpr|None = None

@dataclass
class Collapse(Clause):
    id: ClassVar[str] = "collapse"
    num: PyInt


#### DATA SHARING CLAUSES ####

@dataclass
class Default(Clause):
    id: ClassVar[str] = "default"
    data_sharing_attr_name: Name
    data_sharing_attr: DataSharingAttr = field(init=False)

    class DataSharingAttr(Enum):
        SHARED = 0
        NONE = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_sharing_attr",
            {
                "shared": self.DataSharingAttr.SHARED,
                "none": self.DataSharingAttr.NONE,
            }[self.data_sharing_attr_name.string.lower()],
        )

@dataclass
class Shared(DataScope):
    id: ClassVar[str] = "shared"

@dataclass
class Private(DataScope):
    id: ClassVar[str] = "private"

@dataclass
class FirstPrivate(DataScope):
    id: ClassVar[str] = "first_private"

@dataclass
class LastPrivate(DataScope):
    id: ClassVar[str] = "last_private"

@dataclass
class Reduction(DataScope):
    id: ClassVar[str] = "reduction"
    op: ReductionOp

@dataclass
class Linear(DataScope):
    id: ClassVar[str] = "linear"
    linear_step: PyExpr|None = None
    # TODO: how to represent REF/UVAL/VAL here?


#### DATA COPYING CLAUSES ####

@dataclass
class CopyIn(DataScope):
    id: ClassVar[str] = "copyin"

@dataclass
class CopyPrivate(DataScope):
    id: ClassVar[str] = "copyprivate"


#### MASTER AND SYNCH CLAUSES ####

class TaskDependenceKind(Enum):
    IN = 0
    OUT = 1
    INOUT = 2
    SOURCE = 3
    SINK = 4

    @staticmethod
    def from_name(name: Name|None) -> TaskDependenceKind|None:
        return (
            {
                "in": TaskDependenceKind.IN,
                "out": TaskDependenceKind.OUT,
                "inout": TaskDependenceKind.INOUT,
                "source": TaskDependenceKind.SOURCE,
                "sink": TaskDependenceKind.SINK,
            }[name.string.lower()]
            if name is not None
            else None
        )

@dataclass
class Depend(Clause):
    id: ClassVar[str] = "depend"
    locator_list: list[PyExpr] = field(default_factory=list)
    task_dependence_name: Name|None = None
    task_dependence: TaskDependenceKind|None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_dependence",
            TaskDependenceKind.from_name(self.task_dependence_name)
        )


#### DATA MAPPING CLAUSES ####

@dataclass
class Map(DataScope):
    id: ClassVar[str] = "map"

    # modifiers:
    always_modifier_name: Name|None = None
    map_type_name: Name|None = None
    map_type: MapType|None = field(init=False)

    class MapType(Enum):
        TO = 0
        FROM = 1
        TOFROM = 2
        ALLOC = 3
        RELEASE = 4
        DELETE = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "map_type",
            {
                "to": self.MapType.TO,
                "from": self.MapType.FROM,
                "tofrom": self.MapType.TOFROM,
                "alloc": self.MapType.ALLOC,
                "release": self.MapType.RELEASE,
                "delete": self.MapType.DELETE,
            }[self.map_type_name.string.lower()]
            if self.map_type_name is not None
            else None
        )

@dataclass
class DefaultMap(Clause):
    id: ClassVar[str] = "use_device_addr"
    tofrom_modifier_name: Name
    scalar_modifier_name: Name|None = None


#######################################################################################################################


#### PARALLEL CONSTRUCT CLAUSES ####

@dataclass
class NumThreads(Clause):
    id: ClassVar[str] = "num_threads"
    nthreads: PyExpr

@dataclass
class ProcBind(Clause):
    id: ClassVar[str] = "proc_bind"
    affinity_policy_name: Name
    affinity_policy: AffinityPolicy = field(init=False)

    class AffinityPolicy(Enum):
        CLOSE = 0
        PRIMARY = 1
        SPREAD = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            {
                "close": self.AffinityPolicy.CLOSE,
                "primary": self.AffinityPolicy.PRIMARY,
                "spread": self.AffinityPolicy.SPREAD,
            }[self.affinity_policy_name.string.lower()],
        )


#### FOR CONSTRUCT CLAUSES ####

@dataclass
class Schedule(Clause):
    id: ClassVar[str] = "schedule"
    type: ScheduleType
    chunk: PyExpr|None = None

    # modifiers:
    schedule_modifier_name: Name|None = None
    schedule_modifier: ScheduleModifier|None = field(init=False)
    chunk_modifier_name: Name|None = None

    class ScheduleModifier(Enum):
        MONOTONIC = 0
        NON_MONOTONIC = 1
        SIMD = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordering_modifier",
            {
                "monotonic": self.ScheduleModifier.MONOTONIC,
                "nonmonotonic": self.ScheduleModifier.NON_MONOTONIC,
                "simd": self.ScheduleModifier.SIMD,
            }[self.schedule_modifier_name.string.lower()]
            if self.schedule_modifier_name is not None
            else None
        )

@dataclass
class OrderedClause(Clause):
    id: ClassVar[str] = "ordered"
    n: PyInt|None = None


#### SIMD CONSTRUCT CLAUSES ####

@dataclass
class SafeLen(Clause):
    id: ClassVar[str] = "safelen"
    length: PyExpr

@dataclass
class Simdlen(Clause):
    id: ClassVar[str] = "simdlen"
    length: PyExpr

@dataclass
class Aligned(DataScope):
    id: ClassVar[str] = "aligned"
    alignment_modifier: PyInt|None = None


#### DECLARE_SIMD CONSTRUCT CLAUSES ####

@dataclass
class Uniform(DataScope):
    id: ClassVar[str] = "uniform"

@dataclass
class InBranch(Clause):
    id: ClassVar[str] = "in_branch"
    in_branch: PyExpr|None = None

@dataclass
class NotInBranch(Clause):
    id: ClassVar[str] = "not_in_branch"
    not_in_branch: PyExpr|None = None


#### TASK CONSTRUCT CLAUSES ####

@dataclass
class Final(Clause):
    id: ClassVar[str] = "final"
    finalize: PyExpr

@dataclass
class Untied(Clause):
    id: ClassVar[str] = "untied"
    can_change_threads: PyExpr|None = None

@dataclass
class Mergeable(Clause):
    id: ClassVar[str] = "mergeable"
    can_merge: PyExpr|None = None

@dataclass
class Priority(Clause):
    id: ClassVar[str] = "priority"
    value: PyExpr


#### TASKLOOP CONSTRUCT CLAUSES ####

@dataclass
class GrainSize(Clause):
    id: ClassVar[str] = "grain_size"
    grain_size: PyExpr

@dataclass
class NumTasks(Clause):
    id: ClassVar[str] = "num_tasks"
    num_tasks: PyExpr

@dataclass
class NoGroup(Clause):
    id: ClassVar[str] = "no_group"
    dont_synchronize: PyExpr|None = None


#### TARGET_DATA CONSTRUCT CLAUSES ####

@dataclass
class Device(Clause):
    id: ClassVar[str] = "device"
    device_description: PyExpr

@dataclass
class UseDevicePtr(DataScope):
    id: ClassVar[str] = "use_device_ptr"


#### TARGET CONSTRUCT CLAUSES ####

@dataclass
class IsDevicePtr(DataScope):
    id: ClassVar[str] = "is_device_ptr"


#### TARGET_UPDATE CONSTRUCT CLAUSES ####

@dataclass
class To(DataScope):
    id: ClassVar[str] = "to"

@dataclass
class From(DataScope):
    id: ClassVar[str] = "from_"


#### DECLARE_TARGET CONSTRUCT CLAUSES ####

@dataclass
class Link(DataScope):
    id: ClassVar[str] = "link"


#### TEAMS CONSTRUCT CLAUSES ####

@dataclass
class NumTeams(Clause):
    id: ClassVar[str] = "num_teams"
    upper_bound: PyExpr

@dataclass
class ThreadLimit(Clause):
    id: ClassVar[str] = "thread_limit"
    threadlim: PyExpr


#### DISTRIBUTE CONSTRUCT CLAUSES ####

@dataclass
class DistSchedule(Clause):
    id: ClassVar[str] = "dist_schedule"
    static: Name # = static
    chunk_size: PyExpr|None = None


#### CRITICAL CONSTRUCT CLAUSES ####

@dataclass
class Hint(Clause):
    id: ClassVar[str] = "hint"
    expr: PyExpr

#### DISTRIBUTE CONSTRUCT CLAUSES ####

@dataclass
class Read(Clause):
    id: ClassVar[str] = "read"
    use_semantics: PyExpr|None = None

@dataclass
class Write(Clause):
    id: ClassVar[str] = "write"
    use_semantics: PyExpr|None = None

@dataclass
class Update(Clause):
    id: ClassVar[str] = "update"
    use_semantics: PyExpr|None = None

@dataclass
class Capture(Clause):
    id: ClassVar[str] = "capture"
    use_semantics: PyExpr|None = None

@dataclass
class SeqCst(Clause):
    id: ClassVar[str] = "seq_cst"


#### ORDERED CONSTRUCT CLAUSES ####

@dataclass
class Threads(Clause):
    id: ClassVar[str] = "threads"
    appy_to_threads: PyExpr|None = None

@dataclass
class SimdClause(Clause):
    id: ClassVar[str] = "simd"
    appy_to_simd: PyExpr|None = None


#### DECLARE_REDUCTION CONSTRUCT CLAUSES ####

@dataclass
class Initializer(Clause):
    id: ClassVar[str] = "initializer"
    initializer_stmt: PyStmt

@dataclass
class Combiner(Clause):
    id: ClassVar[str] = "combiner"
    combiner_stmt: PyStmt


#######################################################################################################################
###################################################### Modifiers ######################################################
#######################################################################################################################

@dataclass
class ScheduleType(Modifier):
    id: ClassVar[str] = "type"
    kind_name: Name
    kind: Kind = field(init=False)

    class Kind(Enum):
        STATIC = 0
        DYNAMIC = 1
        GUIDED = 2
        AUTO = 3
        RUNTIME = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            {
                "static": self.Kind.STATIC,
                "dynamic": self.Kind.DYNAMIC,
                "guided": self.Kind.GUIDED,
                "auto": self.Kind.AUTO,
                "runtime": self.Kind.RUNTIME,
            }[self.kind_name.string.lower()],
        )

@dataclass
class ReductionOp(Modifier):
    id: ClassVar[str] = "op"
    value: str


#######################################################################################################################
################################################## Python Modifiers ###################################################
#######################################################################################################################

@dataclass
class PyExpr(Modifier):
    """Represents a Python expression in the AST.

    Attributes:
        value (expr): The AST node representing the expression.
    """

    value: expr
    source: str


@dataclass
class PyInt(Modifier):
    """Represents a Python integer.

    Attributes:
        value (int):

    """

    value: int

    def __int__(self) -> int:
        return self.value


@dataclass
class PyName(Modifier):
    """Represents a Python name.

    Attributes:
        string (str): The string representing the name.
    """

    string: str

    def __str__(self) -> str:
        return self.string


@dataclass
class PyStmt(Modifier):
    """Represents a Python statement in the AST.

    Attributes:
        value (stmt): The AST node representing the statement.
    """

    value: stmt
    source: str
