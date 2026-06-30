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
    "Span",
    "OmpNode",
    "Name",
    "Directive",
    "Construct",
    "Clause",
    "DataScope",
    "Modifier",
    # Constructs
    "ThreadPrivate", "DeclareReduction", "DeclareInduction", "Scan", "DeclareMapper", "GroupPrivate",
    "Allocate",
    "Metadirective", "DeclareVariant", "Dispatch", "DeclareSimd", "DeclareTarget",
    "Requires", "Assume", "Nothing", "Error",
    "Fuse", "Interchange", "Reverse", "Split", "Stripe", "Tile", "Unroll",
    "Parallel", "Teams", "Simd", "Masked",
    "Single", "Scope", "Sections", "Section", "Workshare", "Workdistribute", "For", "Distribute", "Loop",
    "Task", "Taskloop", "TaskIteration", "Taskyield", "Taskgraph",
    "TargetData", "TargetEnterData", "TargetExitData", "Target", "TargetUpdate",
    "InteropConstruct",
    "Critical", "Barrier", "Taskgroup", "Taskwait", "Atomic", "Flush", "Depobj", "Ordered",
    "Cancel", "CancellationPoint",
    # Clauses
    "Combiner", "Initializer", "Inductor", "Collector",
    "Exclusive", "Inclusive", "InitComplete",
    "DeviceType",
    "Align", "Allocator", "When", "Otherwise",
    "AdjustArgs", "AppendArgs", "Match",
    "InteropClause",
    "IsDevicePtr", "HasDeviceAddr", "NoContext", "NoVariants",
    "Aligned", "Linear", "Simdlen", "Uniform", "InBranch", "NotInBranch",
    "Enter", "Indirect", "Link", "Local",
    "AtomicDefaultMemOrder", "DynamicAllocators", "ReverseOffload", "UnifiedAddress", "UnifiedSharedMemory", "SelfMaps", "DeviceSafesync",
    "Absent", "Contains", "Holds", "NoOpenmp", "NoOpenmpConstructs", "NoOpenmpRoutines", "NoParallelism",
    "At", "Message", "Severity",
    "LoopRange", "Permutation", "Counts", "Sizes", "Full", "Partial",
    "CopyIn", "NumThreads", "ProcBind", "SafeSync", "NumTeams",
    "ThreadLimit", "NonTemporal", "Order", "SafeLen", "Filter",
    "CopyPrivate", "OrderedClause", "Schedule", "DistSchedule",
    "Bind", "GrainSize", "NumTasks", "GraphId", "GraphReset", "UseDevicePtr", "UseDeviceAddr",
    "DefaultMap", "UsesAllocators", "From", "To",
    "Destroy", "Init", "Use",
    "Hint", "TaskReduction", "MemScope",
    "Read", "Update", "Write", "Capture", "Compare", "Fail", "Weak", "AcqRel", "Acquire", "Relaxed", "Release", "SeqCst", "DepobjUpdate", "DoAcross", "SimdClause", "Threads",
    "Apply", "Depend", "Device", "Default",
    "Private", "If", "FirstPrivate", "Reduction", "Induction", "Shared", "Collapse", "LastPrivate", "AllocateClause", "NoWait",
    "Final", "Mergeable", "Untied", "Affinity", "Detach", "InReduction", "Priority", "Replayable", "ThreadSet", "Transparent", "NoGroup", "Map",
    # Modifiers
    "DirectiveName",
    "ScheduleType",
    "ReductionOp", "InductionOp",
    "Original", "InteropModifier", "Iterator", "Step", "AllocatorModifier", "AlignModifier",
    "Mapper", "MemSpace", "Traits", "DepInfo", "LoopModifier", "Prefer", "ContextSelector",
    "PyExpr", "PyInt", "PyName", "PyStmt",
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


#### DATA ENVIRONMENT DIRECTIVES ####

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
    ann_list: list[PyExpr]
    combiner: Combiner
    initializer: Initializer | None = None


@dataclass
class DeclareInduction(Construct):
    id: ClassVar[str] = "declare_induction"
    op: InductionOp
    ann_list: list[PyExpr]
    collector: Collector
    inductor: Inductor


@dataclass
class Scan(Construct):
    id: ClassVar[str] = "scan"
    exclusive: Exclusive|None = None
    inclusive: Inclusive|None = None
    init_complete: InitComplete|None = None


@dataclass(kw_only=True)
class DeclareMapper(Construct):
    id: ClassVar[str] = "declare_mapper"
    mapper_identifier: PyName|None = None
    var: PyName
    type: PyExpr
    map: list[Map] = field(default_factory=list)


@dataclass
class GroupPrivate(Construct):
    id: ClassVar[str] = "groupprivate"
    device_type: DeviceType|None = None


#### MEMORY MANAGEMENT DIRECTIVES ####

@dataclass
class Allocate(Construct):
    id: ClassVar[str] = "allocate"
    targets: list[PyName] = field(default_factory=list)

    align: Align|None = None
    allocator: Allocator|None = None

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets]


#### VARIANT DIRECTIVES ####

@dataclass
class Metadirective(Construct):
    id: ClassVar[str] = "metadirective"
    when: list[When] = field(default_factory=list)
    otherwise: Otherwise|None = None


@dataclass(kw_only=True)
class DeclareVariant(Construct):
    id: ClassVar[str] = "declare_variant"
    base_name: PyExpr|None = None
    variant_name: PyExpr

    adjust_args: list[AdjustArgs] = field(default_factory=list)
    append_args: AppendArgs|None = None
    match: Match


@dataclass
class Dispatch(Construct):
    id: ClassVar[str] = "declare_variant"
    depend: list[Depend] = field(default_factory=list)
    device: Device|None = None
    interop: list[InteropClause] = field(default_factory=list)
    is_device_ptr: list[IsDevicePtr] = field(default_factory=list)
    has_device_addr: list[HasDeviceAddr] = field(default_factory=list)
    no_context: NoContext|None = None
    no_variants: NoVariants|None = None
    no_wait: NoWait|None = None



@dataclass
class DeclareSimd(Construct):
    id: ClassVar[str] = "declare_simd"
    proc_name: PyExpr|None = None

    aligned: list[Aligned] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    simdlen: Simdlen|None = None
    uniform: list[Uniform] = field(default_factory=list)
    # TODO: InBranch and NotInBranch are exclusive
    in_branch: InBranch|None = None
    not_in_branch: NotInBranch|None = None


@dataclass
class DeclareTarget(Construct):
    id: ClassVar[str] = "declare_target"
    targets: list[PyName]|None = None

    device_type: DeviceType|None = None
    enter: list[Enter] = field(default_factory=list)
    indirect: Indirect|None = None
    link: list[Link] = field(default_factory=list)
    local: list[Local] = field(default_factory=list)

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets or []]


#### INFORMATIONAL AND UTILITY DIRECTIVES ####

@dataclass
class Requires(Construct):
    id: ClassVar[str] = "requires"
    atomic_default_mem_order: AtomicDefaultMemOrder|None = None
    dynamic_allocators: DynamicAllocators|None = None
    reverse_offload: ReverseOffload|None = None
    unified_address: UnifiedAddress|None = None
    unified_shared_memory: UnifiedSharedMemory|None = None
    self_maps: SelfMaps|None = None
    device_safesync: DeviceSafesync|None = None


@dataclass
class Assume(Construct):
    id: ClassVar[str] = "assume"
    absent: Absent|None = None
    contains: Contains|None = None
    holds: Holds|None = None
    no_openmp: NoOpenmp|None = None
    no_openmop_contructs: NoOpenmpConstructs|None = None
    no_openmp_routines: NoOpenmpRoutines|None = None
    no_parallelism: NoParallelism|None = None


@dataclass
class Nothing(Construct):
    id: ClassVar[str] = "nothing"
    apply: list[Apply] = field(default_factory=list)


@dataclass
class Error(Construct):
    id: ClassVar[str] = "error"
    at: At|None = None
    message: Message|None = None
    severity: Severity|None = None


#### LOOP TRANSFORMING CONSTRUCTS ####

@dataclass
class Fuse(Construct):
    id: ClassVar[str] = "fuse"
    apply: Apply|None = None
    looprange: LoopRange|None = None


@dataclass
class Interchange(Construct):
    id: ClassVar[str] = "interchange"
    apply: Apply|None = None
    permutation: Permutation|None = None


@dataclass
class Reverse(Construct):
    id: ClassVar[str] = "reverse"
    apply: Apply|None = None


@dataclass(kw_only=True)
class Split(Construct):
    id: ClassVar[str] = "split"
    apply: Apply|None = None
    counts: Counts


@dataclass(kw_only=True)
class Stripe(Construct):
    id: ClassVar[str] = "stripe"
    apply: list[Apply] = field(default_factory=list)
    sizes: Sizes


@dataclass(kw_only=True)
class Tile(Construct):
    id: ClassVar[str] = "tile"
    apply: Apply|None = None
    sizes: Sizes


@dataclass
class Unroll(Construct):
    id: ClassVar[str] = "unroll"
    apply: Apply|None = None
    full: Full|None = None
    partial: Partial|None = None


#### PARALLELISM CONSTRUCTS ####

@dataclass
class Parallel(Construct):
    id: ClassVar[str] = "parallel"
    allocate: list[AllocateClause] = field(default_factory=list)
    copyin: list[CopyIn] = field(default_factory=list)
    default: Default|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    if_: If|None = None
    message: Message|None = None
    num_threads: NumThreads | None = None
    private: list[Private] = field(default_factory=list)
    proc_bind: ProcBind|None = None
    reduction: list[Reduction] = field(default_factory=list)
    safesync: SafeSync|None = None
    severity: Severity|None = None
    shared: list[Shared] = field(default_factory=list)


@dataclass
class Teams(Construct):
    id: ClassVar[str] = "teams"
    allocate: list[AllocateClause] = field(default_factory=list)
    default: Default|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    if_: If|None = None
    num_teams: NumTeams|None = None
    reduction: list[Reduction] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    thread_limit: ThreadLimit|None = None


@dataclass
class Simd(Construct):
    id: ClassVar[str] = "simd"
    aligned: list[Aligned] = field(default_factory=list)
    collapse: Collapse|None = None
    if_: If|None = None
    induction: list[Induction] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    non_temporal: list[NonTemporal] = field(default_factory=list)
    order: Order|None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    safelen: SafeLen|None = None
    simdlen: Simdlen|None = None


@dataclass
class Masked(Construct):
    id: ClassVar[str] = "masked"
    filter: Filter|None = None


#### WORKSHARING CONTRUCTS ####

@dataclass
class Single(Construct):
    id: ClassVar[str] = "single"
    # TODO: CopyPrivate and NoWait are exclusive
    allocate: list[AllocateClause] = field(default_factory=list)
    copyprivate: list[CopyPrivate] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    no_wait: NoWait | None = None
    private: list[Private] = field(default_factory=list)


@dataclass
class Scope(Construct):
    id: ClassVar[str] = "scope"
    allocate: list[AllocateClause] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    no_wait: NoWait | None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)


@dataclass
class Sections(Construct):
    id: ClassVar[str] = "sections"
    allocate: list[AllocateClause] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    no_wait: NoWait | None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)

@dataclass
class Section(Construct):
    id: ClassVar[str] = "section"


@dataclass
class Workshare(Construct):
    id: ClassVar[str] = "workshare"
    no_wait: NoWait | None = None


@dataclass
class Workdistribute(Construct):
    id: ClassVar[str] = "workdistribute"


@dataclass
class For(Construct):
    id: ClassVar[str] = "for"
    allocate: list[AllocateClause] = field(default_factory=list)
    collapse: Collapse|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    induction: list[Induction] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    no_wait: NoWait|None = None
    order: Order|None = None
    ordered: OrderedClause|None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    schedule: Schedule|None = None


@dataclass
class Distribute(Construct):
    id: ClassVar[str] = "distribute"
    allocate: list[AllocateClause] = field(default_factory=list)
    collapse: Collapse|None = None
    dist_schedule: DistSchedule|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    induction: list[Induction] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    order: Order|None = None
    private: list[Private] = field(default_factory=list)


@dataclass
class Loop(Construct):
    id: ClassVar[str] = "loop"
    bind: Bind|None = None


#### TASKING CONSTRUCTS ####

@dataclass
class Task(Construct):
    id: ClassVar[str] = "task"
    # TODO: Detach and Mergeable are exclusive
    affinity: list[Affinity] = field(default_factory=list)
    allocate: list[AllocateClause] = field(default_factory=list)
    default: Default|None = None
    depend: list[Depend] = field(default_factory=list)
    detach: Detach|None = None
    final: Final|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    if_: If|None = None
    in_reduction: list[InReduction] = field(default_factory=list)
    mergeable: Mergeable|None = None
    priority: Priority|None = None
    private: list[Private] = field(default_factory=list)
    replayable: list[Replayable] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    thread_set: ThreadSet|None = None
    transparent: Transparent|None = None
    untied: Untied|None = None


@dataclass
class Taskloop(Construct):
    id: ClassVar[str] = "taskloop"
    # TODO: NoGroup and Reduction are exclusive
    # TODO: GrainSize and NumTasks are exclusive
    allocate: list[AllocateClause] = field(default_factory=list)
    collapse: Collapse|None = None
    default: Default|None = None
    final: Final|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    grain_size: GrainSize|None = None
    if_: If|None = None
    in_reduction: list[InReduction] = field(default_factory=list)
    induction: list[Induction] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    mergeable: Mergeable|None = None
    no_group: NoGroup|None = None
    num_tasks: list[NumTasks] = field(default_factory=list)
    priority: Priority|None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    replayable: list[Replayable] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    thread_set: ThreadSet|None = None
    transparent: Transparent|None = None
    untied: Untied|None = None


@dataclass
class TaskIteration(Construct):
    id: ClassVar[str] = "task_iteration"
    affinity: list[Affinity] = field(default_factory=list)
    depend: list[Depend] = field(default_factory=list)
    if_: If|None = None


@dataclass
class Taskyield(Construct):
    id: ClassVar[str] = "taskyield"


@dataclass
class Taskgraph(Construct):
    id: ClassVar[str] = "taskgraph"
    graph_id: GraphId|None = None
    graph_reset: GraphReset|None = None
    if_: If|None = None
    no_group: NoGroup|None = None


#### DEVICE DIRECTIVES & CONSTRUCTS ####

@dataclass(kw_only=True)
class TargetData(Construct):
    id: ClassVar[str] = "target_data"
    # TODO: Map, UseDeviceAddr, UseDevicePtr are required
    affinity: list[Affinity] = field(default_factory=list)
    allocate: list[AllocateClause] = field(default_factory=list)
    default: Default|None = None
    depend: list[Depend] = field(default_factory=list)
    detach: Detach|None = None
    device: Device|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    if_: If|None = None
    in_reduction: list[InReduction] = field(default_factory=list)
    map: list[Map] = field(default_factory=list)
    mergeable: Mergeable|None = None
    no_group: NoGroup|None = None
    no_wait: NoWait|None = None
    priority: Priority|None = None
    private: list[Private] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    transparent: Transparent|None = None
    use_device_ptr: list[UseDevicePtr] = field(default_factory=list)
    use_device_addr: list[UseDeviceAddr] = field(default_factory=list)


@dataclass
class TargetEnterData(Construct):
    id: ClassVar[str] = "target_enter_data"
    depend: list[Depend] = field(default_factory=list)
    device: Device|None = None
    if_: If|None = None
    map: list[Map] = field(default_factory=list)
    no_wait: NoWait|None = None
    priority: Priority|None = None
    replayable: list[Replayable] = field(default_factory=list)


@dataclass
class TargetExitData(Construct):
    id: ClassVar[str] = "target_exit_data"
    depend: list[Depend] = field(default_factory=list)
    device: Device|None = None
    if_: If|None = None
    map: list[Map] = field(default_factory=list)
    no_wait: NoWait|None = None
    priority: Priority|None = None
    replayable: list[Replayable] = field(default_factory=list)


@dataclass
class Target(Construct):
    id: ClassVar[str] = "target"
    allocate: list[AllocateClause] = field(default_factory=list)
    default: Default|None = None
    default_map: DefaultMap|None = None
    depend: list[Depend] = field(default_factory=list)
    device: Device|None = None
    device_type: DeviceType|None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    has_device_addr: list[HasDeviceAddr] = field(default_factory=list)
    if_: If|None = None
    in_reduction: list[InReduction] = field(default_factory=list)
    is_device_ptr: list[IsDevicePtr] = field(default_factory=list)
    map: list[Map] = field(default_factory=list)
    no_wait: NoWait|None = None
    private: list[Private] = field(default_factory=list)
    priority: Priority|None = None
    replayable: list[Replayable] = field(default_factory=list)
    thread_limit: ThreadLimit|None = None
    uses_allocators: list[UsesAllocators] = field(default_factory=list)


@dataclass
class TargetUpdate(Construct):
    id: ClassVar[str] = "target_update"
    # TODO: From and To are required
    depend: list[Depend] = field(default_factory=list)
    device: Device|None = None
    from_: list[From] = field(default_factory=list)
    if_: If|None = None
    no_wait: NoWait|None = None
    priority: Priority|None = None
    replayable: list[Replayable] = field(default_factory=list)
    to: list[To] = field(default_factory=list)


#### INTEROPERABITLITY CONSTRUCTS ####

@dataclass
class InteropConstruct(Construct):
    id: ClassVar[str] = "interop_construct"
    # TODO: Destroy, Init and Use are required
    depend: list[Depend] = field(default_factory=list)
    destroy: list[Destroy] = field(default_factory=list)
    device: Device|None = None
    init: list[Init] = field(default_factory=list)
    no_wait: NoWait|None = None
    use: list[Use] = field(default_factory=list)


#### SYNCHRONIZATION CONSTRUCTS ####

@dataclass
class Critical(Construct):
    id: ClassVar[str] = "critical"
    critical_name: PyName|None = None
    hint: Hint|None = None


@dataclass
class Barrier(Construct):
    id: ClassVar[str] = "barrier"


@dataclass
class Taskgroup(Construct):
    id: ClassVar[str] = "taskgroup"
    allocate: list[AllocateClause] = field(default_factory=list)
    task_reduction: list[TaskReduction] = field(default_factory=list)


@dataclass
class Taskwait(Construct):
    id: ClassVar[str] = "taskwait"
    depend: list[Depend] = field(default_factory=list)
    no_wait: NoWait|None = None
    replayable: list[Replayable] = field(default_factory=list)


@dataclass
class Atomic(Construct):
    id: ClassVar[str] = "atomic"

    # TODO: memory-order and atomic groups are exclusive
    mem_scope: MemScope|None = None
    hint: Hint|None = None
    # atomic clause group:
    read: Read|None = None
    update: Update|None = None
    write: Write|None = None
    # extended-atomic clause group:
    capture: Capture|None = None
    compare: Compare|None = None
    fail: Fail|None = None
    weak: Weak|None = None
    # memory-order clause group:
    acq_rel: AcqRel|None = None
    acquire: Acquire|None = None
    relaxed: Relaxed|None = None
    release: Release|None = None
    seq_cst: SeqCst|None = None


@dataclass
class Flush(Construct):
    id: ClassVar[str] = "flush"
    targets: list[PyName]|None = None

    mem_scope: MemScope|None = None
    # memory-order clause group:
    acq_rel: AcqRel|None = None
    acquire: Acquire|None = None
    relaxed: Relaxed|None = None
    release: Release|None = None
    seq_cst: SeqCst|None = None

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets or []]


@dataclass
class Depobj(Construct):
    id: ClassVar[str] = "depobj"
    object: PyName
    # TODO: destroy, init and update are required
    destroy: Destroy|None = None
    init: Init|None = None
    depobj_update: DepobjUpdate|None = None


@dataclass
class Ordered(Construct):
    id: ClassVar[str] = "ordered"
    # TODO: DoAcross and SimdClause/Threads are exclusive
    do_across: DoAcross|None = None
    simd: SimdClause|None = None
    threads: Threads|None = None


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
    directive_name: DirectiveName|None = None
    nconstruct_type: Name
    construct_type: CancelDirectiveName = field(init=False)

    if_: If|None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "construct_type", CancelDirectiveName.from_name(self.nconstruct_type))


@dataclass
class CancellationPoint(Construct):
    id: ClassVar[str] = "cancellationpoint"
    nconstruct_type: Name
    construct_type: CancelDirectiveName = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "construct_type", CancelDirectiveName.from_name(self.nconstruct_type))


#######################################################################################################################
####################################################### Clauses #######################################################
#######################################################################################################################


#### declare reduction ####
@dataclass
class Combiner(Clause):
    id: ClassVar[str] = "combiner"
    combiner_stmt: PyStmt

@dataclass
class Initializer(Clause):
    id: ClassVar[str] = "initializer"
    initializer_stmt: PyStmt


#### declare induction ####
@dataclass
class Inductor(Clause):
    id: ClassVar[str] = "inductor"
    inductor_stmt: PyStmt

@dataclass
class Collector(Clause):
    id: ClassVar[str] = "collector"
    collector_expr: PyExpr


#### scan ####
@dataclass
class Exclusive(DataScope):
    id: ClassVar[str] = "exclusive"

@dataclass
class Inclusive(DataScope):
    id: ClassVar[str] = "inclusive"

@dataclass
class InitComplete(Clause):
    id: ClassVar[str] = "init_complete"
    create_init_phase: PyExpr|None = None

#### groupprivate ####
@dataclass
class DeviceType(Clause):
    id: ClassVar[str] = "device_type"
    ndevice_type_description: Name
    device_type_description: Kind = field(init=False)

    class Kind(Enum):
        ANY = 0
        HOST = 1
        NOHOST = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device_type_description",
            {
                "any": self.Kind.ANY,
                "host": self.Kind.HOST,
                "nohost": self.Kind.NOHOST,
            }[self.ndevice_type_description.string.lower()],
        )


#### allocate ####
@dataclass
class Align(Clause):
    id: ClassVar[str] = "align"
    alignment: PyExpr

@dataclass
class Allocator(Clause):
    id: ClassVar[str] = "allocator"
    allocator: PyExpr


#### metadirective ####
@dataclass
class When(Clause):
    id: ClassVar[str] = "when"
    directive: Directive
    # modifiers:
    context_selector: ContextSelector

@dataclass
class Otherwise(Clause):
    id: ClassVar[str] = "otherwise"
    directive: Directive|None = None


#### declare_variant ####
@dataclass
class AdjustArgs(DataScope):
    id: ClassVar[str] = "adjust_args"
    # modifiers:
    adjust_op_name: Name
    adjust_op: AdjustOp = field(init=False)

    class AdjustOp(Enum):
        NOTHING = 0
        NEED_DEVICE_PTR = 1
        NEED_DEVICE_ADDR = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adjust_op",
            {
                "nothing": self.AdjustOp.NOTHING,
                "need_device_ptr": self.AdjustOp.NEED_DEVICE_PTR,
                "need_device_addr": self.AdjustOp.NEED_DEVICE_ADDR,
            }[self.adjust_op_name.string.lower()],
        )

@dataclass
class AppendArgs(Clause):
    id: ClassVar[str] = "append_args"
    append_op: list[InteropModifier] = field(default_factory=list)

@dataclass
class Match(Clause):
    id: ClassVar[str] = "match"
    context_selector: ContextSelector


#### dispatch ####
@dataclass
class InteropClause(DataScope):
    id: ClassVar[str] = "interop"

@dataclass
class IsDevicePtr(DataScope):
    id: ClassVar[str] = "is_device_ptr"

@dataclass
class HasDeviceAddr(DataScope):
    id: ClassVar[str] = "has_device_addr"

@dataclass
class NoContext(Clause):
    id: ClassVar[str] = "no_context"
    dont_update_context: PyExpr

@dataclass
class NoVariants(Clause):
    id: ClassVar[str] = "no_variants"
    dont_use_variant: PyExpr


#### declare_simd ####
@dataclass
class Aligned(DataScope):
    id: ClassVar[str] = "aligned"
    # modifiers:
    alignment_modifier: PyInt|None = None

@dataclass
class Linear(DataScope):
    id: ClassVar[str] = "linear"
    # modifiers:
    step_simple_modifier: PyExpr|None = None
    step_modifier: Step|None = None
    linear_modifier_name: Name|None = None
    linear_modifier: LinearModifier|None = field(init=False)

    class LinearModifier(Enum):
        REF = 0
        UVAL = 1
        VAL = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linear_modifier",
            {
                "ref": self.LinearModifier.REF,
                "uval": self.LinearModifier.UVAL,
                "val": self.LinearModifier.VAL,
            }[self.linear_modifier_name.string.lower()]
            if self.linear_modifier_name is not None
            else None
        )

@dataclass
class Simdlen(Clause):
    id: ClassVar[str] = "simdlen"
    length: PyExpr

@dataclass
class Uniform(DataScope):
    id: ClassVar[str] = "uniform"

@dataclass
class InBranch(DataScope):
    id: ClassVar[str] = "in_branch"
    in_branch: PyExpr|None = None

@dataclass
class NotInBranch(DataScope):
    id: ClassVar[str] = "not_in_branch"
    not_in_branch: PyExpr|None = None


#### declare_target ####
@dataclass
class Enter(DataScope):
    id: ClassVar[str] = "enter"
    # modifiers:
    automap_name: Name|None = None

@dataclass
class Indirect(Clause):
    id: ClassVar[str] = "indirect"
    invoked_by_fptr: PyExpr

@dataclass
class Link(DataScope):
    id: ClassVar[str] = "link"

@dataclass
class Local(DataScope):
    id: ClassVar[str] = "local"


#### requires ####
@dataclass
class AtomicDefaultMemOrder(Clause):
    id: ClassVar[str] = "atomic_default_mem_order"
    nmemory_order: Name
    memory_order: MemoryOrder = field(init=False)

    class MemoryOrder(Enum):
        ACQ_REL = 0
        ACQUIRE = 1
        RELAXED = 2
        SEQ_CST = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            {
                "acq_rel": self.MemoryOrder.ACQ_REL,
                "acquire": self.MemoryOrder.ACQUIRE,
                "relaxed": self.MemoryOrder.RELAXED,
                "seq_cst": self.MemoryOrder.SEQ_CST,
            }[self.nmemory_order.string.lower()]
        )

@dataclass
class DynamicAllocators(Clause):
    id: ClassVar[str] = "dynamic_allocators"
    required: PyExpr|None = None

@dataclass
class ReverseOffload(Clause):
    id: ClassVar[str] = "reverse_offload"
    required: PyExpr|None = None

@dataclass
class UnifiedAddress(Clause):
    id: ClassVar[str] = "unified_address"
    required: PyExpr|None = None

@dataclass
class UnifiedSharedMemory(Clause):
    id: ClassVar[str] = "unified_shared_memory"
    required: PyExpr|None = None

@dataclass
class SelfMaps(Clause):
    id: ClassVar[str] = "self_maps"
    required: PyExpr|None = None

@dataclass
class DeviceSafesync(Clause):
    id: ClassVar[str] = "device_safesync"
    required: PyExpr|None = None


#### assume ####
@dataclass
class Absent(Clause):
    id: ClassVar[str] = "absent"
    directive_names: list[DirectiveName] = field(default_factory=list)

@dataclass
class Contains(Clause):
    id: ClassVar[str] = "contains"
    directive_names: list[DirectiveName] = field(default_factory=list)

@dataclass
class Holds(Clause):
    id: ClassVar[str] = "holds"
    hold_expr: PyExpr

@dataclass
class NoOpenmp(Clause):
    id: ClassVar[str] = "no_openmp"
    can_assume: PyExpr|None = None

@dataclass
class NoOpenmpConstructs(Clause):
    id: ClassVar[str] = "no_openmp_constructs"
    can_assume: PyExpr|None = None

@dataclass
class NoOpenmpRoutines(Clause):
    id: ClassVar[str] = "no_openmp_routines"
    can_assume: PyExpr|None = None

@dataclass
class NoParallelism(Clause):
    id: ClassVar[str] = "no_parallelism"
    can_assume: PyExpr|None = None


#### error ####
@dataclass(kw_only=True)
class At(Clause):
    id: ClassVar[str] = "at"
    naction_time: Name
    action_time: ActionTime = field(init=False)

    class ActionTime(Enum):
        COMPILATION = 0
        EXECUTION = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_time",
            {
                "compilation": self.ActionTime.COMPILATION,
                "execution": self.ActionTime.EXECUTION,
            }[self.naction_time.string.lower()]
        )

@dataclass
class Message(Clause):
    id: ClassVar[str] = "message"
    msg_string: PyExpr

@dataclass
class Severity(Clause):
    id: ClassVar[str] = "severity"
    nseverity_level: Name
    severity_level: SeverityLevel = field(init=False)

    class SeverityLevel(Enum):
        FATAL = 0
        WARNING = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity_level",
            {
                "fatal": self.SeverityLevel.FATAL,
                "warning": self.SeverityLevel.WARNING,
            }[self.nseverity_level.string.lower()]
        )


#### fuse ####
@dataclass
class LoopRange(Clause):
    id: ClassVar[str] = "looprange"
    first: PyExpr
    count: PyExpr

#### interchange ####
@dataclass
class Permutation(Clause):
    id: ClassVar[str] = "permutation"
    permutation_list: list[PyExpr] = field(default_factory=list)

#### split ####
@dataclass
class Counts(Clause):
    id: ClassVar[str] = "counts"
    count_list: list[PyExpr] = field(default_factory=list)

#### stripe ####
@dataclass
class Sizes(Clause):
    id: ClassVar[str] = "sizes"
    size_list: list[PyExpr] = field(default_factory=list)

#### unroll ####
@dataclass
class Full(Clause):
    id: ClassVar[str] = "full"
    fully_unroll: PyExpr|None = None

class Partial(Clause):
    id: ClassVar[str] = "partial"
    unroll_factor: PyExpr|None = None


#### parallel ####
@dataclass
class CopyIn(DataScope):
    id: ClassVar[str] = "copyin"

@dataclass
class NumThreads(Clause):
    id: ClassVar[str] = "num_threads"
    nthreads: PyExpr
    # modifiers:
    strict_name: Name|None = None

@dataclass
class ProcBind(Clause):
    id: ClassVar[str] = "proc_bind"
    naffinity_policy: Name
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
            }[self.naffinity_policy.string.lower()],
        )

@dataclass
class SafeSync(Clause):
    id: ClassVar[str] = "safesync"
    width: PyExpr|None = None

#### teams ####
@dataclass
class NumTeams(Clause):
    id: ClassVar[str] = "num_teams"
    upper_bound: PyExpr
    lower_bound: PyExpr|None = None

@dataclass
class ThreadLimit(Clause):
    id: ClassVar[str] = "thread_limit"
    threadlim: PyExpr


#### simd ####
@dataclass
class NonTemporal(DataScope):
    id: ClassVar[str] = "non_temporal"

@dataclass
class Order(Clause):
    id: ClassVar[str] = "non_temporal"
    ordering: Name
    # modifiers:
    order_modifier_name: Name|None = None
    order_modifier: OrderModifier|None = field(init=False)

    class OrderModifier(Enum):
        REPRODUCIBLE = 0
        UNCONSTRAINED = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "order_modifier",
            {
                "reproducible": self.OrderModifier.REPRODUCIBLE,
                "unconstrained": self.OrderModifier.UNCONSTRAINED,
            }[self.order_modifier_name.string.lower()]
            if self.order_modifier_name is not None
            else None
        )

@dataclass
class SafeLen(Clause):
    id: ClassVar[str] = "safelen"
    length: PyExpr


#### masked ####
@dataclass
class Filter(Clause):
    id: ClassVar[str] = "filter"
    thread_num: PyExpr


#### single ####
@dataclass
class CopyPrivate(DataScope):
    id: ClassVar[str] = "copyprivate"


#### for ####
@dataclass
class OrderedClause(Clause):
    id: ClassVar[str] = "ordered"
    n: PyInt|None = None

@dataclass
class Schedule(Clause):
    id: ClassVar[str] = "schedule"
    type: ScheduleType
    chunk: PyExpr|None = None
    # modifiers:
    ordering_modifier_name: Name|None = None
    ordering_modifier: OrderingModifier|None = field(init=False)
    chunk_modifier_name: Name|None = None

    class OrderingModifier(Enum):
        MONOTONIC = 0
        NON_MONOTONIC = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordering_modifier",
            {
                "monotonic": self.OrderingModifier.MONOTONIC,
                "nonmonotonic": self.OrderingModifier.NON_MONOTONIC,
            }[self.ordering_modifier_name.string.lower()]
            if self.ordering_modifier_name is not None
            else None
        )


#### distribute ####
@dataclass
class DistSchedule(Clause):
    id: ClassVar[str] = "dist_schedule"
    static: Name # = static
    chunk_size: PyExpr|None = None


#### loop ####
@dataclass
class Bind(Clause):
    id: ClassVar[str] = "bind"
    nbinding: Name
    binding: BindingKind = field(init=False)

    class BindingKind(Enum):
        PARALLEL = 0
        TEAMS = 1
        THREAD = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding",
            {
                "parallel": self.BindingKind.PARALLEL,
                "teams": self.BindingKind.TEAMS,
                "thread": self.BindingKind.THREAD,
            }[self.nbinding.string.lower()]
        )


#### taskloop ####
@dataclass
class GrainSize(Clause):
    id: ClassVar[str] = "grain_size"
    grain_size: PyExpr
    # modifiers:
    nstrict: Name|None = None

@dataclass
class NumTasks(Clause):
    id: ClassVar[str] = "num_tasks"
    num_tasks: PyExpr
    # modifiers:
    strict: Name|None = None


#### taskgraph ####
@dataclass
class GraphId(Clause):
    id: ClassVar[str] = "graph_id"
    graph_id_value: PyExpr

@dataclass
class GraphReset(Clause):
    id: ClassVar[str] = "graph_reset"
    expr: PyExpr

#### target_data ####
@dataclass
class UseDevicePtr(DataScope):
    id: ClassVar[str] = "use_device_ptr"

@dataclass
class UseDeviceAddr(DataScope):
    id: ClassVar[str] = "use_device_addr"


#### target ####
class VariableCategory(Enum):
    ALL = 0
    AGGREGATE = 1
    ALLOCATABLE = 2
    POINTER = 3
    SCALAR = 4

    @staticmethod
    def from_name(name: Name|None) -> VariableCategory|None:
        return (
            {
                "all": VariableCategory.ALL,
                "aggregate": VariableCategory.AGGREGATE,
                "allocatable": VariableCategory.ALLOCATABLE,
                "pointer": VariableCategory.POINTER,
                "scalar": VariableCategory.SCALAR,
            }[name.string.lower()]
            if name is not None
            else None
        )

@dataclass
class DefaultMap(Clause):
    id: ClassVar[str] = "use_device_addr"
    implicit_behavior_name: Name
    implicit_behavior: ImplicitBehavior = field(init=False)
    # modifiers:
    variable_category_name: Name|None = None
    variable_category: VariableCategory|None = None

    class ImplicitBehavior(Enum):
        DEFAULT = 0
        FIRST_PRIVATE = 1
        FROM = 2
        NONE = 3
        PRESENT = 4
        PRIVATE = 5
        SELF = 6
        STORAGE = 7
        TO = 8
        TOFROM = 9

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implicit_behavior",
            {
                "default ": self.ImplicitBehavior.DEFAULT,
                "first_private ": self.ImplicitBehavior.FIRST_PRIVATE,
                "from ": self.ImplicitBehavior.FROM,
                "none ": self.ImplicitBehavior.NONE,
                "present ": self.ImplicitBehavior.PRESENT,
                "private ": self.ImplicitBehavior.PRIVATE,
                "self ": self.ImplicitBehavior.SELF,
                "storage ": self.ImplicitBehavior.STORAGE,
                "to ": self.ImplicitBehavior.TO,
                "tofrom ": self.ImplicitBehavior.TOFROM,
            }[self.implicit_behavior_name.string.lower()],
        )
        object.__setattr__(self, "variable_category", VariableCategory.from_name(self.variable_category_name))

@dataclass
class UsesAllocators(Clause):
    id: ClassVar[str] = "use_device_addr"
    allocator: PyExpr
    # modifiers:
    memspace_modifier: MemSpace|None = None
    traits_modifier: Traits|None = None


#### target_update ####
@dataclass
class From(DataScope):
    id: ClassVar[str] = "from_"
    # modifiers:
    present_name: Name|None = None
    mapper_modifier: Mapper|None = None
    iterator_modifier: Iterator|None = None


@dataclass
class To(DataScope):
    id: ClassVar[str] = "to"
    # modifiers:
    present_name: Name|None = None
    mapper_modifier: Mapper|None = None
    iterator_modifier: Iterator|None = None


#### interop ####
@dataclass
class Destroy(Clause):
    id: ClassVar[str] = "destroy"
    destroy_var: PyName

@dataclass
class Init(Clause):
    id: ClassVar[str] = "init"
    init_var: PyName
    # modifiers:
    prefer_modifier: Prefer|None = None
    depinfo_modifier: DepInfo|None = None
    interop_type_modifier_name: list[Name] = field(default_factory=list)
    interop_type_modifier: list[InteropType] = field(init=False)

    class InteropType(Enum):
        TARGET = 0
        TARGETSYNC = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interop_type_modifier",
            [{
                "target": self.InteropType.TARGET,
                "targetsync": self.InteropType.TARGETSYNC,
            }[e.string.lower()] for e in self.interop_type_modifier_name]
        )

@dataclass
class Use(Clause):
    id: ClassVar[str] = "use"
    interop_var: PyName


#### critical ####
@dataclass
class Hint(Clause):
    id: ClassVar[str] = "hint"
    expr: PyExpr


#### taskgroup ####
@dataclass
class TaskReduction(DataScope):
    id: ClassVar[str] = "task_reduction"
    op: ReductionOp


#### atomic ####
@dataclass
class MemScope(Clause):
    id: ClassVar[str] = "mem_scope"
    nscope: Name
    scope: ScopeType = field(init=False)

    class ScopeType(Enum):
        ALL = 0
        CGROUP = 1
        DEVICE = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope",
            {
                "all": self.ScopeType.ALL,
                "cgroup": self.ScopeType.CGROUP,
                "device": self.ScopeType.DEVICE,
            }[self.nscope.string.lower()]
        )

@dataclass
class Read(Clause):
    id: ClassVar[str] = "read"
    use_semantics: PyExpr|None = None
@dataclass
class Update(Clause):
    id: ClassVar[str] = "update"
    use_semantics: PyExpr|None = None
@dataclass
class Write(Clause):
    id: ClassVar[str] = "write"
    use_semantics: PyExpr|None = None

@dataclass
class Capture(Clause):
    id: ClassVar[str] = "capture"
    use_semantics: PyExpr|None = None
@dataclass
class Compare(Clause):
    id: ClassVar[str] = "compare"
    use_semantics: PyExpr|None = None
@dataclass
class Fail(Clause):
    id: ClassVar[str] = "fail"
    nmem_order: Name
    mem_order: MemOrder = field(init=False)
    class MemOrder(Enum):
        ACQUIRE = 0
        RELAXED = 1
        SEQ_CST = 2
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mem_order",
            {
                "acquire": self.MemOrder.ACQUIRE,
                "relaxed": self.MemOrder.RELAXED,
                "seq_cst": self.MemOrder.SEQ_CST,
            }[self.nmem_order.string.lower()]
        )
@dataclass
class Weak(Clause):
    id: ClassVar[str] = "weak"
    use_semantics: PyExpr|None = None

@dataclass
class AcqRel(Clause):
    id: ClassVar[str] = "acq_rel"
    use_semantics: PyExpr|None = None
@dataclass
class Acquire(Clause):
    id: ClassVar[str] = "acquire"
    use_semantics: PyExpr|None = None
@dataclass
class Relaxed(Clause):
    id: ClassVar[str] = "relaxed"
    use_semantics: PyExpr|None = None
@dataclass
class Release(Clause):
    id: ClassVar[str] = "release"
    use_semantics: PyExpr|None = None
@dataclass
class SeqCst(Clause):
    id: ClassVar[str] = "seq_cst"
    use_semantics: PyExpr|None = None


#### depobj ####
@dataclass
class DepobjUpdate(Clause):
    id: ClassVar[str] = "depobj_update"
    update_var: PyName
    # modifiers:
    task_dependence_name: Name|None = None
    task_dependence: TaskDependenceKind|None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_dependence",
            TaskDependenceKind.from_name(self.task_dependence_name)
        )


#### ordered ####
@dataclass
class DoAcross(Clause):
    id: ClassVar[str] = "do_across"
    iterator_specifier: IteratorSpecifier
    # modifiers:
    dependence_type_name: Name
    dependence_type: DependenceType = field(init=False)

    class DependenceType(Enum):
        SINK = 0
        SOURCE = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dependence_type",
            {
                "sink": self.DependenceType.SINK,
                "source": self.DependenceType.SOURCE,
            }[self.dependence_type_name.string.lower()]
        )


@dataclass
class SimdClause(Clause):
    id: ClassVar[str] = "simd"
    appy_to_simd: PyExpr|None = None


@dataclass
class Threads(Clause):
    id: ClassVar[str] = "threads"
    appy_to_threads: PyExpr|None = None


########################
#### common clauses ####
########################

@dataclass
class Apply(Clause):
    id: ClassVar[str] = "apply"
    directives: list[Name] = field(default_factory=list)
    # modifiers:
    loop_modifier: LoopModifier|None = None


class TaskDependenceKind(Enum):
    DEPOBJ = 0
    IN = 1
    INOUT = 2
    INOUTSET = 3
    MUTEXINOUTSET = 4
    OUT = 5

    @staticmethod
    def from_name(name: Name|None) -> TaskDependenceKind|None:
        return (
            {
                "depobj": TaskDependenceKind.DEPOBJ,
                "in": TaskDependenceKind.IN,
                "inout": TaskDependenceKind.INOUT,
                "inoutset": TaskDependenceKind.INOUTSET,
                "mutexinoutset": TaskDependenceKind.MUTEXINOUTSET,
                "out": TaskDependenceKind.OUT,
            }[name.string.lower()]
            if name is not None
            else None
        )


@dataclass
class Depend(Clause):
    id: ClassVar[str] = "depend"
    locator_list: list[PyExpr] = field(default_factory=list)
    # modifiers:
    task_dependence_name: Name|None = None
    task_dependence: TaskDependenceKind|None = field(init=False)
    iterator_modifier: Iterator|None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_dependence",
            TaskDependenceKind.from_name(self.task_dependence_name)
        )


@dataclass
class Device(Clause):
    id: ClassVar[str] = "device"
    device_description: PyExpr

    # modifiers:
    device_modifier_name: Name|None = None
    device_modifier: DeviceModifier|None = field(init=False)

    class DeviceModifier(Enum):
        ANCESTOR = 0
        DEVICE_NUM = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device_modifier",
            {
                "ancestor": self.DeviceModifier.ANCESTOR,
                "device_num": self.DeviceModifier.DEVICE_NUM,
            }[self.device_modifier_name.string.lower()]
            if self.device_modifier_name is not None
            else None
        )


@dataclass
class Default(Clause):
    id: ClassVar[str] = "default"
    data_sharing_attr_name: Name
    data_sharing_attr: DataSharingAttr = field(init=False)
    # modifiers:
    variable_category_name: Name|None = None
    variable_category: VariableCategory|None = None

    class DataSharingAttr(Enum):
        SHARED = 0
        FIRST_PRIVATE = 1
        PRIVATE = 2
        NONE = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_sharing_attr",
            {
                "shared": self.DataSharingAttr.SHARED,
                "firstprivate": self.DataSharingAttr.FIRST_PRIVATE,
                "private": self.DataSharingAttr.PRIVATE,
                "none": self.DataSharingAttr.NONE,
            }[self.data_sharing_attr_name.string.lower()],
        )
        object.__setattr__(
            self,
            "variable_category",
            VariableCategory.from_name(self.variable_category_name)
        )


@dataclass
class Private(DataScope):
    id: ClassVar[str] = "private"


@dataclass
class If(Clause):
    id: ClassVar[str] = "if_"
    expr: PyExpr


@dataclass
class FirstPrivate(DataScope):
    id: ClassVar[str] = "first_private"
    # modifiers:
    saved_name: Name|None = None


@dataclass
class Reduction(DataScope):
    id: ClassVar[str] = "reduction"
    op: ReductionOp
    # modifiers:
    reduction_modifier_name: Name|None = None
    reduction_modifier: ReductionModifier|None = field(init=False)
    original_modifier: Original|None = None

    class ReductionModifier(Enum):
        DEFAULT = 0
        INSCAN = 1
        TASK = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reduction_modifier",
            {
                "default": self.ReductionModifier.DEFAULT,
                "inscan": self.ReductionModifier.INSCAN,
                "task": self.ReductionModifier.TASK,
            }[self.reduction_modifier_name.string.lower()]
            if self.reduction_modifier_name is not None
            else None
        )


@dataclass
class Induction(DataScope):
    id: ClassVar[str] = "induction"
    op: InductionOp
    # modifiers:
    step_modifier: Step
    induction_modifier_name: Name|None = None
    induction_modifier: InductionModifier|None = field(init=False)

    class InductionModifier(Enum):
        RELAXED = 0
        STRICT = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "induction_modifier",
            {
                "relaxed": self.InductionModifier.RELAXED,
                "strict": self.InductionModifier.STRICT,
            }[self.induction_modifier_name.string.lower()]
            if self.induction_modifier_name is not None
            else None
        )


@dataclass
class Shared(DataScope):
    id: ClassVar[str] = "shared"


@dataclass
class Collapse(Clause):
    id: ClassVar[str] = "collapse"
    num: PyInt


@dataclass
class LastPrivate(DataScope):
    id: ClassVar[str] = "last_private"
    # modifiers:
    conditional_name: Name|None = None


@dataclass
class AllocateClause(DataScope):
    id: ClassVar[str] = "allocate"
    # modifiers:
    allocator_simple_modifier: PyExpr|None = None
    allocator_modifier: AllocatorModifier|None = None
    align_modifier: AlignModifier|None = None


@dataclass
class NoWait(Clause):
    id: ClassVar[str] = "no_wait"
    dont_synchronize: PyExpr|None = None


@dataclass
class Final(Clause):
    id: ClassVar[str] = "final"
    finalize: PyExpr


@dataclass
class Mergeable(Clause):
    id: ClassVar[str] = "mergeable"
    can_merge: PyExpr|None = None


@dataclass
class Untied(Clause):
    id: ClassVar[str] = "untied"
    can_change_threads: PyExpr|None = None


@dataclass
class Affinity(DataScope):
    id: ClassVar[str] = "affinity"
    # modifiers:
    iterator_modifier: Iterator|None = None


@dataclass
class Detach(Clause):
    id: ClassVar[str] = "detach"
    event_handle: PyName


@dataclass
class InReduction(DataScope):
    id: ClassVar[str] = "in_reduction"
    op: ReductionOp


@dataclass
class Priority(Clause):
    id: ClassVar[str] = "priority"
    value: PyExpr


@dataclass
class Replayable(Clause):
    id: ClassVar[str] = "affinity"
    expr: PyExpr


@dataclass
class ThreadSet(Clause):
    id: ClassVar[str] = "thread_set"
    nset: Name
    set: ThreadSetType = field(init=False)

    class ThreadSetType(Enum):
        OMP_POOL = 0
        OMP_TEAM = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "set",
            {
                "omp_pool": self.ThreadSetType.OMP_POOL,
                "omp_team": self.ThreadSetType.OMP_TEAM,
            }[self.nset.string.lower()]
        )


@dataclass
class Transparent(Clause):
    id: ClassVar[str] = "transparent"
    impex_type: PyExpr|None = None


@dataclass
class NoGroup(Clause):
    id: ClassVar[str] = "no_group"
    dont_synchronize: PyExpr|None = None


@dataclass
class Map(DataScope):
    id: ClassVar[str] = "map"
    # modifiers:
    always_modifier_name: Name|None = None
    close_modifier_name: Name|None = None
    present_modifier_name: Name|None = None
    self_modifier_name: Name|None = None
    delete_modifier_name: Name|None = None

    ref_modifier_name: Name|None = None
    ref_modifier: RefModifier|None = field(init=False)

    map_type_name: Name|None = None
    map_type: MapType|None = field(init=False)

    mapper_modifier: Mapper|None = None
    iterator_modifier: Iterator|None = None

    class RefModifier(Enum):
        REF_PTEE = 0
        REF_PTR = 1
        REF_PTR_PTEE = 2

    class MapType(Enum):
        FROM = 0
        STORAGE = 1
        TO = 2
        TOFROM = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ref_modifier",
            {
                "ref_ptee": self.RefModifier.REF_PTEE,
                "ref_ptr": self.RefModifier.REF_PTR,
                "ref_ptr_ptee": self.RefModifier.REF_PTR_PTEE,
            }[self.ref_modifier_name.string.lower()]
            if self.ref_modifier_name is not None
            else None
        )
        object.__setattr__(
            self,
            "map_type",
            {
                "from": self.MapType.FROM,
                "storage": self.MapType.STORAGE,
                "to": self.MapType.TO,
                "tofrom": self.MapType.TOFROM,
            }[self.map_type_name.string.lower()]
            if self.map_type_name is not None
            else None
        )


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

@dataclass
class InductionOp(Modifier):
    id: ClassVar[str] = "op"
    value: str


@dataclass
class Original(Modifier):
    id: ClassVar[str] = "original_modifier"
    name: Name # = original
    sharing_name: Name
    sharing: Sharing = field(init=False)

    class Sharing(Enum):
        DEFAULT = 0
        PRIVATE = 1
        SHARED = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sharing",
            {
                "default": self.Sharing.DEFAULT,
                "private": self.Sharing.PRIVATE,
                "shared": self.Sharing.SHARED,
            }[self.sharing_name.string.lower()]
        )


@dataclass
class InteropModifier(Modifier):
    id: ClassVar[str] = "iterop_modifier"
    name: Name # = interop
    nkind: list[Name] = field(default_factory=list)
    kind: list[Kind] = field(init=False)

    class Kind(Enum):
        TARGET = 0
        TARGETSYNC = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            [{
                "target": self.Kind.TARGET,
                "targetsync": self.Kind.TARGETSYNC,
            }[e.string.lower()] for e in self.nkind]
        )


@dataclass
class Iterator(Modifier):
    id: ClassVar[str] = "iterator_modifier"
    name: Name # = iterator
    specifiers: list[IteratorSpecifier] = field(default_factory=list)

@dataclass
class IteratorSpecifier(Modifier):
    id: ClassVar[str] = "iterator_specifier"
    name: PyName
    begin: PyExpr
    end: PyExpr
    step: PyExpr|None = None


@dataclass
class Step(Modifier):
    id: ClassVar[str] = "step_modifier"
    name: Name
    expr: PyExpr


@dataclass
class AllocatorModifier(Modifier):
    id: ClassVar[str] = "allocator_modifier"
    name: Name # = allocator
    allocator: PyExpr


@dataclass
class AlignModifier(Modifier):
    id: ClassVar[str] = "align_modifier"
    name: Name # = align
    alignment: PyExpr


@dataclass
class Mapper(Modifier):
    id: ClassVar[str] = "mapper_modifier"
    name: Name # = mapper
    identifier: PyName

@dataclass
class MemSpace(Modifier):
    id: ClassVar[str] = "memspace_modifier"
    name: Name # = memspace
    handle: PyExpr

@dataclass
class Traits(Modifier):
    id: ClassVar[str] = "traits_modifier"
    name: Name # = traits
    traits: PyExpr


@dataclass
class DepInfo(Modifier):
    id: ClassVar[str] = "depinfo_modifier"
    name: Name
    kind: Kind = field(init=False)
    locator_list: list[PyName] = field(default_factory=list)

    class Kind(Enum):
        IN = 0
        INOUT = 1
        INOUTSET = 2
        MUTEXINOUTSET = 3
        OUT = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            {
                "in": self.Kind.IN,
                "inout": self.Kind.INOUT,
                "inoutset": self.Kind.INOUTSET,
                "mutexinoutset": self.Kind.MUTEXINOUTSET,
                "out": self.Kind.OUT,
            }[self.name.string.lower()]
        )


@dataclass
class LoopModifier(Modifier):
    id: ClassVar[str] = "loop_modifier"
    name: Name
    kind: Kind = field(init=False)
    indices: list[PyExpr] = field(default_factory=list)

    class Kind(Enum):
        FUSED = 0
        GRID = 1
        IDENTITY = 2
        INTERCHANGED = 3
        INTRATILE = 4
        OFFSETS = 5
        REVERSED = 6
        SPLIT = 7
        UNROLLED = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            {
                "fused": self.Kind.FUSED,
                "grid": self.Kind.GRID,
                "identity": self.Kind.IDENTITY,
                "interchanged": self.Kind.INTERCHANGED,
                "intratile": self.Kind.INTRATILE,
                "offsets": self.Kind.OFFSETS,
                "reversed": self.Kind.REVERSED,
                "split": self.Kind.SPLIT,
                "unrolled": self.Kind.UNROLLED,
            }[self.name.string.lower()]
        )


@dataclass
class Prefer(Modifier):
    id: ClassVar[str] = "prefer_modifier"
    name: Name # = prefer_type
    spec: list[list[FrSelector|AttrSelector] | PyName] = field(default_factory=list)

@dataclass
class FrSelector(Modifier):
    name: Name
    identifier: PyName

@dataclass
class AttrSelector(Modifier):
    name: Name
    expr_list: list[PyExpr]


# TODO: this uses context_selector as a stmt_list, which is not exactly what the standard required
@dataclass
class ContextSelector(Modifier):
    id: ClassVar[str] = "context_selector"
    stmt_list: list[PyStmt] = field(default_factory=list)


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
