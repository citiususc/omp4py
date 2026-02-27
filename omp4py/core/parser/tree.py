"""Parser nodes for directives, clauses, and expressions.

An OpenMP directive consists of a construct and zero or more clauses. Each
keyword within the directive represents either the construct or a clause,
and together they form the directive's elements. Both constructs and clauses
can include modifiers, which are arguments that further define their behavior.
"""

from __future__ import annotations

from ast import alias, arg, expr, keyword, pattern, stmt, type_param
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

__all__ = [
    "Clause",
    "Collapse",
    "Construct",
    "DataScope",
    "DeclareReduction",
    "Default",
    "Directive",
    "FirstPrivate",
    "For",
    "If",
    "LastPrivate",
    "Modifier",
    "Name",
    "NoWait",
    "NumThreads",
    "OmpNode",
    "Ordered",
    "Parallel",
    "ParallelFor",
    "Private",
    "PyExpr",
    "PyInt",
    "PyName",
    "PyStmt",
    "Reduction",
    "ReductionOp",
    "Schedule",
    "ScheduleType",
    "Shared",
    "Span",
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


@dataclass
class Directive(OmpNode):
    string: str
    construct: Construct


#######################################################################################################################
##################################################### Constructs ######################################################
#######################################################################################################################


@dataclass
class Construct(OmpNode):
    name: Name


@dataclass
class DeclareReduction(Construct):
    id: ReductionOp
    ann_list: list[PyExpr]
    combiner: Combiner
    initializer: Initializer | None = None


@dataclass
class Parallel(Construct):
    default: Default | None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    if_: If | None = None
    num_threads: NumThreads | None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)


@dataclass
class For(Construct):
    collapse: Collapse | None = None
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[LastPrivate] = field(default_factory=list)
    no_wait: NoWait | None = None
    ordered: Ordered | None = None
    private: list[Private] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)
    schedule: Schedule | None = None


#######################################################################################################################
################################################# Combined Constructs #################################################
#######################################################################################################################


@dataclass
class ParallelFor(Construct):
    parallel: Parallel
    for_: For


#######################################################################################################################
####################################################### Clauses #######################################################
#######################################################################################################################


@dataclass
class Clause(OmpNode):
    id: ClassVar[str] = "clause"  # must be redefined
    name: Name


@dataclass
class DataScope(Clause):
    targets: list[PyName]

    @property
    def str_targets(self) -> list[str]:
        return [v.string for v in self.targets]


@dataclass
class Collapse(Clause):
    id: ClassVar[str] = "collapse"
    num: PyInt


@dataclass
class Combiner(Clause):
    id: ClassVar[str] = "combiner"
    stmt: PyStmt


@dataclass
class Default(Clause):
    id: ClassVar[str] = "default"
    ntype: Name

    class Type(Enum):
        SHARED = 0
        FIRST_PRIVATE = 1
        PRIVATE = 2
        NONE = 3

    type: Type = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            {
                "shared": self.Type.SHARED,
                "firstprivate": self.Type.FIRST_PRIVATE,
                "private": self.Type.PRIVATE,
                "none": self.Type.NONE,
            }[self.ntype.string.lower()],
        )


@dataclass
class FirstPrivate(DataScope):
    id: ClassVar[str] = "first_private"


@dataclass
class If(Clause):
    id: ClassVar[str] = "if_"
    expr: PyExpr


@dataclass
class Initializer(Clause):
    id: ClassVar[str] = "initializer"
    stmt: PyStmt


@dataclass
class LastPrivate(DataScope):
    id: ClassVar[str] = "last_private"


@dataclass
class NoWait(Clause):
    id: ClassVar[str] = "no_wait"
    expr: PyExpr | None


@dataclass
class NumThreads(Clause):
    id: ClassVar[str] = "num_threads"
    expr: PyExpr


@dataclass
class Ordered(Clause):
    id: ClassVar[str] = "ordered"
    n: PyInt | None


@dataclass
class Private(DataScope):
    id: ClassVar[str] = "private"


@dataclass
class Reduction(DataScope):
    id: ClassVar[str] = "reduction"
    op: ReductionOp


@dataclass
class Schedule(Clause):
    id: ClassVar[str] = "schedule"
    type: ScheduleType
    chunk: PyExpr | None = None


@dataclass
class Shared(DataScope):
    id: ClassVar[str] = "shared"


#######################################################################################################################
###################################################### Modifiers ######################################################
#######################################################################################################################


@dataclass
class Modifier(OmpNode):
    id: ClassVar[str] = "modifier"  # must be redefined


@dataclass
class ReductionOp(Modifier):
    id: ClassVar[str] = "op"
    value: str


@dataclass
class ScheduleType(Modifier):
    id: ClassVar[str] = "type"
    name: Name

    class Kind(Enum):
        STATIC = 0
        DYNAMIC = 1
        GUIDED = 2
        AUTO = 3
        RUNTIME = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            {
                "static": self.Kind.STATIC,
                "dynamic": self.Kind.DYNAMIC,
                "guided": self.Kind.GUIDED,
                "auto": self.Kind.AUTO,
                "runtime": self.Kind.RUNTIME,
            }[self.name.string.lower()],
        )

    kind: Kind = field(init=False)


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
