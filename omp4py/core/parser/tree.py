"""Parser nodes for directives, clauses, and expressions.

An OpenMP directive consists of a construct and zero or more clauses. Each
keyword within the directive represents either the construct or a clause,
and together they form the directive's elements. Both constructs and clauses
can include modifiers, which are arguments that further define their behavior.
"""

from __future__ import annotations

import ast
from ast import Constant, alias, arg, expr, keyword, pattern, stmt, type_param
from dataclasses import dataclass, field
from typing import ClassVar

__all__ = [
    "Clause",
    "Construct",
    "Construct",
    "Directive",
    "Modifier",
    "Name",
    "OmpNode",
    "Parallel",
    "PyExpr",
    "PyInt",
    "PyName",
    "PyStmt",
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


@dataclass
class OmpNode:
    span: Span


@dataclass
class Name(OmpNode):
    value: str

    def __str__(self):
        return self.value

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
class Parallel(Construct):
    num_threads: NumThreads | None = None
    private: list[Private] = field(default_factory=list)
    shared: list[Shared] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)


@dataclass
class For(Construct):
    private: list[Private] = field(default_factory=list)
    first_private: list[FirstPrivate] = field(default_factory=list)
    last_private: list[DataScope] = field(default_factory=list)
    reduction: list[Reduction] = field(default_factory=list)


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
        return [v.value for v in self.targets]


@dataclass
class Collapse(Clause):
    id: ClassVar[str] = "collapse"
    value: PyInt


@dataclass
class Default(DataScope):
    id: ClassVar[str] = "default"


@dataclass
class FirstPrivate(DataScope):
    id: ClassVar[str] = "first_private"


@dataclass
class If(Clause):
    id: ClassVar[str] = "if"
    value: PyExpr


@dataclass
class LastPrivate(DataScope):
    id: ClassVar[str] = "last_private"


@dataclass
class NoWait(Clause):
    id: ClassVar[str] = "no_wait"
    value: PyExpr | None


@dataclass
class NumThreads(Clause):
    id: ClassVar[str] = "num_threads"
    value: PyExpr | None


@dataclass
class Ordered(Clause):
    id: ClassVar[str] = "ordered"


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
    type: str

@dataclass
class ScheduleType(Modifier):
    id: ClassVar[str] = "type"
    name: Name
    kind: str

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


@dataclass
class PyName(Modifier):
    """Represents a Python name in the AST.

    Attributes:
        value (stmt): The AST node representing the name.
    """

    value: str


@dataclass
class PyStmt(Modifier):
    """Represents a Python statement in the AST.

    Attributes:
        value (stmt): The AST node representing the statement.
    """

    value: stmt
