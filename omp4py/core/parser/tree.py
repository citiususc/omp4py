"""Parser nodes for directives, clauses, and expressions.

An OpenMP directive consists of a construct and zero or more clauses. Each
keyword within the directive represents either the construct or a clause,
and together they form the directive's elements. Both constructs and clauses
can include modifiers, which are arguments that further define their behavior.
"""

from ast import Constant, alias, arg, expr, keyword, pattern, stmt, type_param
from dataclasses import dataclass


@dataclass
class Span:
    lineno: int
    offset: int
    end_lineno: int = -1
    end_offset: int = -1

    @staticmethod
    def from_ast(
        node: expr | stmt | arg | keyword | alias | pattern | type_param,
    ) -> "Span":
        return Span(
            node.lineno,
            node.col_offset,
            node.end_lineno if node.end_lineno is not None else -1,
            node.end_col_offset if node.end_col_offset is not None else -1,
        )


class OmpNode:
    span: Span

@dataclass
class Construct(OmpNode):
    name: str

@dataclass
class Directive(OmpNode):
    construct: Construct

@dataclass
class Clause(OmpNode):
    name: str


@dataclass
class Modifier(OmpNode):
    name: str


@dataclass
class PyExpr(Modifier):
    """Represents a Python expression in the AST.

    Attributes:
        value (expr): The AST node representing the expression.
    """

    value: expr


@dataclass
class PyLiteral(PyExpr):
    """Represents a literal Python expression.

    Extends:
        PyExpr

    Attributes:
        value (Constant): An AST Constant node representing a literal value
            (e.g., numbers, strings, booleans, None).

    """

    value: Constant


@dataclass
class PyStmt(Modifier):
    """Represents a Python statement in the AST.

    Attributes:
        value (stmt): The AST node representing the statement.
    """

    value: stmt


@dataclass
class PyID(Modifier):
    value: str


@dataclass
class NumThreads(Clause):
    value: PyExpr


@dataclass
class If(Clause):
    value: PyExpr


@dataclass
class NoWait(Clause):
    value: PyExpr


class DataScope(Clause):
    targets: list[PyID]

    @property
    def str_targets(self) -> list[str]:
        return [v.value for v in self.targets]


class Reduction(DataScope):
    pass


####


@dataclass
class Parallel(Construct):
    num_threads: NumThreads | None
    private: list[DataScope] | None
    shared: list[DataScope] | None
    first_private: list[DataScope] | None
    reduction: list[Reduction] | None


@dataclass
class For(Construct):
    private: list[DataScope] | None
    first_private: list[DataScope] | None
    last_private: list[DataScope] | None
    reduction: list[Reduction] | None
