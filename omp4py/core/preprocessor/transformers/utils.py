"""Utility helpers for AST transformations in the `omp4py` preprocessor.

This module provides a collection of helper functions used across the
preprocessor to simplify AST manipulation and traversal.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Generator

__all__ = ["ControlFlow", "check_omp_if", "fix_body_locations", "scope_if", "unpack_if", "walk"]


def walk(node: ast.AST, descend: Callable[[ast.AST], bool] = lambda _: True) -> Generator[ast.AST]:
    """Recursively walk the AST starting from `node`, with optional control over descent.

    This function behaves similarly to `ast.walk`, but allows the caller to
    decide whether to recurse into the children of a given node.

    Args:
        node (ast.AST): The root AST node to start walking from.
        descend (Callable[[ast.AST], bool], optional): A function that takes an
            AST node and returns True if the walker should recurse into its
            children, or False to skip them. Defaults to a function that always returns True.

    Yields:
        ast.AST: Each node in the AST, in pre-order traversal (parent before children).
    """
    yield node
    if descend(node):
        for child in ast.iter_child_nodes(node):
            yield from walk(child, descend)


def fix_body_locations(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply `ast.fix_missing_locations` to each statement in a body.

    Args:
        body (list[ast.stmt]): A list of AST statement nodes representing
            a syntactic body (e.g., the body of a function, class, or module).

    Returns:
        list[ast.stmt]: A new list of statements where each node has had
            missing location information filled in recursively.
    """
    return [ast.fix_missing_locations(node) for node in body]


def unpack_if(body: list[ast.stmt]) -> ast.If:
    """Wrap a list of statements in a temporary `if` block.

    It is used to temporarily group multiple statements into a
    single block so they can be processed uniformly by
    transformation passes.

    Args:
        body (list[ast.stmt]): List of statements to group.

    Returns:
        ast.If: Synthetic `if` node wrapping the given body.
    """
    return ast.If(ast.Constant("_omp_: unpack"), body)


def scope_if(body: list[ast.stmt], names: list[str]) -> ast.If:
    """Create a temporary scope for a list of statements using an `if` block.

    This is used to differentiate private variables from shared variables via data clauses,
    enabling the detection and marking of argument variables as private within the block.

    Args:
        body (list[ast.stmt]): The list of statements to wrap.
        names (list[str]): A list of variable names to be marked as private in the scope.

    Returns:
        ast.If: A synthetic `if` node wrapping the given body.
    """
    return ast.If(ast.Constant("_omp_: scope : " + " ".join(names)), body)


def check_omp_if(if_: ast.If) -> list[str] | None:
    """Check whether an `if` node is an omp special funcion.

    This function detects `if` nodes created by special omp funcions
    by checking for the special marker condition.

    Args:
        if_ (ast.If): AST node to check.

    Returns:
        list[str] | None: None if the node is normal `if`, list[str] with
          function arguments in other case.
    """
    match if_.test:
        case ast.Constant(value=str(value)) if value.startswith("_omp_"):
            return [arg.strip() for arg in value.split(":")[1:]]
    return None


class ControlFlow(ast.NodeVisitor):
    """Visitor to analyze control flow structures within an AST node.

    Tracks whether statements (such as break, continue, return, or yield)
    occur within valid loops or scopes.

    Attributes:
        in_loop (bool): Flag indicating whether the visitor is currently inside a loop.
        found (ast.stmt | ast.expr | None): The specific node found that violates
            or triggers the control flow condition.
    """

    in_loop: bool
    found: ast.stmt | ast.expr | None

    def __init__(self) -> None:
        """Initialize the control flow visitor state."""
        self.in_loop = False
        self.found = None

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Skip nested function definitions to prevent crossing function boundaries."""

    def visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        """Track loop entry and exit to validate loop-dependent statements.

        Args:
            node (ast.For | ast.AsyncFor | ast.While): The loop statement node.
        """
        in_loop, self.in_loop = self.in_loop, True
        self.generic_visit(node)
        self.in_loop = in_loop

    def visit_loop_keys(self, node: ast.Break | ast.Continue) -> None:
        """Detect `break` or `continue` statements occurring outside a loop.

        Args:
            node (ast.Break | ast.Continue): The break or continue node.
        """
        if not self.in_loop:
            self.found = node

    def visit_def_keys(self, node: ast.Return | ast.Yield | ast.YieldFrom) -> None:
        """Detect `return`, `yield`, or `yield from` statements.

        Args:
            node (ast.Return | ast.Yield | ast.YieldFrom): The definition key node.
        """
        self.found = node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_function
    visit_For = visit_AsyncFor = visit_While = visit_loop
    visit_Break = visit_Continue = visit_loop_keys
    visit_Return = visit_Yield = visit_YieldFrom = visit_def_keys
