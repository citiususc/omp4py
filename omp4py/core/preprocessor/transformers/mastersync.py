"""Master and synchronization constructs for the `omp4py` preprocessor.

This module implements OpenMP synchronization constructs that coordinate
execution among threads within a parallel region.

It provides transformations for:

- `barrier`: Global synchronization point for all threads
- `critical`: Mutual exclusion region protected by a lock
- `master`: Code executed only by the master thread
- `ordered`: Enforced execution order within parallel loops

Each construct is translated into calls to runtime primitives that handle
synchronization semantics, ensuring correct behavior under concurrent
execution.

All constructs are registered through the `construct` dispatcher and
applied during the transformation phase of the preprocessor.
"""

from __future__ import annotations

import ast
import typing

from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct

if typing.TYPE_CHECKING:
    from omp4py.core.parser.tree import Barrier, Critical, Master, Ordered

__all__ = []


@construct.register
def _(ctr: Barrier, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `barrier` construct.

    Inserts a runtime barrier call that forces all threads in the current
    team to synchronize before continuing execution.

    Args:
        ctr (Barrier): Parsed `barrier` construct.
        body (list[ast.stmt]): Statements following the barrier.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    barrier: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("barrier"))))
    ast.fix_missing_locations(barrier)

    return [barrier, *body]


@construct.register
def _(ctr: Critical, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `critical` construct.

    Wraps the body with runtime lock/unlock calls to ensure mutual
    exclusion, allowing only one thread at a time to execute the
    critical section.

    Args:
        ctr (Critical): Parsed `critical` construct.
        body (list[ast.stmt]): Body of the critical section.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    lock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_lock"))))
    unlock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_unlock"))))

    ast.fix_missing_locations(lock)
    ast.fix_missing_locations(unlock)

    return [lock, *body, unlock]


@construct.register
def _(ctr: Master, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `master` construct.

    Ensures that the enclosed block is executed only by the master
    thread of the team, without implicit synchronization.

    Args:
        ctr (Master): Parsed `master` construct.
        body (list[ast.stmt]): Body to execute.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    master: ast.If = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("master"))))

    ast.fix_missing_locations(master)
    master.body = body

    return [master]


@construct.register
def _(ctr: Ordered, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `ordered` construct.

    Enforces a sequential execution order within a parallel loop by
    inserting runtime calls that delimit ordered regions.

    Args:
        ctr (Ordered): Parsed `ordered` construct.
        body (list[ast.stmt]): Ordered region body.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    init: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_init"))))
    end: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_end"))))

    ast.fix_missing_locations(init)
    ast.fix_missing_locations(end)

    return [init, *body, end]
