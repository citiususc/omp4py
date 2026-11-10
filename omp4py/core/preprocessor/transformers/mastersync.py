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

from omp4py.core.parser.tree import Barrier, Critical, Master, Ordered  # noqa: TC001 required
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, StructuredBlock, construct, with_block

__all__ = []


@construct.register
def _(ctr: Barrier, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `barrier` construct.

    Inserts a runtime barrier call that forces all threads in the current
    team to synchronize before continuing execution.

    Args:
        ctr (Barrier): Parsed `barrier` construct.
        block (StructuredBlock | None): Structured block of the barrier region if was defined using with.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    barrier: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("barrier"))))
    ast.fix_missing_locations(barrier)

    return [barrier] + ([block] if block is not None else [])


@construct.register
def _(ctr: Critical, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `critical` construct.

    Wraps the body with runtime lock/unlock calls to ensure mutual
    exclusion, allowing only one thread at a time to execute the
    critical section.

    Args:
        ctr (Critical): Parsed `critical` construct.
        block (StructuredBlock | None): Structured block of the critical region.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    block = with_block(ctr, block, ctx)
    lock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_lock"))))
    unlock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_unlock"))))

    ast.fix_missing_locations(lock)
    ast.fix_missing_locations(unlock)

    return [lock, *block.body, unlock]


@construct.register
def _(ctr: Master, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `master` construct.

    Ensures that the enclosed block is executed only by the master
    thread of the team, without implicit synchronization.

    Args:
        ctr (Master): Parsed `master` construct.
        block (StructuredBlock | None): Structured block of the master region.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    block = with_block(ctr, block, ctx)
    master: ast.If = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("master"))))

    ast.fix_missing_locations(master)
    master.body = [block]

    return [master]


@construct.register
def _(ctr: Ordered, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `ordered` construct.

    Enforces a sequential execution order within a parallel loop by
    inserting a runtime call that delimit ordered region.

    Args:
        ctr (Ordered): Parsed `ordered` construct.
        block (StructuredBlock | None): Structured block of the ordered region if was defined using with.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    block = with_block(ctr, block, ctx)
    init: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_start"))))
    end: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_next"))))

    ast.fix_missing_locations(init)
    ast.fix_missing_locations(end)

    return [init, block, end]
