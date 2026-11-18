"""Master and synchronization constructs for the `omp4py` preprocessor.

This module implements OpenMP synchronization constructs that coordinate
execution among threads within a parallel region.

It provides transformations for:

- `barrier`: Global synchronization point for all threads
- `critical`: Mutual exclusion region protected by a lock
- `master`: Code executed only by the master thread
- `masked`: Code executed only by the selected thread

Each construct is translated into calls to runtime primitives that handle
synchronization semantics, ensuring correct behavior under concurrent
execution.

All constructs are registered through the `construct` dispatcher and
applied during the transformation phase of the preprocessor.
"""

from __future__ import annotations

import ast

from omp4py.core.parser.tree import Barrier, Critical, Masked, Master  # noqa: TC001 required
from omp4py.core.preprocessor.transformers.symtable import omp_name, runtime_ast
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
    if ctr.critical_name is None:
        lock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_lock"))))
        unlock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_unlock"))))

    else:
        if (critical_name := ctx.module_storage.named_mutex.get(ctr.critical_name.string)) is None:
            critical_name: str = omp_name(ctx, "mutex")
            critical_get: ast.AnnAssign = ctr.span.to_ast(
                ast.AnnAssign(
                    ast.Name(critical_name, ast.Store()),
                    runtime_ast("Mutex"),
                    ast.Call(runtime_ast("critical_named"), [ast.Constant(critical_name)]),
                    simple=1,
                ),
            )

            ctx.finalizers.append(lambda: ctx.module.body.insert(0, critical_get))

        lock: ast.stmt = ctr.span.to_ast(
            ast.Expr(ast.Call(runtime_ast("critical_named_lock"), [ast.Name(critical_name)])),
        )
        unlock: ast.stmt = ctr.span.to_ast(
            ast.Expr(ast.Call(runtime_ast("critical_named_unlock"), [ast.Name(critical_name)])),
        )

    ast.fix_missing_locations(lock)
    ast.fix_missing_locations(unlock)

    return [lock, block, unlock]


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
    master: ast.If = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("masked"), [ast.Constant(0)])))

    ast.fix_missing_locations(master)
    master.body = [block]

    return [master]


@construct.register
def _(ctr: Masked, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `masted` construct.

    Ensures that the enclosed block is executed only by the selected
    thread of the team, without implicit synchronization.

    Args:
        ctr (Master): Parsed `masked` construct.
        block (StructuredBlock | None): Structured block of the master region.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.
    """
    block = with_block(ctr, block, ctx)

    select: ast.expr = ctr.filter.thread.value if ctr.filter else ast.Constant(0)

    masked: ast.If = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("masked"), [select])))

    ast.fix_missing_locations(masked)
    masked.body = [block]

    return [masked]
