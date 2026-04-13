"""Parallel region constructs for the `omp4py` preprocessor.

This module implements the OpenMP `parallel` construct, which defines
a region of code to be executed by multiple threads.

The transformation rewrites a parallel region into a callable function
and a corresponding runtime invocation that manages thread creation,
execution, and synchronization.

It handles:

- Creation of a new variable scope with OpenMP data-sharing semantics
- Evaluation of clauses such as `if`, `num_threads`, `proc_bind`, and `copyin`
- Integration with runtime primitives via generated AST calls
- Validation of threadprivate variables for `copyin`

All constructs are registered through the `construct` dispatcher and
applied during the transformation phase of the preprocessor.
"""

from __future__ import annotations

import ast
import typing

from omp4py.core.preprocessor.transformers.scopes import check_scopes, create_scope
from omp4py.core.preprocessor.transformers.symtable import omp_name, runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx
from omp4py.core.preprocessor.transformers.utils import fix_body_locations

if typing.TYPE_CHECKING:
    from omp4py.core.parser.tree import Parallel

__all__ = []


@construct.register
def _(ctr: Parallel, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `parallel` construct.

    This transformation encapsulates the parallel region body into a
    generated function and emits a runtime call to execute it in parallel.

    It also processes and validates the following clauses:

    - `if`: Conditional parallel execution
    - `num_threads`: Number of threads to use
    - `proc_bind`: Thread affinity policy
    - `copyin`: Initialization of threadprivate variables

    A new variable scope is created to enforce OpenMP data-sharing rules,
    including handling of `shared`, `private`, `firstprivate`, and
    `reduction` clauses.

    Args:
        ctr (Parallel): Parsed `parallel` construct.
        body (list[ast.stmt]): Body of the parallel region.
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements including the generated
        function and runtime invocation.
    """
    f_name: str = omp_name(ctx, "parallel")
    f_ast: ast.FunctionDef = ctr.span.to_ast(ast.FunctionDef(f_name, ast.arguments(), body=body))

    create_scope(ctr, ctx, f_ast, ctr.default, ctr.shared, ctr.private, ctr.first_private, ctr.reduction)

    if_ast: ast.expr = ast.Constant(True)
    num_threads_ast = ast.Tuple()
    proc_bind_ast = ast.Constant(0)
    copyin_ast = ast.Tuple()

    if ctr.if_:
        if_ast = ctr.if_.span.to_ast(ast.Call(ast.Name("bool"), [ctr.if_.expr.value]))

    if ctr.num_threads:
        expr: ast.expr = ctr.num_threads.expr.value
        match expr:
            case ast.Constant(value=value) if isinstance(value, int):
                num_threads_ast = ast.Tuple([expr])
            case ast.Tuple():
                num_threads_ast = expr
            case ast.List():
                num_threads_ast = ast.Tuple(expr.elts)
            case _:  # dynamic check
                tvar = omp_name(ctx)
                num_threads_ast = ast.IfExp(
                    test=ast.Call(
                        ast.Name(id="isinstance"),
                        [ast.NamedExpr(ast.Name(tvar, ast.Store()), expr), ast.Name(id="tuple")],
                    ),
                    body=ast.Name(tvar),
                    orelse=ast.Tuple([ast.Name(tvar)]),
                )

    if ctr.proc_bind:
        proc_bind_ast = ctr.proc_bind.span.to_ast(ast.Constant(ord(ctr.proc_bind.ntype.string.lower()[0])))

    if ctr.copyin:
        check_scopes(ctx, *ctr.copyin, allow_threadprivate=True)
        for dataclause in ctr.copyin:
            dataclause.span.to_ast(proc_bind_ast)
            for var in dataclause.targets:
                if var.string not in ctx.module_storage.threadprivate:
                    msg = f"'{var}' must be 'threadprivate' for 'copyin'"
                    raise syntax_error_ctx(msg, var.span, ctx)

                s = ctx.symtable.get(var.string, True, True)
                if not s or not s.threadprivate:
                    msg = f"'{var}' is not 'threadprivate' in the current scope"
                    raise syntax_error_ctx(msg, var.span, ctx)
                copyin_ast.elts.append(var.span.to_ast(ast.Constant(var.string)))

    parallel_run: ast.Expr = ctr.span.to_ast(
        ast.Expr(
            ast.Call(
                runtime_ast("parallel"),
                [ast.Name(f_name), if_ast, num_threads_ast, proc_bind_ast, copyin_ast],
            ),
        ),
    )

    return fix_body_locations([f_ast, parallel_run])
