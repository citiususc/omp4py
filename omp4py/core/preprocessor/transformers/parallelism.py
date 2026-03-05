import ast

from omp4py.core.parser.tree import Parallel
from omp4py.core.preprocessor.transformers.scopes import create_scope, check_scopes
from omp4py.core.preprocessor.transformers.symtable import new_omp_name, new_omp_uname, runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx
from omp4py.core.preprocessor.transformers.utils import fix_body_locations

__all__ = []


@construct.register
def _(ctr: Parallel, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    f_name: str = new_omp_name(ctx, "parallel")
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
                tvar = new_omp_uname(ctx)
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
