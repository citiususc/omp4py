import ast

from omp4py.core.parser.tree import Barrier, Critical, Master, Ordered, Span
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx

__all__ = []


@construct.register
def _(ctr: Barrier, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    barrier: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("barrier"))))
    ast.fix_missing_locations(barrier)

    return [barrier, *body]


@construct.register
def _(ctr: Critical, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    lock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_lock"))))
    unlock: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("critical_unlock"))))

    ast.fix_missing_locations(lock)
    ast.fix_missing_locations(unlock)

    return [lock, *body, unlock]


@construct.register
def _(ctr: Master, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    master: ast.If = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("master"))))

    ast.fix_missing_locations(master)
    master.body = body

    return [master]


@construct.register
def _(ctr: Ordered, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    init: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_init"))))
    end: ast.stmt = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("ordered_end"))))

    ast.fix_missing_locations(init)
    ast.fix_missing_locations(end)

    return [init, *body, end]
