import ast

from omp4py.core.directive import names, OmpItem, tokenizer
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpClause, OmpArgs
from omp4py.core.processor.nodes import NodeContext, check_body, check_nobody
from omp4py.core.processor import common

__all__ = []

@omp_processor(names.D_CRITICAL)
def critical(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)
    new_body: list[ast.stmt] = list()

    lock: ast.Call = ctx.new_call(f'{ctx.r}.mutex_lock')
    unlock: ast.Call = ctx.new_call(f'{ctx.r}.mutex_unlock')

    new_body.append(ctx.copy_pos(ast.Expr(lock)))
    new_body.extend(body)

    return [ctx.new_try(new_body, [ctx.copy_pos(ast.Expr(unlock))])]


@omp_processor(names.D_BARRIER)
def barrier(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_nobody(ctx, body)
    return [ctx.copy_pos(common.barrier(ctx))]
