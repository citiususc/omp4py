import ast

from omp4py.core.directive import names
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpClause, OmpArgs
from omp4py.core.processor.nodes import NodeContext, check_body, check_nobody
from omp4py.core.processor import common

__all__ = []

@omp_processor(names.D_CRITICAL)
def critical(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)
    return common.mutex(ctx, body)


@omp_processor(names.D_BARRIER)
def barrier(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_nobody(ctx, body)
    return [ctx.copy_pos(common.barrier(ctx))]
