import ast

from omp4py.core.directive import names
from omp4py.core.processor.processor import omp_processor, OMP_PROCESSOR
from omp4py.core.directive import OmpClause, OmpArgs
from omp4py.core.processor.nodes import NodeContext

__all__ = []

def clause_filter(name: str, clauses: list[OmpClause]) -> list[OmpClause]:
    return list(filter(lambda c: c.directive == names.D_FOR, clauses))


@omp_processor(f'{names.D_PARALLEL} {names.D_FOR}')
def parallel(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    body = OMP_PROCESSOR[names.D_FOR](body, clause_filter(names.D_FOR, clauses), args, ctx)
    body = OMP_PROCESSOR[names.D_PARALLEL](body, clause_filter(names.D_PARALLEL, clauses), args, ctx)
    return body
