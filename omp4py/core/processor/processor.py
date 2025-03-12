import ast
from typing import Protocol, Callable

from omp4py.core.directive import OmpClause, OmpArgs
from omp4py.core.processor.nodes import NodeContext

__all__ = ['OmpProcessor', 'OMP_PROCESSOR', 'omp_processor']


class OmpProcessor(Protocol):

    def __call__(self, body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) \
            -> list[ast.stmt]:
        pass


OMP_PROCESSOR: dict[str, OmpProcessor] = {}


def omp_processor(name) -> Callable[[OmpProcessor], OmpProcessor]:
    def w(f: OmpProcessor) -> OmpProcessor:
        OMP_PROCESSOR[name] = f
        return f

    return w
