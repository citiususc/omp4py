import ast
from typing import cast

from omp4py.core.parser.tree import Construct, Parallel
from omp4py.core.preprocessor.transformers.context import Context


def parallel(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    ctr: Parallel = cast("Parallel", ctr)
    a = ctr.private

    return []
