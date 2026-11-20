"""A device that overrides ONE transformation and delegates everything else to `omp`."""

import ast

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_device, omp_construct
from omp4py.core.parser import Barrier, Construct
from omp4py.core.preprocessor.transformers.context import Context


@add_device("loud")
def loud(ctr: Construct, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    if isinstance(ctr, Barrier):                         # our own transformation for `barrier`
        say = ast.parse("print('--- barrier reached ---')").body[0]
        ctr.span.to_ast(say)
        return [ast.fix_missing_locations(say), *omp_construct(ctr, block, ctx)]   # ...then the standard one
    return omp_construct(ctr, block, ctx)                # base case: delegate to `omp`
