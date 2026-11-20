"""A custom device that ignores parallelism and runs everything sequentially."""

import ast

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_device, omp_construct, syntax_error_ctx
from omp4py.core.parser import Barrier, Construct, For, Parallel
from omp4py.core.preprocessor.transformers.context import Context


@add_device("serial")
def serial(ctr: Construct, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    match ctr:
        case Parallel() | For():
            # Keep the body, drop the parallel machinery.
            return [block] if block is not None else []
        case Barrier():
            return []
        case _:
            # Anything else: fall back to the default OpenMP transformation.
            return omp_construct(ctr, block, ctx)
