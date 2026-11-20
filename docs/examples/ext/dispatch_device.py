"""A device organised as a per-construct dispatch table."""

import ast
from functools import singledispatch

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_device, omp_construct
from omp4py.core.parser import Construct, For
from omp4py.core.preprocessor.transformers.context import Context


@singledispatch
def lower(ctr: Construct, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    """Default: use the standard OpenMP transformation."""
    return omp_construct(ctr, block, ctx)


@lower.register
def _(ctr: For, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    # Reproducible runs: ignore any schedule(...) the user asked for and use the default static one.
    ctr.schedule = None
    return omp_construct(ctr, block, ctx)


@add_device("deterministic")
def deterministic(ctr: Construct, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    return lower(ctr, block, ctx)
