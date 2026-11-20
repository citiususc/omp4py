"""A subparser (`timed`) and a device (`profile`) working together."""

import ast
from dataclasses import dataclass

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_device, add_parser, omp_construct, syntax_error_ctx
from omp4py.core.parser import Construct, Directive, Name, Parallel, Span
from omp4py.core.preprocessor.transformers.context import Context
from omp4py.core.preprocessor.transformers.symtable import omp_name


@dataclass
class Timed(Construct):
    """`timed("label")`: measure the wall time of a block."""

    label: str = ""


def timing_wrapper(label: str, body: list[ast.stmt], ctx: Context, span: Span) -> list[ast.stmt]:
    var = omp_name(ctx, "t0")
    code = f"import time as _t\n{var} = _t.perf_counter()\n"
    tail = f"print({label!r}, 'took', round(_t.perf_counter() - {var}, 4), 's')\n"
    nodes = [*ast.parse(code).body, *body, *ast.parse(tail).body]
    for node in nodes:
        ast.fix_missing_locations(span.to_ast(node))
    return nodes


@add_parser("timed")
def parse_timed(code: str, span: Span, filename: str) -> Directive:
    return Directive(span, code, [Timed(span, Name(span, "timed"), label=code.strip())])


@add_device("profile")
def profile(ctr: Construct, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    match ctr:
        case Timed():
            if block is None:
                raise syntax_error_ctx("timed must be used in a 'with' block", ctr.span, ctx)
            return timing_wrapper(ctr.label, [block], ctx, ctr.span)
        case Parallel():
            # Reuse the default transformation and add timing around the parallel call.
            return timing_wrapper("parallel region", omp_construct(ctr, block, ctx), ctx, ctr.span)
        case _:
            return omp_construct(ctr, block, ctx)
