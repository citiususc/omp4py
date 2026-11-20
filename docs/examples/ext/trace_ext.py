"""A custom subparser (`trace`) with its own construct and transformation."""

import ast
from dataclasses import dataclass

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_parser, omp_construct, syntax_error_ctx
from omp4py.core.parser import Construct, Directive, Name, Span
from omp4py.core.preprocessor.transformers.context import Context


@dataclass
class Trace(Construct):
    """`trace("label")`: print a message before and after a block."""

    label: str = ""


@add_parser("trace")
def parse_trace(code: str, span: Span, filename: str) -> Directive:
    label = code.strip()
    if not label:
        raise SyntaxError("trace() needs a label")
    return Directive(span, code, [Trace(span, Name(span, "trace"), label=label)])


@omp_construct.register
def _(ctr: Trace, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    if block is None:
        raise syntax_error_ctx("trace must be used in a 'with' block", ctr.span, ctx)

    def say(text: str) -> ast.stmt:
        node = ast.Expr(ast.Call(ast.Name("print"), [ast.Constant(text)], []))
        return ast.fix_missing_locations(ctr.span.to_ast(node))

    return [say(f"[trace] enter {ctr.label}"), block, say(f"[trace] exit {ctr.label}")]
