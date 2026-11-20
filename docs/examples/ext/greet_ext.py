"""A minimal subparser: the `greet` directive."""

import ast
from dataclasses import dataclass

from omp4py import omp
from omp4py.core.ext import StructuredBlock, add_parser, omp_construct
from omp4py.core.parser import Construct, Directive, Name, Span
from omp4py.core.preprocessor.transformers.context import Context


@dataclass
class Greet(Construct):
    """New tree node: `greet("<word>")`."""

    word: str = ""


@add_parser("greet")                                    # 1. the subparser: text -> tree
def parse_greet(code: str, span: Span, filename: str) -> Directive:
    word = code.strip()                                 # the whole directive is one word
    return Directive(span, code, [Greet(span, Name(span, "greet"), word=word)])


@omp_construct.register                                 # 2. the transformer: tree -> Python AST
def _(ctr: Greet, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    node = ast.Expr(ast.Call(ast.Name("print"), [ast.Constant(f"hello, {ctr.word}")], []))
    return [ast.fix_missing_locations(ctr.span.to_ast(node))]
