"""Directive parsing for the `omp4py` preprocessor.

This module provides the low-level parsing helpers used during the
OpenMP-like directive preprocessing stage.

The parser is responsible for:

1. Extracting raw directive source code from Python AST nodes.
2. Preserving accurate source locations for tokens and constructs.
3. Building structured directive representations from textual directives.
4. Generating contextual syntax errors with precise source mapping.

The parsing process operates on directive strings extracted from calls
such as ``omp("parallel for")`` and converts them into structured
directive trees used later during AST transformation.

The module also contains helper utilities for creating syntax errors
that include filename, line, and column information compatible with
Python's native error reporting system.
"""
from __future__ import annotations

import ast
import typing
from pathlib import Path

from . import openmp_parser as omp
from . import string_parser as pre

from .source_view import SourceView
from .transformer import AstTransformer
from .tree import Directive, Span

__all__ = ["extract_directive", "parse_directive", "syntax_error"]

@pre.v_args(inline=True)
class PreTransformer(pre.Transformer):
    def scape_seq(self, token: pre.Token) -> str:
        return " "*len(str(token))

    def string_token(self, token: pre.Token) -> str:
        return str(token)

    @pre.v_args(inline=False)
    def start(self, children: list) -> tuple[str, int, int]:
        # children is either [OPEN_DELIM, ...content..., CLOSE_DELIM]
        # or [PREFIX, OPEN_DELIM, ...content..., CLOSE_DELIM]
        has_prefix = (
            len(children) > 2 and
            isinstance(children[0], pre.Token) and
            children[0].type == "STRING_PREFIX"
        )

        prefix_len     = len(children[0]) if has_prefix else 0
        delim_len      = len(children[1 if has_prefix else 0])
        last_delim_len = len(children[-1])

        content_offset = prefix_len + delim_len
        content = "".join(str(c) for c in (children[2:-1] if has_prefix else children[1:-1]))

        return content, content_offset, last_delim_len


preprocesor   = pre.Lark_StandAlone(transformer=PreTransformer())
openmp_parser = omp.Lark_StandAlone()
begin_offset = 0
end_offset = 0


def syntax_error(message: str, span: Span, source: str, filename: str) -> SyntaxError:
    """Create a syntax error associated with a source code span.

    This helper constructs a ``SyntaxError`` instance using positional
    information stored in a ``Span`` object. The resulting exception
    includes filename, line number, column offsets, and the original
    source line, enabling accurate and user-friendly error reporting.

    If end position information is available, it is also attached to the
    exception to support precise source highlighting.

    Args:
        message (str): Error message.
        span (Span): Source span associated with the error.
        source (str): Full source code being processed.
        filename (str): Source filename.

    Returns:
        SyntaxError: Constructed syntax error with contextual information.
    """
    text: str = source.split("\n")[span.lineno - 1]
    if span.end_lineno < 0 or span.end_offset < 0:
        return SyntaxError(message, (filename, span.lineno, span.offset + 1, text))
    return SyntaxError(
        message,
        (filename, span.lineno, span.offset + 1, text, span.end_lineno, span.end_offset + 1),
    )


def extract_directive(node: ast.Constant, full_source: str, filename: str) -> str:
    """Extract the raw directive source from an AST constant node.

    This function retrieves the textual representation of a directive
    stored inside a string constant, preserving the original source
    positions possible.

    Simple string literals are returned directly from their evaluated
    value. More complex string forms, such as multiline literals or
    escaped strings, require recovering the original source segment in
    order to preserve accurate token positioning during parsing.

    Args:
        node (ast.Constant): Constant node containing the directive string.
        full_source (str): Full source code being processed.
        filename (str): Source filename.

    Returns:
        str: Extracted directive source string.

    Raises:
        SyntaxError: If the original source segment cannot be recovered.
    """
    node_value = str(node.value)
    raw_source: str | None = ast.get_source_segment(full_source, node)
    if raw_source is None:
        msg = "source directive not found"
        raise syntax_error(msg, Span.from_ast(node), full_source, filename)

    if len(raw_source) - 2 == len(node_value):
        return node_value

    global begin_offset, end_offset
    contents, begin_offset, end_offset = preprocesor.parse(raw_source)
    return contents


# Required for the tests, to avoid duplicating the error handling
def _parse(code: str, source_view: SourceView) -> Directive:
    transformer = AstTransformer(source_view)
    try:
        parse_tree = openmp_parser.parse(code)
        return transformer.transform(parse_tree)
    except omp.UnexpectedToken as e:
        raise source_view.error(e) from None
    except omp.VisitError as e:
        if isinstance(e.orig_exc, SyntaxError):
            raise e.orig_exc from None
        raise


def parse_directive(code: str, span: Span, filename: str) -> Directive:
    """Parse a directive string into a structured directive tree.

    This function performs the parsing stage for OpenMP-like directives,
    converting the textual directive representation into a structured
    syntax tree composed of constructs, clauses, and modifiers.

    The resulting ``Directive`` object is later consumed by the
    transformation pipeline to apply AST-level code generation and
    rewriting.

    Parsing preserves source location information to enable accurate
    diagnostics and contextual error reporting during later stages.

    Args:
        code (str): Raw directive source code.
        span (Span): Source span associated with the directive.
        filename (str): Source filename.

    Returns:
        Directive: Parsed directive representation.

    Raises:
        SyntaxError if the directive is incorrect.
    """
    span.offset     += begin_offset
    span.end_offset -= end_offset
    source_view = SourceView.from_file(span, filename, code)
    return _parse(code, source_view)



