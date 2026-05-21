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

from omp4py.core.parser.tree import Directive, Span

__all__ = ["extract_directive", "parse_directive", "syntax_error"]


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

    msg = "Complex directives is not supported yet"
    raise NotImplementedError(msg)


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
    """
    msg = "New parser is not implemented yet"
    raise NotImplementedError(msg)
