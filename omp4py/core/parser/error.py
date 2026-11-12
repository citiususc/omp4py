"""Error handling utilities for the `omp4py` directive parser.

This module converts parser errors into Python `SyntaxError` exceptions
with source locations mapped back to the original Python file.

It also provides helpers for identifying the source span associated with
a parsing failure and for producing more specific diagnostic messages.
"""

from __future__ import annotations

import difflib

import omp4py.core.parser.openmp_lark as pl
from omp4py.core.parser.tree import Span

__all__ = ["parser_error", "syntax_error"]


def syntax_error(message: str, span: Span, source: str, filename: str) -> SyntaxError:
    """Create a syntax error associated with a source code span.

    Constructs a `SyntaxError` using the location information stored in
    `span`. The exception includes the filename, line number, column
    offset, and corresponding source line. When end-position information
    is available, it is included to support precise error highlighting.

    Args:
        message (str): Error message to display.
        span (Span): Source span associated with the error.
        source (str): Complete source code being processed.
        filename (str): Source filename.

    Returns:
        SyntaxError: Exception populated with source location information.
    """
    text: str = source.split("\n")[span.lineno - 1]
    if span.end_lineno < 0 or span.end_offset < 0:
        return SyntaxError(message, (filename, span.lineno, span.offset + 1, text))
    return SyntaxError(
        message,
        (filename, span.lineno, span.offset + 1, text, span.end_lineno, span.end_offset + 1),
    )


def span_error(ex: pl.UnexpectedInput, span: Span, token: pl.Token | None, last_token: pl.Token | None) -> Span:
    """Determine the source span associated with a parser error.

    Maps the parser's error location or offending token to the original
    Python source. If the parser reaches the end of input, uses the last
    available token to construct a span immediately after it.

    Args:
        ex (pl.UnexpectedInput): Parsing exception containing location data.
        span (Span): Base source span of the directive being parsed.
        token (pl.Token | None): Offending token, if available.
        last_token (pl.Token | None): Last token consumed by the lexer,
            if available.

    Returns:
        Span: Error span mapped to the original Python source.
    """
    if token is not None and token.type == "$END":
        if last_token:
            new_span = span_error(ex, span, last_token, None)
            new_span.lineno = new_span.end_lineno
            new_span.offset = new_span.end_offset
            new_span.end_offset = new_span.end_offset + 1
            return new_span
        return span

    if (
        token
        and token.line is not None
        and token.end_line is not None
        and token.column is not None
        and token.end_column is not None
    ):
        line, end_line, col, end_col = token.line, token.end_line, token.column, token.end_column
    else:
        line, end_line, col, end_col = ex.line, ex.line, ex.column, ex.column + 1


    return Span(
        span.lineno + line - 1,
        (span.offset if line == 1 else 0) + col - 1,
        span.lineno + end_line - 1,
        (span.offset if line == 1 else 0) + end_col - 1,
    )


def parser_error(parser: pl.Lark, ex: pl.UnexpectedInput, base_span: Span, source: str, filename: str) -> SyntaxError:  # ruff: ignore[complex-structure, too-many-branches]
    """Convert a Lark parsing exception into a contextual syntax error.

    Uses parser state and token information to determine the error span.
    It may provide a more specific diagnostic by suggesting a similar
    accepted token, recognizing certain expected directive or clause
    contexts.

    If no specialized diagnostic applies, reports the expected terminals
    when the set is small, or falls back to a generic parsing error.

    Args:
        parser (pl.Lark): Parser that encountered the error.
        ex (pl.UnexpectedInput): Lark exception describing the parse failure.
        base_span (Span): Source span of the directive in the original code.
        source (str): Complete source code being processed.
        filename (str): Source filename.

    Returns:
        SyntaxError: Contextual syntax error with an appropriate message
            and source location.

    Raises:
        SyntaxError: If no specialized diagnostic can be returned, raises
            a generic parsing error.
    """
    token: pl.Token | None = None
    last_token: pl.Token | None = None
    parser_state: pl.ParserState | None = None
    accepts: set[str] | None = ex.accepts if isinstance(ex, pl.UnexpectedToken) else None
    terminals: dict[str, str] = {
        key: f"'{value.pattern.value}'"
        for key, value in parser.lexer_conf.terminals_by_name.items()
        if value.pattern.type == "str"
    }
    if accepts and ("PYTHON_BODY" in accepts or "PYTHON_LITE" in accepts):
        if "PYTHON_LITE" in accepts:
            accepts -= { "LBRACE", "LSQB"}
        accepts -= {"PYTHON_BODY","PYTHON_STRING","PYTHON_LONG_STRING","PYTHON_LITE", "LPAR"}
        accepts.add("PYTHON")

    terminals.update({"NAME": "identifier", "PYTHON": "python code"})
    str_accepts : set[str] | None = {terminals[x] for x in accepts if x in terminals} if accepts else None

    if (t := getattr(ex, "token", None)) is not None and isinstance(t, pl.Token):
        token = t

    if isinstance(ex.state, pl.ParserState):
        parser_state = ex.state

    if parser_state and hasattr(parser_state.lexer, "state") and isinstance(parser_state.lexer.state, pl.LexerState):
        last_token = parser_state.lexer.state.last_token

    span = span_error(ex, base_span, token, last_token)
    if accepts and len(accepts) == 1 and "_HM" in accepts:
        msg = "modifier required"
        if token and isinstance(ex, pl.UnexpectedToken):
            msg += f", found '{token.strip()}'"
        else:
            msg += " before end of line"
        return syntax_error(msg, span, source, filename)
    if accepts:
        accepts -= {"_HM"}

    if token and str_accepts:
        close = difflib.get_close_matches(token.lower(), str_accepts, 1)
        if close:
            msg = f"'{token}' is not defined. Did you mean: '{close[0]}'"
            return syntax_error(msg, span, source, filename)

    if accepts and ("DIST_SCHEDULE" in accepts or ("SEQ_CST" in accepts)):
        msg = "expected clause"
        if token and isinstance(ex, pl.UnexpectedToken):
            msg += f", found '{token.strip()}'"
        return syntax_error(msg, span, source, filename)

    if accepts and "TARGET_ENTER_DATA" in accepts:
        msg = "directive required"
        if token and isinstance(ex, pl.UnexpectedToken):
            msg += f", found '{token.strip()}'"
        else:
            msg += " before end of line"
        return syntax_error(msg, span, source, filename)

    if isinstance(ex, pl.UnexpectedCharacters) or (token is not None and token.type.startswith("PYTHON_")):
        span.end_lineno, span.end_offset = span.lineno, span.offset + 1

    if accepts and str_accepts and len(str_accepts) == len(accepts) and len(accepts) < 6 : # ruff: ignore[magic-value-comparison]
        msg = "expected " + " or ".join(sorted(str_accepts))
        if token and isinstance(ex, pl.UnexpectedToken):
            msg += f", found '{token.strip()}'"
        return syntax_error(msg, span, source, filename)

    msg = "parsing error"
    raise syntax_error(msg, span, source, filename)

