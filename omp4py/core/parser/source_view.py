"""Source access and diagnostics for OpenMP directive parsing.

This module provides utilities to map parser and AST diagnostics back to the
original Python source code containing an ``omp()`` directive.

It defines ``SourceView``, which swns the source text and provides helpers to:

* Extract source fragments
* Build annotated ``SyntaxError`` instances

Line numbers are 1-based and column offsets are 0-based, matching Python's
``SyntaxError`` conventions.

This module does not perform parsing; it only formats and contextualizes
errors produced by the parser.
"""
from __future__ import annotations
from cmath import exp

import typing
from dataclasses import dataclass
from pathlib import Path

from .tree import Span

if typing.TYPE_CHECKING:
    from .openmp_parser import Token, Meta, UnexpectedToken

__all__ = ["SourceView"]

# Look up table to translate token names to a more meaningfull name.
# Defaults to lowercase the name of the token with double quotes.
_TOKEN_DISPLAY: dict[str, str] = {
    # Reduction operators
    "PLUS"        : '"+"',
    "MINUS"       : '"-"',
    "MULT"        : '"*"',
    "BITWISE_AND" : '"&"',
    "BITWISE_OR"  : '"|"',
    "BITWISE_XOR" : '"^"',
    "LOGIC_AND"   : '"and"',
    "LOGIC_OR"    : '"or"',
    "MAX"         : '"max"',
    "MIN"         : '"min"',

    # Other
    "INTEGER"    : "integer",
    "IDENTIFIER" : "variable name",

    # Defined by Lark
    "LPAR"  : '"("',
    "RPAR"  : '")"',
    "LBRACE": '"{"',
    "RBRACE": '"}"',
    "LSQB"  : '"["',
    "RSQB"  : '"]"',
    "COMMA" : '","',
    "COLON" : '":"',
    "$END"  : "end of input",
}


class SourceView:
    """View and query helper for directive source code.

    This class encapsulates access to the original source code associated with
    an ``omp()`` directive invocation. It stores the full file contents and
    provides utilities to:

    - Extract text fragments using line/column spans
    - Compute absolute positions for diagnostics
    - Construct rich ``SyntaxError`` objects with annotations and notes

    All line numbers are 1-based, and column offsets are 0-based, matching
    Python's ``SyntaxError`` conventions.
    """
    def __init__(
        self,
        span: Span,
        filename: str,
        lines: list[str],
        directive: str
    ) -> None:
        self.span = span
        self.filename = filename
        self.lines = lines
        self.directive = directive

    @staticmethod
    def from_file(span: Span, filename: str, directive: str) -> SourceView:
        """Construct a SourceView by loading source lines from a file.

        Args:
            span (tree.Span): Span locating the directive in the file.
            filename (str): Path to the source file.
            directive (str): Raw text of the directive.
        Returns:
            SourceView: A new instance populated with the file contents.
        """
        return SourceView(
            span,
            filename,
            Path(filename).read_text().splitlines(),
            directive,
        )

    #### TEXT RETRIEVAL ########################################################

    def source_line(self, lineno: int) -> str:
        """Return a single line of source code.

        Args:
            lineno (int): 1-based line number.
        Returns:
            str: The corresponding line of source code.
        """
        if not self.lines:
            return ''
        return self.lines[lineno - 1]

    def source_text(self, span: Span) -> str:
        """Extract source text referenced by a span.

        Args:
            span (tree.Span): Span identifying a region of the source file.
        Returns:
            str: The concatenated source text covered by the span.
        """
        if span.lineno == span.end_lineno:
            return self.lines[span.lineno - 1][span.offset:span.end_offset]

        parts = [self.lines[span.lineno - 1][span.offset:]]
        parts.extend(self.lines[span.lineno:span.end_lineno - 1])
        parts.append(self.lines[span.end_lineno - 1][:span.end_offset])

        return "\n".join(parts)

    #### POSITION TRANSFORMATION ###############################################

    def token2span(self, token: Token) -> Span:
        if token.line is None or token.column is None:
            msg = "Missing position information"
            raise ValueError(msg)

        # Lark starts at line 1 column 1, but the Span expects an offset.
        line_offset      = max(self.span.lineno, 1)
        start_col_offset = self.span.offset if token.line == 1 else 0
        end_col_offset   = self.span.offset if token.end_line == 1 else 0

        return Span(
            line_offset      + token.line - 1,
            start_col_offset + token.column - 1,
            line_offset      + token.end_line - 1   if token.end_line   is not None else -1,
            end_col_offset   + token.end_column - 1 if token.end_column is not None else -1,
        )

    def meta2span(self, meta: Meta) -> Span:
        if meta.empty:
            msg = "Meta object is empty"
            raise ValueError(msg)

        line_offset      = max(self.span.lineno, 1)
        start_col_offset = self.span.offset if meta.line == 1 else 0
        end_col_offset   = self.span.offset if meta.end_line == 1 else 0

        return Span(
            line_offset      + meta.line - 1,
            start_col_offset + meta.column - 1,
            line_offset      + meta.end_line - 1,
            end_col_offset   + meta.end_column - 1,
        )

    def absolute_position(
        self,
        anchor: Span,
        rel_line: int,
        rel_col: int,
        first_offset: int = 0,
    ) -> tuple[int, int]:
        """Convert a position relative to an anchor span into absolute coordinates.

        No validation is performed.

        Args:
            anchor (tree.Span): Anchor span.
            rel_line (int): Line number relative to the anchor (1-based).
            rel_col (int): Column number relative to the anchor (1-based).
            first_offset (int): Additional column offset applied only if ``rel_line == 1``.
        Returns:
            tuple[int, int]: Absolute (line, column) position in the source file.
        """
        abs_line = anchor.lineno + rel_line - 1
        abs_col = (
            anchor.offset + first_offset + rel_col - 1 # offset only applies on the first line
            if rel_line == 1
            else rel_col - 1
        )
        return abs_line, abs_col


    #### ERRORS ################################################################

    def syntax_error(
        self,
        message,
        span: Span,
        *,
        diagnostics: list[tuple[str, Span]|str] | None = None,
    ) -> SyntaxError:
        """Construct a ``SyntaxError`` with optional diagnostic notes.

        Args:
            message (str): Error message.
            span (tree.Span): Span identifying the error location.
            diagnostics: Optional list of (message, span) pairs used to generate additional annotated notes.
        Returns:
            SyntaxError: A fully populated ``SyntaxError`` instance.
        """
        text = self.source_line(span.lineno)

        error: SyntaxError
        if span.end_lineno < 0 or span.end_offset < 0:
            error = SyntaxError(
                message,
                (self.filename, span.lineno, span.offset + 1, text),
            )
        else:
            error = SyntaxError(
                message,
                (self.filename, span.lineno, span.offset + 1, text, span.end_lineno, span.end_offset + 1),
            )

        if diagnostics is not None:
            for diag in diagnostics:
                if isinstance(diag, str):
                    error.add_note(f"  note: {diag}")
                else:
                    msg, span = diag
                    line, cursor = self.annotate(span, indent=2, show_lineno=True)
                    error.add_note(f"  note: {msg}\n{line}{cursor[:-1]}")

        return error


    def error(
        self,
        error: UnexpectedToken,
        diagnostics: list[tuple[str, Span]] | None = None,
    ) -> SyntaxError:
        """Convert a parser ``UnexpectedToken`` into a ``SyntaxError``.

        Args:
            error (lark.exceptions.UnexpectedToken): Parser error raised by Lark.
            diagnostics: Optional diagnostic notes forwarded to :meth:`syntax_error`.
        Returns:
            SyntaxError: A formatted syntax error suitable for user display.
        """
        token = typing.cast("Token", error.token)
        msg, span = self._msg_from_error(error, self.token2span(token))
        return self.syntax_error(msg, span, diagnostics=diagnostics)


    def _msg_from_error(self, error: UnexpectedToken, span: Span) -> tuple[str, Span]:
        token = typing.cast("Token", error.token)

        print("DEBUG:", error.expected)
        print("DEBUG:", token.type)

        #### Expected tokens ####

        expected_token_names: list[str] = []
        expected_directive = False
        expected_clause    = False
        expected_code      = False
        expected_end       = False

        for expected_token in error.expected:
            if expected_token.endswith("_DIRECTIVE"):
                expected_directive = True
                continue

            if expected_token.endswith("_CLAUSE"):
                expected_clause = True
                continue

            if expected_token == "PY_CODE_IN" or expected_token == "PY_CODE_OUT":
                expected_code = True
                continue

            if expected_token == "<END-OF-FILE>":
                expected_end = True
                continue

            expected_token_names.append(_TOKEN_DISPLAY.get(expected_token, f'"{expected_token.lower()}"'))

        # Make sure these are last
        if expected_clause:
            expected_token_names.append("OpenMP clause")
        if expected_directive:
            expected_token_names.append("OpenMP directive")

        # Convert to expected_str
        expected_token_names = sorted(expected_token_names)
        if len(expected_token_names) == 0:
            expected_str = "OpenMP directive"
        elif len(expected_token_names) == 1:
            expected_str = expected_token_names[0]
        else:
            expected_str = ", ".join(expected_token_names[:-1]) + f" or {expected_token_names[-1]}"


        #### Actual token received ####
        found_token = _TOKEN_DISPLAY.get(token.type, f'"{token}"')

        # If the current token is an identifier and the last token was an integer,
        # it is probably because the integer was invalid and the lexer broke it into parts:
        #     "0o9"  ==> integer 0 + "o9" identifier
        #     "0o19" ==> integer 1 + "9" integer
        # This is only applies if both tokens are next to each other with no whitespace in between,
        # because "0 o9" should get a different error.
        if (token.type == "IDENTIFIER" or token.type == "INTEGER") and error.token_history:
            last_token = typing.cast("Token", error.token_history[-1])

            if (
                last_token and
                last_token.type == "INTEGER" and
                last_token.column is not None and
                last_token.end_column is not None and
                token.column is not None and
                last_token.line == token.line and
                last_token.end_column == token.column
            ):
                start_col_offset = self.span.offset if last_token.line == 1 else 0
                span.offset = start_col_offset + last_token.column - 1
                return f'invalid integer literal "{last_token}{token}".', span

        # If the token is PY_CODE, it means that we got unexpected characters.
        # The problem here is that PY_CODE will consume everything until a parentheses,
        # therefore the error location will be wrong.
        # Only point to the first incorrect caracter.
        if token.type == "PY_CODE_IN" or token.type == "PY_CODE_OUT":
            first_char = token[0]
            display = "integer" if first_char.isdigit() else f"'{first_char}'"
            span.end_offset = span.offset
            span.end_lineno = span.lineno
            return f'expected {expected_str} instead of {display}.', span

        if expected_end and token.type.endswith("_CLAUSE"):
            return f'this directive does not accept any clauses.', span

        if expected_clause and (token.type.endswith(("_DIRECTIVE", "_CLAUSE")) or token.type == "IDENTIFIER"):
            return f'{token} clause is invalid for this directive.', span

        # If we expected a directive and the token was a directive,
        # it means that this directive was incorrect
        if expected_directive and token.type.endswith("_DIRECTIVE"):
            return f'{token} directive is invalid here.', span

        if expected_directive and expected_clause:
            return f"expected OpenMP clause or directive before {found_token}.", span

        if expected_code:
            return f"expected Python expression before {found_token}.", span

        return f"expected {expected_str} before {found_token}.", span


    def annotate(self, span: Span, indent: int=0, show_lineno: bool=False) -> tuple[str, str]:
        """Generate an annotated source line and cursor indicator.

        Args:
            span (tree.Span): Span to highlight.
            indent (int): Number of spaces to indent both line and cursor.
            show_lineno (bool): Whether to prefix the line with its line number.
        Returns:
            tuple[str, str]: (source line, cursor line)
        """

        line = self.lines[span.lineno - 1] + "\n"
        width = (
            span.end_offset - span.offset
            if span.lineno == span.end_lineno
            else len(line) - span.offset
        )
        cursor = " " * span.offset + "^" * width + "\n"

        if show_lineno:
            line = f"{span.lineno:>5} | {line}"
            cursor = " "*5 + " | " + cursor

        line   = " " * indent + line
        cursor = " " * indent + cursor

        return line, cursor


