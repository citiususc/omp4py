from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omp4py.core.parser.tree import Span

def syntax_error(message: str, span: Span, source: str, filename: str) -> SyntaxError:
    text: str = source.split("\n")[span.lineno - 1]
    if span.end_lineno < 0 or span.end_offset < 0:
        return SyntaxError(message, (filename, span.lineno, span.offset + 1, text))
    return SyntaxError(
        message,
        (filename, span.lineno, span.offset + 1, text, span.end_lineno, span.end_offset + 1),
    )
