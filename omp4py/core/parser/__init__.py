from omp4py.core.parser.tree import Directive, Span


__all__ = ["parse", "syntax_error"]


# TODO: take this to create parser interface
def parse(directive: str, lineno: int, col: int) -> Directive:
    msg = "New parser is not implemented yet"
    raise NotImplementedError(msg)


def syntax_error(message: str, span: Span, source: str, filename: str) -> SyntaxError:
    text: str = source.split("\n")[span.lineno - 1]
    if span.end_lineno < 0 or span.end_offset < 0:
        return SyntaxError(message, (filename, span.lineno, span.offset + 1, text))
    return SyntaxError(
        message,
        (filename, span.lineno, span.offset + 1, text, span.end_lineno, span.end_offset + 1),
    )
