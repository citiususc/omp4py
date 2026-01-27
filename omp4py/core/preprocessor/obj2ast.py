import ast
import inspect
import tokenize
import typing
from io import StringIO
from collections.abc import Callable, Iterator
from typing import Any

from omp4py.core.parser import Span, syntax_error

__all__ = ["from_object"]


def get_indentation(lines: list[str]) -> int:
    try:
        it: Iterator[str] = iter(lines)
        token: tokenize.TokenInfo
        for token in tokenize.tokenize(lambda: next(it).encode()):
            if token.type == tokenize.INDENT:
                return token.end[1] - token.start[1]
    except StopIteration:
        pass
    return 0


def from_object(obj: Callable[..., Any] | type) -> tuple[str, str, ast.Module]:
    lines: list[str]
    start: int
    lines, start = inspect.findsource(obj)
    full_source: str = "".join(lines)
    filename: str = inspect.getfile(obj)
    indentation: int = get_indentation(lines) if lines[start][0].isspace() else 0
    indent_level: int = 0 if indentation == 0 else (len(lines[start]) - len(lines[start].lstrip())) // indentation

    source: StringIO = StringIO(newline="")
    source.write("\n" * (start - indent_level))
    source.writelines([" " * (indentation * i) + "if True:\n" for i in range(indent_level)])
    source.writelines(inspect.getblock(lines[start:]))
    source.seek(0)

    module: ast.Module = ast.parse(source.read(), filename)
    for _ in range(indent_level):  # remove the fake ifs
        child: ast.If = typing.cast("ast.If", module.body[0])
        module.body = child.body

    return filename, full_source, module
