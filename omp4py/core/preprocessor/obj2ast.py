"""Source extraction utility module.

This module provides functionality to extract the original source code,
filename, and AST representation of a Python callable or class object.

It is primarily used by the preprocessing pipeline to reconstruct a
normalized AST module from runtime Python objects, preserving structural
information such as indentation levels and nested blocks.
"""

from __future__ import annotations

import ast
import inspect
import tokenize
import typing
from collections.abc import Callable, Iterator
from io import StringIO
from typing import Any

__all__ = ["from_object"]


def get_indentation(lines: list[str]) -> int:
    """Detect indentation size (in spaces) used in a block of source code.

    This function tokenizes the input source lines and searches for the
    first INDENT token to determine the base indentation width.

    Args:
        lines (list[str]): Source code lines.

    Returns:
        int: Number of spaces used for a single indentation level.
             Returns 0 if no indentation is detected.
    """
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
    """Extract filename, full source code, and AST module from a Python object.

    This function reconstructs a normalized AST representation of a function
    or class, even when the object is nested or indented inside another scope.

    It does this by locating the original source code, analyzing its indentation,
    and temporarily rebuilding a valid Python structure so it can be parsed
    correctly by the AST parser. After parsing, the temporary structure is
    removed to recover the original code layout.

    Args:
        obj (Callable[..., Any] | type): Python function or class object.

    Returns:
        tuple[str, str, ast.Module]:
            - filename: Path to the source file where the object is defined
            - full_source: Raw source code of the file
            - module: Parsed AST module representing the object definition

    Notes:
        This transformation is necessary because Python's AST parser
        cannot directly parse isolated nested blocks without a valid
        top-level structure. The synthetic `if True:` wrappers temporarily
        reconstruct valid syntax for parsing and are removed afterward.
    """
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
