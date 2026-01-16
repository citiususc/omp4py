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


class NamespaceVisitor(ast.NodeVisitor):
    namespace: int
    line: int
    name: str
    result: tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, int] | None

    def __init__(self, name: str, line: int) -> None:
        self.namespace = 0
        self.name = name
        self.line: int = line
        self.result = None

    def search(self, module: ast.Module) -> tuple[ast.stmt, int]:
        self.generic_visit(module)
        if self.result is None:
            msg: str = "source code not found"
            raise ValueError(msg)
        return self.result

    def visit_namespace(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.namespace += 1
        lineno: int = node.lineno - 1
        if len(node.decorator_list) > 0:
            lineno = node.decorator_list[0].lineno - 1
        if lineno == self.line and node.name == self.name:
            self.result = (node, self.namespace)

    visit_FunctionDef = visit_namespace  # noqa: N815
    visit_ClassDef = visit_namespace  # noqa: N815
    visit_AsyncFunctionDef = visit_namespace  # noqa: N815


def check_object(obj: Callable[..., Any] | type):
    if not (inspect.isclass(obj) or inspect.iscoroutinefunction(obj) or inspect.isfunction(obj)):
        msg: str = "Invalid object type: only classes, functions, or async functions are allowed"
        raise ValueError(msg)
    qname: str = getattr(obj, "__qualname__", "")

    if "." in qname:
        msg: str = "Decorator must be applied to an outer function or class"
        filename: str = inspect.getfile(obj)
        lines: list[str]
        start: int
        lines, start = inspect.findsource(obj)
        offset: int = len(lines[start]) - len(lines[start].lstrip())
        full_source: str = "".join(lines)
        raise syntax_error(msg, Span(start + 1, offset), full_source, filename)


def from_object(obj: Callable[..., Any] | type) -> tuple[str, str, ast.Module, int]:
    check_object(obj)
    lines: list[str]
    start: int
    lines, start = inspect.findsource(obj)
    full_source: str = "".join(lines)
    filename: str = inspect.getfile(obj)
    module: ast.Module = ast.parse(full_source, filename)
    name: str = getattr(obj, "__name__", "")
    obj_smt: ast.stmt
    namespace: int
    obj_smt, namespace = NamespaceVisitor(name, start).search(module)
    module.body = [obj_smt]

    return filename, full_source, module, namespace


def from_object2(obj: Callable[..., Any] | type) -> tuple[str, str, ast.Module]:
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
