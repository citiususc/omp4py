"""TODO: write docstring."""

import ast
import typing
import symtable
from collections.abc import Callable
from collections.abc import Iterator

from omp4py.core.parser import syntax_error
from omp4py.core.parser.tree import Construct, Parallel
from omp4py.core.preprocessor.transformers import parallelism
from omp4py.core.preprocessor.transformers.context import Context, Params

__all__ = ["OmpTransformer", "Params"]


available: dict[type[Construct], Callable[[Construct, list[ast.stmt], Context], list[ast.stmt]]] = {
    Parallel: parallelism.parallel,
}


def not_implemented(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    msg: str = f"{ctr.name} not implemented"
    raise syntax_error(msg, ctr.span, ctx.full_source, ctx.filename)


def transform(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    return available.get(type(ctr), not_implemented)(ctr, body, ctx)


def traverse_table(table: symtable.SymbolTable) -> Iterator[symtable.SymbolTable]:
    if table.get_type() in ("module", "class", "function"):
        yield table
    for child in table.get_children():
        yield from traverse_table(child)


def next_table(it: Iterator[symtable.SymbolTable], name: str | None = None) -> symtable.SymbolTable:
    table: symtable.SymbolTable = next(it)
    if name is not None and table.get_name() != name:
        msg: str = (
            "Symbol table mismatch: the symbol table does not match the current execution namespace "
            f"'{name} != {table.get_name()}'"
        )
        raise ValueError(msg)
    return table


def take_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ctx: Context) -> ast.expr | None:
    i: int
    child: ast.expr
    name: str
    for i, child in enumerate(node.decorator_list):
        if isinstance(child, ast.Name):
            name = child.id
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            name = child.func.id
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            name = child.func.attr
        elif isinstance(child, ast.Attribute):
            name = child.attr
        else:
            continue
        if name == ctx.params.alias:
            if ctx.decorator is not None:
                pass  # TODO: raise multiple omp decorator error or module with multiple omp
            ctx.decorator = node.decorator_list.pop(i)
            break


class OmpTransformer(ast.NodeTransformer):
    ctx: Context
    tables: Iterator[symtable.SymbolTable]

    def __init__(self, full_source: str, filename: str, module: ast.Module, namespace: int, params: Params):
        self.ctx = Context(full_source, filename, module, namespace, params)
        self.tables = traverse_table(symtable.symtable(full_source, filename, "exec"))

    def transform(self) -> ast.Module:
        if len(self.ctx.node_stack) > 0:
            self.ctx.node_stack.pop()
            self.visit(self.ctx.module)
        return self.ctx.module

    def visit(self, node: ast.AST) -> ast.AST:
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.ctx.symtable_stack.append(next_table(self.tables, "top"))
        for _ in range(self.ctx.namespace):
            next_table(self.tables)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        take_decorator(node, self.ctx)
        self.ctx.symtable_stack.append(next_table(self.tables, node.name))
        self.generic_visit(node)
        self.ctx.symtable_stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        take_decorator(node, self.ctx)
        self.ctx.symtable_stack.append(next_table(self.tables, node.name))
        self.generic_visit(node)
        self.ctx.symtable_stack.pop()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        take_decorator(node, self.ctx)
        self.ctx.symtable_stack.append(next_table(self.tables, node.name))
        self.generic_visit(node)
        self.ctx.symtable_stack.pop()
        return node
