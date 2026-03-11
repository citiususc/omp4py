from __future__ import annotations

import ast
import builtins
import re
from collections.abc import KeysView
from dataclasses import dataclass
from symtable import symtable as native_symtable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omp4py.core.preprocessor.transformers.context import Context

__all__ = ["SymbolEntry", "SymbolTable", "global_symtable", "omp_name", "runtime_ast"]

PREFIX: str = "_omp_"

var_check: re.Pattern[str] = re.compile(rf"^({PREFIX})?([0-9]+)?(.+)$")


def global_symtable(data: str, filename: str) -> SymbolTable:
    return SymbolTable(list(native_symtable(data, filename, "exec").get_identifiers()))


def is_variable(name: str) -> bool:
    return not name.startswith(PREFIX) or name[len(PREFIX) : len(PREFIX) + 1].isdigit()


@dataclass
class SymbolEntry:
    real_name: str
    scope_name: str
    old_name: str
    used: bool = False
    assigned: bool = False
    global_: bool = False
    threadprivate: bool = False
    annotation: ast.expr | None = None

    @property
    def renamed(self):
        return self.scope_name != self.old_name


class SymbolTableVisitor(ast.NodeVisitor):
    symbols: dict[str, SymbolEntry]
    check_namespace: bool
    to_rename: set[str]
    global_: bool

    def __init__(self, global_vars: list[str] | None):
        self.to_rename = set()
        self.check_namespace = False
        self.symbols = {}
        self.global_ = False

        if global_vars is not None:
            for name in dir(builtins):
                self.update_symbol(name).assigned = True
            self.global_ = True
            name: str
            for name in global_vars:
                self.update_symbol(name).assigned = True

    def update_symbol(self, name: str) -> SymbolEntry:
        match: re.Match[str] | None = var_check.match(name)
        if not match:
            return SymbolEntry("", "", "")
        omp: str | None = match.group(1)
        n: str | None = match.group(2)
        real_name: str = match.group(3)
        if omp is not None and n is None:
            return SymbolEntry("", "", "")  # internal _omp_ var

        symbol: SymbolEntry
        if real_name in self.symbols:
            symbol = self.symbols[real_name]
        else:
            if real_name in self.to_rename:
                old_name = name
            else:
                old_name: str = real_name if n is None or n == "1" else (PREFIX + str(int(n) - 1) + real_name)
            symbol = self.symbols[real_name] = SymbolEntry(real_name, name, old_name)

        if real_name in self.to_rename and symbol.old_name == symbol.scope_name:
            new_name: str = PREFIX + str(1 if n is None else (int(n) + 1)) + real_name
            symbol.old_name = symbol.scope_name
            symbol.scope_name = new_name

        if self.global_:
            symbol.global_ = True

        return symbol

    def must_rename(self, name: str, s: SymbolEntry) -> bool:
        return name == s.old_name and s.old_name != s.scope_name and s.real_name in self.to_rename

    def check(self, node: ast.AST, namespace: bool = False) -> None:
        self.check_namespace = False
        self.visit(node)
        if namespace:
            self.check_namespace = namespace
            super().generic_visit(node)

    def rename(self, names: set[str], node: ast.AST) -> None:
        self.to_rename.update(names)
        self.visit(node)
        self.to_rename.clear()

    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        if len(self.to_rename) > 0 or (
            self.check_namespace and not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            super().generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name if alias.asname is None else alias.asname
            if self.must_rename(name, s := self.update_symbol(name)):
                alias.asname = s.scope_name
            s.assigned = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.name if alias.asname is None else alias.asname
            if self.must_rename(name, s := self.update_symbol(name)):
                alias.asname = s.scope_name
            s.assigned = True

    def visit_Global(self, node: ast.Global) -> None:
        for i, name in enumerate(node.names):
            if self.must_rename(name, s := self.update_symbol(name)):
                node.names[i] = s.scope_name
            s.assigned = True
            s.global_ = True

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for i, name in enumerate(node.names):
            if self.must_rename(name, s := self.update_symbol(name)):
                node.names[i] = s.scope_name
            s.assigned = True

    def visit_Name(self, node: ast.Name) -> None:
        if self.must_rename(node.id, s := self.update_symbol(node.id)):
            node.id = s.scope_name

        if isinstance(node.ctx, ast.Load):
            s.used = True
        else:
            s.assigned = True

    def visit_arg(self, node: ast.arg) -> None:
        if self.must_rename(node.arg, s := self.update_symbol(node.arg)):
            node.arg = s.scope_name
        s.assigned = True
        s.annotation = node.annotation

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            if self.must_rename(node.target.id, s := self.update_symbol(node.target.id)):
                node.target.id = s.scope_name
            s.annotation = node.annotation


class SymbolTable:
    _parent: SymbolTable | None
    _visitor: SymbolTableVisitor

    def __init__(self, global_vars: list[str] | None = None):
        self._visitor = SymbolTableVisitor(global_vars)
        self._parent = None

    def parent(self) -> SymbolTable | None:
        return self._parent

    def check_namespace(self, node: ast.AST) -> SymbolTable:
        child: SymbolTable = self.new_child()
        child._visitor.check(node, namespace=True)
        return child

    def update(self, node: ast.AST) -> None:
        self._visitor.check(node)

    def new_child(self) -> SymbolTable:
        child: SymbolTable = SymbolTable()
        child._parent = self
        return child

    def rename(self, names: set[str], node: ast.AST) -> SymbolTable:
        if len(names) == 0:
            names = {PREFIX}
        child: SymbolTable = self.new_child()
        child._visitor.rename(names, node)
        return child

    def symbols(self) -> list[SymbolEntry]:
        return list(self._visitor.symbols.values())

    def identifiers(self) -> KeysView[str]:
        return self._visitor.symbols.keys()

    def __getitem__(self, name: str) -> SymbolEntry:
        return self._visitor.symbols[name]

    def get(self, name: str, parents: bool = False, module: bool = False, ann: bool = False) -> SymbolEntry | None:
        if name in self._visitor.symbols:
            s = self._visitor.symbols[name]
            if ann and not s.annotation:
                s.annotation = self.getAnnotation(name)
            return s
        if parents and self._parent and (not self._parent._visitor.global_ or module):
            return self._parent.get(name, parents)
        return None

    def getAnnotation(self, name: str) -> ast.expr | None:
        if name in self._visitor.symbols and (ann := self._visitor.symbols[name].annotation):
            return ann
        if self._parent:
            return self._parent.getAnnotation(name)
        return None

    def __contains__(self, name: str) -> bool:
        return name in self._visitor.symbols


def runtime_ast(name: str) -> ast.Attribute:
    return ast.Attribute(ast.Name("_omp"), name)


def omp_name(ctx: Context, raw_name: str = "") -> str:
    if len(raw_name) == 0:
        raw_name = "v"
    new_name = PREFIX + raw_name

    if (last := ctx.scope.omp_names.get(new_name, None)) is None:
        ctx.scope.omp_names[new_name] = 0
        return new_name

    ctx.scope.omp_names[new_name] += 1
    return new_name + str(last + 1)

