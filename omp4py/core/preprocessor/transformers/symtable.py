from __future__ import annotations

import ast
import re
from collections.abc import KeysView
from dataclasses import dataclass

PREFIX: str = "_omp_"

var_check: re.Pattern[str] = re.compile(rf"^({PREFIX})?([0-9]+)?(.+)$")


def is_variable(name: str) -> bool:
    return not name.startswith(PREFIX) or name[len(PREFIX) : len(PREFIX) + 1].isdigit()


@dataclass
class SymbolEntry:
    scope_name: str
    old_name: str
    used: bool = False
    assigned: bool = False
    annotation: ast.expr | None = None

    @property
    def renamed(self):
        return self.scope_name != self.old_name


class SymbolTableVisitor(ast.NodeVisitor):
    symbols: dict[str, SymbolEntry]
    check_namespace: bool
    to_rename: set[str]

    def __init__(self, global_vars: list[str] | None):
        self.to_rename = set()
        self.check_namespace = False
        self.symbols = {}
        if global_vars is not None:
            name: str
            for name in global_vars:
                self.update_symbol(name).assigned = True

    def update_symbol(self, name: str) -> SymbolEntry:
        match: re.Match[str] | None = var_check.match(name)
        if not match:
            return SymbolEntry("", "")
        omp: str | None = match.group(1)
        n: str | None = match.group(2)
        real_name: str = match.group(3)
        if omp is not None and n is None:
            return SymbolEntry("", "")  # internal _omp_ var

        symbol: SymbolEntry
        if real_name in self.symbols:
            symbol = self.symbols[real_name]
        else:
            old_name: str = real_name if n is None or n == 1 else (PREFIX + str(int(n) - 1) + real_name)
            symbol = self.symbols[real_name] = SymbolEntry(name, old_name)

        if real_name in self.to_rename:
            new_name: str = PREFIX + str(1 if n is None else (int(n) + 1)) + real_name
            symbol.old_name = symbol.scope_name
            symbol.scope_name = new_name

        return symbol

    def check(self, node: ast.AST, namespace: bool = False) -> None:
        self.check_namespace = False
        self.visit(node)
        if namespace:
            self.check_namespace = namespace
            super().generic_visit(node)

    def rename(self, names: set[str], node: ast.AST) -> None:
        self.to_rename = names
        self.visit(node)

    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        if len(self.to_rename) > 0 or (
            self.check_namespace and not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            super().generic_visit(node)

    def generic_visit(self, node: ast.AST):
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (s := self.update_symbol(node.name)).renamed:
            node.name = s.scope_name
        s.assigned = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if (s := self.update_symbol(node.name)).renamed:
            node.name = s.scope_name
        s.assigned = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if (s := self.update_symbol(node.name)).renamed:
            node.name = s.scope_name
        s.assigned = True

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (s := self.update_symbol(alias.name if alias.asname is None else alias.asname)).renamed:
                alias.asname = s.scope_name
            s.assigned = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if (s := self.update_symbol(alias.name if alias.asname is None else alias.asname)).renamed:
                alias.asname = s.scope_name
            s.assigned = True

    def visit_Global(self, node: ast.Global) -> None:
        for i, name in enumerate(node.names):
            if (s := self.update_symbol(name)).renamed:
                node.names[i] = s.scope_name
            s.assigned = True

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for i, name in enumerate(node.names):
            if (s := self.update_symbol(name)).renamed:
                node.names[i] = s.scope_name
            s.assigned = True

    def visit_Name(self, node: ast.Name) -> None:
        if (s := self.update_symbol(node.id)).renamed:
            node.id = s.scope_name

        if isinstance(node.ctx, ast.Load):
            s.used = True
        else:
            s.assigned = True

    def visit_arg(self, node: ast.arg) -> None:
        if (s := self.update_symbol(node.arg)).renamed:
            node.arg = s.scope_name
        s.assigned = True
        s.annotation = node.annotation

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            if (s := self.update_symbol(node.target.id)).renamed:
                node.target.id = s.scope_name
            s.annotation = node.annotation


class SymbolTable:
    _parent: SymbolTable | None
    _visitor: SymbolTableVisitor

    def __init__(self, global_vars: list[str] | None = None):
        self._visitor = SymbolTableVisitor(global_vars)

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

    def rename(self, names: set[str], node: ast.AST) -> None:
        self._visitor.rename(names, node)

    def symbols(self) -> list[SymbolEntry]:
        return list(self._visitor.symbols.values())

    def identifiers(self) -> KeysView[str]:
        return self._visitor.symbols.keys()

    def __getattr__(self, name: str) -> SymbolEntry:
        return self._visitor.symbols[name]

    def __contains__(self, name: str) -> bool:
        return name in self._visitor.symbols
