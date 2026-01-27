import ast
import re
from collections.abc import Callable, KeysView
from dataclasses import dataclass
from typing import Optional

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
                self.update_symbol(name, lambda x: None).assigned = True

    def update_symbol(self, name: str, rename: Callable[[str], None]) -> SymbolEntry:
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
            old_name: str = real_name if n is None else (PREFIX + str(int(n) - 1) + real_name)
            symbol = self.symbols[real_name] = SymbolEntry(name, old_name)

        if real_name in self.to_rename:
            new_name: str = PREFIX + str(1 if n is None else (int(n) + 1)) + real_name
            symbol.old_name = symbol.scope_name
            symbol.scope_name = new_name
            rename(new_name)
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
        self.update_symbol(node.name, lambda x: setattr(node, "name", x)).assigned = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.update_symbol(node.name, lambda x: setattr(node, "name", x)).assigned = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.update_symbol(node.name, lambda x: setattr(node, "name", x)).assigned = True

    def visit_Import(self, node: ast.Import) -> None:
        alias: ast.alias
        for alias in node.names:
            name: str = alias.name if alias.asname is None else alias.asname
            self.update_symbol(name, lambda x: setattr(node, "asname", x)).assigned = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        alias: ast.alias
        for alias in node.names:
            name: str = alias.name if alias.asname is None else alias.asname
            self.update_symbol(name, lambda x: setattr(alias, "asname", x)).assigned = True  # noqa: B023

    def visit_Global(self, node: ast.Global) -> None:
        i: int
        name: str
        for i, name in enumerate(node.names):
            self.update_symbol(name, lambda x: node.names.__setitem__(i, x)).assigned = True  # noqa: B023

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        i: int
        name: str
        for i, name in enumerate(node.names):
            self.update_symbol(name, lambda x: node.names.__setitem__(i, x)).assigned = True  # noqa: B023

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.update_symbol(node.id, lambda x: setattr(node, "id", x)).used = True
        else:
            self.update_symbol(node.id, lambda x: setattr(node, "id", x)).assigned = True

    def visit_arg(self, node: ast.arg) -> None:
        s: SymbolEntry = self.update_symbol(node.arg, lambda x: setattr(node, "arg", x))
        s.assigned = True
        s.annotation = node.annotation

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.update_symbol(node.target.id, lambda x: setattr(node.target, "id", x)).annotation = node.annotation


class SymbolTable:
    _parent: Optional["SymbolTable"]
    _visitor: SymbolTableVisitor

    def __init__(self, global_vars: list[str] | None = None):
        self._visitor = SymbolTableVisitor(global_vars)

    def check_namespace(self, node: ast.AST) -> "SymbolTable":
        child: SymbolTable = self.new_child()
        child._visitor.check(node, namespace=True)
        return child

    def update(self, node: ast.AST) -> None:
        self._visitor.check(node)

    def new_child(self) -> "SymbolTable":
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
