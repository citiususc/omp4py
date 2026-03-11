from __future__ import annotations

import ast
import typing
from collections.abc import Callable
from dataclasses import dataclass, field

from omp4py.core.preprocessor.transformers.symtable import SymbolEntry, SymbolTable, global_symtable

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options
    from omp4py.core.parser import Directive


__all__ = ["Context", "Scope", "SymbolTable"]

_module_storage: dict[str, ModuleStorage] = {}


@dataclass
class ModuleStorage:
    reductions: dict[str, tuple[ast.stmt, ast.stmt]] = field(default_factory=dict)
    threadprivate: dict[str, SymbolEntry] = field(default_factory=dict)


@dataclass
class Scope:
    node: ast.AST
    symtable: SymbolTable
    omp_names: dict[str, int] = field(default_factory=dict)

    def new_child(self, node: ast.AST) -> Scope:
        return Scope(node, self.symtable.new_child(), self.omp_names.copy())


@dataclass
class Context:
    full_source: str
    filename: str
    module: ast.Module
    is_module: bool
    opt: Options

    scope: Scope = field(init=False)
    module_storage: ModuleStorage = field(init=False)
    node_stack: list[ast.AST] = field(init=False, default_factory=list)
    finalizers: list[Callable[[], None]] = field(init=False, default_factory=list)
    directives: dict[ast.With | ast.Expr, Directive] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.scope = Scope(self.module, global_symtable(self.full_source, self.filename))
        if self.is_module:
            self.module_storage = ModuleStorage()
        else:
            self.module_storage = _module_storage.setdefault(self.filename, ModuleStorage())
        self.node_stack.append(self.module)

    @property
    def symtable(self) -> SymbolTable:
        return self.scope.symtable
