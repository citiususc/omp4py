from __future__ import annotations

import ast
import typing
from dataclasses import dataclass, field

from omp4py.core.preprocessor.transformers.symtable import SymbolTable, global_symtable

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options
    from omp4py.core.parser import Directive


__all__ = ["Context", "SymbolTable"]

_module_storage: dict[str, ModuleStorage] = {}


@dataclass
class ModuleStorage:
    reductions: dict[str, tuple[ast.stmt, ast.stmt]] = field(default_factory=dict)


class Context:
    opt: Options
    filename: str
    full_source: str
    module: ast.Module
    namespace: int
    symtable: SymbolTable
    uname_i: int
    node_stack: list[ast.AST]
    module_storage: ModuleStorage
    directives: dict[ast.With | ast.Expr, Directive]

    def __init__(
        self,
        full_source: str,
        filename: str,
        module: ast.Module,
        is_module: bool,
        opt: Options,
    ) -> None:
        self.opt = opt
        self.filename = filename
        self.module = module
        self.is_module = is_module
        self.full_source = full_source
        self.symtable = global_symtable(self.full_source, self.filename)
        self.uname_i = 0
        if self.is_module:
            self.module_storage = ModuleStorage()
        else:
            self.module_storage = _module_storage.setdefault(self.filename, ModuleStorage())
        self.node_stack = [module]
        self.directives = {}
