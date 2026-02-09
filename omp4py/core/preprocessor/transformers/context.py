from __future__ import annotations

import ast
from symtable import symtable as native_symtable

from omp4py.core.parser import Directive
from omp4py.core.preprocessor.transformers.symtable import SymbolTable
from omp4py.core.options import Options

__all__ = ["Context", "SymbolTable", "global_symtable"]


def global_symtable(data: str, filename: str) -> SymbolTable:
    return SymbolTable(list(native_symtable(data, filename, "exec").get_identifiers()))


class Context:
    opt: Options
    filename: str
    full_source: str
    module: ast.Module
    namespace: int
    symtable: SymbolTable
    node_stack: list[ast.AST]
    decorator: ast.expr | None
    directive: Directive | None

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
        self.symtable = SymbolTable()
        self.node_stack = [module]
        self.decorator = None
        self.directive = None
