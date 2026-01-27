import ast
import os
from dataclasses import dataclass, field
from symtable import symtable as native_symtable

from omp4py.core.parser import Directive
from omp4py.core.preprocessor.transformers.symtable import SymbolTable

__all__ = ["Context", "Params", "SymbolTable", "global_symtable"]


def global_symtable(data: str, filename: str) -> "SymbolTable":
    return SymbolTable(list(native_symtable(data, filename, "exec").get_identifiers()))


def environ_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Params:
    pure: bool = field(init=False, default=environ_bool("OMP4PY_PURE", default=False))
    alias: str = "omp"
    debug: bool = environ_bool("OMP4PY_DEBUG", default=False)


class Context:
    params: Params
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
        params: Params,
    ) -> None:
        self.params = params
        self.filename = filename
        self.module = module
        self.is_module = is_module
        self.full_source = full_source
        self.symtable = SymbolTable()
        self.node_stack = [module]
        self.decorator = None
        self.directive = None
