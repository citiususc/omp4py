from omp4py.core.directive.schema import Directive
import ast
import os
import symtable
from dataclasses import dataclass, field

__all__ = ["Context", "Params"]


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
    symtable_stack: list[symtable.SymbolTable]
    node_stack: list[ast.AST]
    decorator: ast.expr | None
    directive: Directive | None

    def __init__(self, full_source: str, filename: str, module: ast.Module, namespace: int, params: Params) -> None:
        self.params = params
        self.filename = filename
        self.module = module
        self.namespace = namespace
        self.full_source = full_source
        self.symtable_stack = []
        self.node_stack = [module]
        self.decorator = None
        self.directive = None

