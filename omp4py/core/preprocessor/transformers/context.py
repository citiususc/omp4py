"""Transformation context and scope store for the `omp4py` preprocessor.

This module defines the execution context used during AST traversal in the
`omp4py` core preprocessor. It provides structures to maintain state across
transformations, including symbol tables, scope hierarchy, and module-level
storage.

The context is responsible for coordinating all information required during
code transformation, such as directive handling, symbol resolution, and
runtime metadata generation.

It also uses persistent module-level state (`ModuleStorage`) to support OpenMP-like
semantics such as reductions and thread-private variables that can be declared
in same module but in diferent decorated functions.
"""

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
    """Persistent storage for module-level transformation state.

    This structure stores information that must persist across multiple
    transformation passes or function-level contexts within the same module.

    It is primarily used to manage OpenMP-like constructs that require
    global coordination, such as reductions and thread-private variables.

    Attributes:
        reductions (dict[str, tuple[ast.stmt, ast.stmt]]): Mapping of reduction
            identifiers to their initialization and combination statements.
        threadprivate (dict[str, SymbolEntry]): Mapping of thread-private
            variable names to their corresponding symbol entries.
    """

    reductions: dict[str, tuple[ast.stmt, ast.stmt]] = field(default_factory=dict)
    threadprivate: dict[str, SymbolEntry] = field(default_factory=dict)


@dataclass
class Scope:
    """Represents a transformation scope during AST traversal.

    A `Scope` encapsulates the current AST node, its associated symbol
    table, and the mapping of generated runtime names. Scopes form a
    hierarchy aligned with the structure of the AST.

    This abstraction allows the preprocessor to manage variable visibility,
    renaming, and symbol resolution in a way consistent with Python's
    lexical scoping rules.

    Attributes:
        node (ast.AST): AST node associated with this scope.
        symtable (SymbolTable): Symbol table for the current scope.
        omp_names (dict[str, int]): Mapping used to generate unique
            runtime-managed identifiers.
    """

    node: ast.AST
    symtable: SymbolTable
    omp_names: dict[str, int] = field(default_factory=dict)

    def new_child(self, node: ast.AST) -> Scope:
        """Create a child scope for a nested AST node.

        The child scope inherits the symbol table hierarchy and copies
        the current runtime name mapping to preserve deterministic name
        generation.

        Args:
            node (ast.AST): AST node defining the new scope.

        Returns:
            Scope: Newly created child scope.
        """
        return Scope(node, self.symtable.new_child(), self.omp_names.copy())


@dataclass
class Context:
    """Transformation context for the `omp4py` preprocessor.

    The `Context` object maintains all state required during AST traversal
    and transformation. It acts as the central coordination structure,
    linking source code, symbol tables, scopes, and directive handling.

    It is used by all transformation constructs to ensure consistent behavior
    and to propagate information such as variable state, directives, and
    runtime metadata.

    Attributes:
        full_source (str): Original source code.
        filename (str): Name of the source file.
        module (ast.Module): Root AST node.
        is_module (bool): Whether this context represents a module-level or function-level transformation.
        opt (Options): Compilation and transformation options.

        scope (Scope): Current active scope.
        module_storage (ModuleStorage): Persistent module-level storage.
        node_stack (list[ast.AST]): Stack of visited AST nodes.
        finalizers (list[Callable[[], None]]): Deferred actions executed after end of AST traversal.
        directives (dict[ast.With | ast.Expr, Directive]): Mapping of AST nodes with Open-MP parsed directives.
    """

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
        """Initialize derived context state after construction.

        This method sets up the root scope using a global symbol table
        and initializes module-level storage.

        If the context represents a full module, a new storage instance
        is created. Otherwise, an existing storage associated with the
        filename is reused, allowing shared state across multiple
        transformations within the same module.

        It also initializes the node stack with the root module node.
        """
        self.scope = Scope(self.module, global_symtable(self.full_source, self.filename))
        if self.is_module:
            self.module_storage = ModuleStorage()
        else:
            self.module_storage = _module_storage.setdefault(self.filename, ModuleStorage())
        self.node_stack.append(self.module)

    @property
    def symtable(self) -> SymbolTable:
        """Return the current symbol table.

        This provides convenient access to the symbol table associated
        with the active scope.

        Returns:
            SymbolTable: Current scope symbol table.
        """
        return self.scope.symtable
