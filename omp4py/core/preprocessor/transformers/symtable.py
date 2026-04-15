"""Symbol table and name management utilities for the `omp4py` core preprocessor.

This module provides the symbol table infrastructure used during source
code analysis and transformation in `omp4py`. It is responsible for tracking
variable usage, assignments, annotations, and scope relationships while
processing the Python AST.

The symbol table is a key component in enabling OpenMP-like semantics,
allowing the preprocessor to reason about variable scoping, detect conflicts,
and safely perform transformations such as variable renaming and
thread-private handling.

Additionally, this module includes helper utilities for generating
runtime-specific identifiers (`omp_name`) and constructing AST nodes
targeting the runtime API (`runtime_ast`).
"""

from __future__ import annotations

import ast
import builtins
import re
import typing
from collections.abc import KeysView
from dataclasses import dataclass
from symtable import symtable as native_symtable

if typing.TYPE_CHECKING:
    from omp4py.core.preprocessor.transformers.context import Context

__all__ = ["SymbolEntry", "SymbolTable", "global_symtable", "omp_name", "runtime_ast"]

PREFIX: str = "_omp_"

var_check: re.Pattern[str] = re.compile(rf"^({PREFIX})?([0-9]+)?(.+)$")


def global_symtable(data: str, filename: str) -> SymbolTable:
    """Create a global symbol table from source code.

    This function builds an initial symbol table using Python's native
    `symtable` module, extracting top-level identifiers from the given
    source code. The resulting table is used as the root scope for
    subsequent transformations.

    Args:
        data (str): Source code to analyze.
        filename (str): Name of the source file.

    Returns:
        SymbolTable: Initialized global symbol table.
    """
    return SymbolTable(list(native_symtable(data, filename, "exec").get_identifiers()))


def is_variable(name: str) -> bool:
    """Determine whether a name corresponds to a user-level variable.

    This function filters out internal runtime-generated identifiers,
    which are prefixed with `_omp_` and may include numeric suffixes.

    Args:
        name (str): Identifier name.

    Returns:
        bool: True if the name represents a user variable, False otherwise.
    """
    return not name.startswith(PREFIX) or name[len(PREFIX) : len(PREFIX) + 1].isdigit()


@dataclass
class SymbolEntry:
    """Represents a symbol entry within a scope.

    A `SymbolEntry` stores metadata about a variable encountered during
    AST traversal, including its original name, renamed version within
    the current scope, and usage information.

    This structure enables the preprocessor to track variable lifetimes,
    detect conflicts, and apply transformations required to enforce
    OpenMP-like semantics.

    Attributes:
        real_name (str): Canonical variable name (without prefixes).
        scope_name (str): Name used in the current scope (possibly renamed).
        old_name (str): Previous name before transformation.
        used (bool): Whether the variable is read.
        __assigned (bool): Whether the variable is assigned. (it uses a setter)
        global_ (bool): Whether the variable is declared as global.
        dec_global (bool): Whether the variable has been explicitly declared
            as `global` in the current scope.
        dec_nonlocal (bool): Whether the variable has been explicitly declared
            as `nonlocal` in the current scope.
        threadprivate (bool): Whether the variable is thread-private.
        annotation (ast.expr | None): Type annotation, if provided.
    """

    real_name: str
    scope_name: str
    old_name: str
    used: bool = False
    __assigned: bool = False
    global_: bool = False
    dec_global: bool = False
    dec_nonlocal: bool = False
    threadprivate: bool = False
    annotation: ast.expr | None = None

    @property
    def assigned(self) -> bool:
        """Whether the variable has been assigned a value.

        Setting this property may affect the `global_` flag: if the symbol
        is marked as global but has not been explicitly declared with a
        `global` statement, assigning to it will clear the `global_` flag.
        """
        return self.__assigned

    @assigned.setter
    def assigned(self, value: bool) -> None:
        if self.global_ and not self.dec_global:
            self.global_ = False
        self.__assigned = value

    @property
    def renamed(self) -> bool:
        """Check whether the symbol has been renamed in the current scope.

        Returns:
            bool: True if the symbol name differs from its original name,
            False otherwise.
        """
        return self.scope_name != self.old_name


class SymbolTableVisitor(ast.NodeVisitor):
    """AST visitor for symbol collection and transformation.

    This visitor traverses the Python AST to build and update symbol
    entries, tracking variable usage, assignments, and annotations.

    It also performs controlled renaming of identifiers to avoid naming
    conflicts introduced during parallel transformation, ensuring that
    generated code remains semantically correct under OpenMP-like rules.

    The visitor supports:
    - Symbol registration and lookup
    - Usage and assignment tracking
    - Scoped renaming of variables
    - Propagation of global and nonlocal declarations

    Attributes:
        parent (SymbolTableVisitor | None): Parent scope.
        symbols (dict[str, SymbolEntry]): Collected symbols in the current scope.
        check_namespace (bool): Whether to enforce namespace traversal rules.
        to_rename (set[str]): Set of identifiers marked for renaming.
        global_ (bool): Whether the current scope is global.
    """

    parent: SymbolTableVisitor | None
    symbols: dict[str, SymbolEntry]
    check_namespace: bool
    to_rename: set[str]
    global_: bool

    def __init__(self, global_vars: list[str] | None) -> None:
        """Initialize the symbol table visitor.

        This constructor sets up the internal state used to track symbols,
        renaming operations, and namespace traversal.

        If a list of global variables is provided, the visitor is initialized
        in global mode. In this mode, all built-in names and provided global
        identifiers are pre-registered as assigned symbols.

        Args:
            global_vars (list[str] | None): List of global variable names to
                initialize in the symbol table. If None, the visitor operates
                in a local scope.
        """
        self.parent = None
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
        """Register or update a symbol entry.

        This method register names, resolves internal naming
        patterns, and ensures that each logical variable is represented
        by a unique `SymbolEntry`.

        It also handles name versioning for variables that require
        renaming due to scope conflicts.

        Args:
            name (str): Identifier name.

        Returns:
            SymbolEntry: Updated or newly created symbol entry.
        """
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
            # Check symbol in parent tables
            parent_table: SymbolTableVisitor | None = self.parent
            while parent_table and real_name not in parent_table.symbols:
                parent_table = parent_table.parent
            if parent_table:
                parent = parent_table.symbols[real_name]
                symbol.annotation = parent.annotation
                symbol.threadprivate = parent.threadprivate
                symbol.global_ = parent.global_

        if real_name in self.to_rename and symbol.old_name == symbol.scope_name:
            new_name: str = PREFIX + str(1 if n is None else (int(n) + 1)) + real_name
            symbol.old_name = symbol.scope_name
            symbol.scope_name = new_name

        if self.global_:
            symbol.dec_global = True
            symbol.global_ = True

        return symbol

    def must_rename(self, name: str, s: SymbolEntry) -> bool:
        """Determine whether a symbol must be renamed.

        Args:
            name (str): Current identifier name.
            s (SymbolEntry): Associated symbol entry.

        Returns:
            bool: True if the identifier should be renamed.
        """
        return name == s.old_name and s.old_name != s.scope_name and s.real_name in self.to_rename

    def check(self, node: ast.AST, namespace: bool = False) -> None:
        """Check an AST node to collect symbol information.

        Optionally enables namespace-aware traversal, which enforces
        stricter rules on how nested scopes are visited.

        Args:
            node (ast.AST): AST node to analyze.
            namespace (bool): Whether to enable namespace checking.
        """
        self.check_namespace = False
        self.visit(node)
        if namespace:
            self.check_namespace = namespace
            super().generic_visit(node)

    def rename(self, names: set[str], node: ast.AST) -> None:
        """Rename selected identifiers within an AST subtree.

        This method marks a set of identifiers for renaming and applies
        the transformation during traversal.

        Args:
            names (set[str]): Identifiers to rename.
            node (ast.AST): AST node to transform.
        """
        self.to_rename.update(names)
        self.visit(node)
        self.to_rename.clear()

    def visit(self, node: ast.AST) -> None:
        """Visit an AST node with optional extended traversal.

        This method extends the default `NodeVisitor.visit` behavior by
        conditionally performing a secondary traversal using
        `generic_visit`.

        A second traversal is triggered when:
        - Renaming is active, or
        - Namespace checking is enabled and the node is not a scope-defining
          construct (e.g., function or class).

        Args:
            node (ast.AST): AST node to visit.
        """
        super().visit(node)
        if len(self.to_rename) > 0 or (
            self.check_namespace and not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            super().generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Disable default generic traversal.

        The standard `NodeVisitor.generic_visit` is intentionally overridden
        to prevent automatic traversal. Traversal is explicitly controlled
        in `visit` to support fine-grained handling of renaming and
        namespace-aware analysis.
        """

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process a class definition.

        Registers the class name as an assigned symbol and applies renaming
        if required.

        Args:
            node (ast.ClassDef): Class definition node.
        """
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process an asynchronous function definition.

        Registers the function name as an assigned symbol and applies
        renaming if required.

        Args:
            node (ast.AsyncFunctionDef): Async function definition node.
        """
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process a function definition.

        Registers the function name as an assigned symbol and applies
        renaming if required.

        Args:
            node (ast.FunctionDef): Function definition node.
        """
        if self.must_rename(node.name, s := self.update_symbol(node.name)):
            node.name = s.scope_name
        s.assigned = True

    def visit_Import(self, node: ast.Import) -> None:
        """Process an import statement.

        Registers imported names as assigned symbols and applies renaming
        to aliases when required.

        Args:
            node (ast.Import): Import statement node.
        """
        for alias in node.names:
            name = alias.name if alias.asname is None else alias.asname
            if self.must_rename(name, s := self.update_symbol(name)):
                alias.asname = s.scope_name
            s.assigned = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process a from-import statement.

        Registers imported names as assigned symbols and applies renaming
        to aliases when required.

        Args:
            node (ast.ImportFrom): From-import statement node.
        """
        for alias in node.names:
            name = alias.name if alias.asname is None else alias.asname
            if self.must_rename(name, s := self.update_symbol(name)):
                alias.asname = s.scope_name
            s.assigned = True

    def visit_Global(self, node: ast.Global) -> None:
        """Process a global declaration.

        Marks declared identifiers as global and assigned, and applies
        renaming if required.

        Args:
            node (ast.Global): Global declaration node.
        """
        for i, name in enumerate(node.names):
            if self.must_rename(name, s := self.update_symbol(name)):
                node.names[i] = s.scope_name
            s.assigned = True
            s.global_ = True
            s.dec_global = True

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Process a nonlocal declaration.

        Marks declared identifiers as assigned and applies renaming if
        required.

        Args:
            node (ast.Nonlocal): Nonlocal declaration node.
        """
        for i, name in enumerate(node.names):
            if self.must_rename(name, s := self.update_symbol(name)):
                node.names[i] = s.scope_name
            s.assigned = True
            s.dec_nonlocal = True

    def visit_Name(self, node: ast.Name) -> None:
        """Process a variable reference.

        Updates symbol usage information depending on the context
        (load or store) and applies renaming if required.

        Args:
            node (ast.Name): Name node.
        """
        if self.must_rename(node.id, s := self.update_symbol(node.id)):
            node.id = s.scope_name

        if isinstance(node.ctx, ast.Load):
            s.used = True
        else:
            s.assigned = True

    def visit_arg(self, node: ast.arg) -> None:
        """Process a function argument.

        Registers the argument as an assigned symbol, applies renaming
        if required, and stores its type annotation if present.

        Args:
            node (ast.arg): Argument node.
        """
        if self.must_rename(node.arg, s := self.update_symbol(node.arg)):
            node.arg = s.scope_name
        s.assigned = True
        match node.annotation:
            case ast.Attribute(value=ast.Name(id="_omp")):
                return
        s.annotation = node.annotation

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process an annotated assignment.

        Registers the target variable, applies renaming if required,
        and stores the associated type annotation.

        Args:
            node (ast.AnnAssign): Annotated assignment node.
        """
        if isinstance(node.target, ast.Name):
            if self.must_rename(node.target.id, s := self.update_symbol(node.target.id)):
                node.target.id = s.scope_name
            match node.annotation:
                case ast.Attribute(value=ast.Name(id="_omp")):
                    return
            s.annotation = node.annotation


class SymbolTable:
    """Hierarchical symbol table for AST-based transformations.

    This class provides a scoped symbol table abstraction used during
    preprocessing. Each instance represents a scope and may reference
    a parent scope, forming a hierarchy consistent with Python's
    lexical scoping rules.

    Attributes:
        _parent (SymbolTable | None): Parent scope.
        _visitor (SymbolTableVisitor): Underlying AST visitor.
    """

    _parent: SymbolTable | None
    _visitor: SymbolTableVisitor

    def __init__(self, global_vars: list[str] | None = None) -> None:
        """Initialize a new symbol table.

        This constructor creates a symbol table associated with a new
        `SymbolTableVisitor`. If a list of global variables is provided,
        the table is initialized in global mode, marking builtins and
        given identifiers as assigned.

        Args:
            global_vars (list[str] | None): List of global variable names
                to pre-register in the symbol table. If None, the table
                is initialized as a non-global (local) scope.
        """
        self._visitor = SymbolTableVisitor(global_vars)
        self._parent = None

    def parent(self) -> SymbolTable | None:
        """Return the parent scope of this symbol table.

        This method provides access to the enclosing scope, allowing
        traversal of the symbol table hierarchy during lookup or
        analysis.

        Returns:
            SymbolTable | None: Parent symbol table, or None if this is
            the root scope.
        """
        return self._parent

    def check_namespace(self, node: ast.AST) -> SymbolTable:
        """Analyze a node with namespace-aware traversal.

        Creates a child scope and performs symbol collection while
        respecting namespace boundaries.

        Args:
            node (ast.AST): AST node to analyze.

        Returns:
            SymbolTable: Child symbol table containing collected symbols.
        """
        child: SymbolTable = self.new_child()
        child._visitor.check(node, namespace=True)
        return child

    def update(self, node: ast.AST) -> None:
        """Update the symbol table with information from an AST node.

        This function is called by the preprocessor during their tree traversal.

        Args:
            node (ast.AST): AST node to process.
        """
        self._visitor.check(node)

    def new_child(self) -> SymbolTable:
        """Create a new child symbol table.

        Returns:
            SymbolTable: New child scope.
        """
        child: SymbolTable = SymbolTable()
        child._parent = self
        child._visitor.parent = self._visitor
        return child

    def rename(self, names: set[str], node: ast.AST) -> SymbolTable:
        """Create a child scope with renamed identifiers.

        Args:
            names (set[str]): Identifiers to rename.
            node (ast.AST): AST node to transform.

        Returns:
            SymbolTable: Child symbol table with applied renaming.
        """
        if len(names) == 0:
            names = {PREFIX}
        child: SymbolTable = self.new_child()
        child._visitor.rename(names, node)
        return child

    def symbols(self) -> list[SymbolEntry]:
        """Return all symbol entries defined in this scope.

        Returns:
            list[SymbolEntry]: List of symbol entries collected in the
            current scope.
        """
        return list(self._visitor.symbols.values())

    def identifiers(self) -> KeysView[str]:
        """Return the identifiers defined in this scope.

        This provides a view over the symbol names without exposing
        the underlying symbol entries.

        Returns:
            KeysView[str]: View of identifier names in the current scope.
        """
        return self._visitor.symbols.keys()

    def __getitem__(self, name: str) -> SymbolEntry:
        """Retrieve a symbol entry by name.

        This method provides direct access to symbols in the current
        scope and raises an exception if the symbol is not found.

        Args:
            name (str): Identifier name.

        Returns:
            SymbolEntry: Corresponding symbol entry.

        Raises:
            KeyError: If the identifier does not exist in this scope.
        """
        return self._visitor.symbols[name]

    def get(self, name: str, parents: bool = False, module: bool = False) -> SymbolEntry | None:
        """Retrieve a symbol entry by name.

        Supports lookup across parent scopes and optional annotation
        resolution.

        Args:
            name (str): Identifier name.
            parents (bool): Whether to search parent scopes.
            module (bool): Whether to allow module-level lookup.

        Returns:
            SymbolEntry | None: Matching symbol entry, if found.
        """
        if name in self._visitor.symbols:
            return self._visitor.symbols[name]
        if parents and self._parent and (not self._parent._visitor.global_ or module):  # noqa: SLF001
            return self._parent.get(name, parents)
        return None

    def __contains__(self, name: str) -> bool:
        """Check whether a symbol exists in the current scope.

        This method only checks the current symbol table and does not
        traverse parent scopes.

        Args:
            name (str): Identifier name.

        Returns:
            bool: True if the symbol exists in this scope, False otherwise.
        """
        return name in self._visitor.symbols


def runtime_ast(name: str) -> ast.Attribute:
    """Construct an AST node referencing the runtime API.

    This helper generates an attribute access node targeting the
    `_omp` runtime object, used to invoke runtime functions during
    code generation.

    Args:
        name (str): Runtime attribute name.

    Returns:
        ast.Attribute: AST node representing `_omp.<name>`.
    """
    return ast.Attribute(ast.Name("_omp"), name)


def omp_name(ctx: Context, raw_name: str = "") -> str:
    """Generate a unique runtime-managed identifier.

    This function creates deterministic, collision-free variable names
    used internally by the preprocessor. Names are prefixed with `_omp_`
    and may include numeric suffixes to ensure uniqueness within a scope.

    Args:
        ctx (Context): Transformation context containing scope state.
        raw_name (str): Base name for the identifier.

    Returns:
        str: Generated unique identifier.
    """
    if len(raw_name) == 0:
        raw_name = "v"
    new_name = PREFIX + raw_name

    if (last := ctx.scope.omp_names.get(new_name, None)) is None:
        ctx.scope.omp_names[new_name] = 0
        return new_name

    ctx.scope.omp_names[new_name] += 1
    return new_name + str(last + 1)
