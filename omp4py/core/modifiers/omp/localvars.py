"""Modifier that reads the variables of the enclosing scopes through local names.

The block of a `parallel` is compiled into a function nested in yours, so the variables that the block uses are
*free variables* (closure cells) or globals. On a free-threaded interpreter, reading them from many threads at once
makes the threads compete for them, and a loop that reads a shared number can be slower with eight threads than
with one. Reading a local variable does not have that cost.

This modifier binds, at the beginning of every function nested in another one, a local name to each variable that
the function reads and never assigns, and uses that name instead:

    def _omp_parallel():
        _omp_1w = w                    # the same object, bound once per call
        ...
        x = (i + 0.5) * _omp_1w        # a local read

It is the same object, so mutable objects are still shared. The new names are the ones that the symbol table of the
transformer creates when it renames a variable. The modifier is off by default, because binding a variable when the
function starts changes what the program does if some code that the function calls assigns that variable while it
runs. Enable it with `omp(modifiers={"omp_local_aliases": True})`. It does not run when the code is compiled.
"""

from __future__ import annotations

import ast
import builtins
import typing

from omp4py.core.modifiers.engine import Modifier, modifier
from omp4py.core.preprocessor.transformers.symtable import SymbolTable

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options

__all__ = []

# Names whose meaning depends on the namespace of the function, so the function is left as it is.
DYNAMIC: frozenset[str] = frozenset({"exec", "eval", "locals", "globals", "vars", "dir", "super"})


@modifier
class LocalAliases(Modifier, name="omp_local_aliases", default=False):
    """Modifier that reads the variables that a nested function does not assign through local names.

    See the module documentation.

    Attributes:
        depth (int): Number of functions that contain the node being visited.
    """

    depth: int = 0

    @classmethod
    def should_run(cls, options: Options) -> bool:
        """Determine whether the modifier should be applied.

        Args:
            options (Options): Active preprocessing configuration.

        Returns:
            bool: `True` if the modifier is enabled (it is not by default) and the code is not compiled.
        """
        return super().should_run(options) and not options.compile

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        """Bind local names in the function if it is nested in another one.

        Args:
            node (ast.FunctionDef): Function definition.

        Returns:
            ast.FunctionDef: The same function, modified.
        """
        if self.depth > 0:  # before the functions inside it, so their reads of its names go through its aliases
            localize(node)
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node


def localize(func: ast.FunctionDef) -> None:
    """Bind a local name to every variable that `func` reads and never assigns.

    Args:
        func (ast.FunctionDef): Function nested in another one.
    """
    inner = list(ast.walk(func))[1:]
    if any(isinstance(n, (ast.ClassDef, ast.Yield, ast.YieldFrom, ast.Await)) for n in inner):
        return  # a class body does not look the names up as a function does, and a generator runs in pieces
    if any(isinstance(n, ast.Name) and n.id in DYNAMIC for n in inner):
        return

    # Names that the symbol table does not see as assigned: the ones that a nested function declares `nonlocal` or
    # `global` (assigning them there must change the original variable) and the ones that `except` and `match` bind.
    bound: set[str] = set()
    for n in inner:
        match n:
            case ast.Global(names=names) | ast.Nonlocal(names=names):
                bound.update(names)
            case ast.ExceptHandler(name=str(name)) | ast.MatchAs(name=str(name)) | ast.MatchStar(name=str(name)):
                bound.add(name)
            case ast.MatchMapping(rest=str(name)):
                bound.add(name)

    table = SymbolTable().check_namespace(func)
    symbols = [
        s
        for s in table.symbols()
        if s.used
        and not s.assigned
        and s.real_name not in bound
        and s.real_name != "_omp"
        and not hasattr(builtins, s.real_name)
    ]
    if not symbols:
        return

    # One renaming for everything: the transformer shares some nodes between several statements, and the table
    # renames a name only once. The tuple holds the names as the code has them, to get their new names.
    names = {s.real_name for s in symbols}
    old = [s.scope_name for s in symbols]
    new = ast.Tuple([ast.Name(name) for name in old])
    table.rename(names, ast.Module([*func.body, ast.Expr(new)], []))

    aliases = [
        ast.copy_location(ast.Assign([ast.Name(alias.id, ast.Store())], ast.Name(name)), func)
        for name, alias in zip(old, new.elts, strict=True)
    ]
    first = next((i for i, stmt in enumerate(func.body) if not isinstance(stmt, (ast.Global, ast.Nonlocal))), 0)
    func.body[first:first] = aliases  # after the `nonlocal` and `global` declarations
    ast.fix_missing_locations(func)
