"""Typing-related AST modifiers for compiled and pure-Python execution modes.

This module defines modifiers that adapt generated AST code depending on
the selected execution mode.

The transformations implemented here specialize type-related helper calls
used internally by `omp4py`. These helpers provide a independent
abstraction layer for type operations such as casting and runtime type
queries, allowing generated code to remain compatible across both
pure-Python and compiled execution modes.

Two execution strategies are supported:

- Pure Python mode:
  Runtime helper calls are simplified or removed to avoid unnecessary
  overhead during interpretation.

- Cython compilation mode:
  Generic `omp4py` typing helpers are rewritten into native Cython
  equivalents so the generated code can benefit from static typing,
  optimized casts, and improved compiled performance.

The modifiers defined in this module are automatically enabled depending
on the active compilation configuration.
"""

from __future__ import annotations

import ast
import typing

from omp4py.core.modifiers.engine import Modifier, modifier

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options

__all__ = []


@modifier
class CythonTypes(Modifier, name="omp_cython_types", default=True):
    """Modifier that rewrites generic typing helpers into Cython equivalents.

    This modifier is enabled only when the generated code is compiled
    using the Cython compiler.

    The modifier replaces helper calls such as `_omp.tp_cast` and
    `_omp.tp_typeof` with direct calls to the native `cython.cast`
    and `cython.typeof` APIs. This allows the generated code to
    take advantage of Cython typing optimizations during
    compilation.

    It also injects the required `cython` import into the transformed
    module and applies temporary workarounds for known Cython
    limitations related to nested cast expressions.
    """

    @classmethod
    def should_run(cls, options: Options) -> bool:
        """Determine whether the modifier should be applied.

        The modifier is only enabled when:
        - The modifier itself is not disabled in configuration.
        - Compilation mode is active.
        - The selected compiler backend is `cython`.

        Args:
            options (Options):
                Active preprocessing configuration.

        Returns:
            bool:
                `True` if the modifier should run, otherwise `False`.
        """
        return super().should_run(options) and options.compile and options.compiler.lower() == "cython"

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Insert the required Cython import into the module.

        A generated import equivalent to:

            import cython as _omp_cython

        is inserted at the beginning of the module so rewritten typing
        helpers can reference the Cython runtime API.

        Args:
            node (ast.Module):
                Module AST node being transformed.

        Returns:
            ast.Module:
                Updated module node.
        """
        cython = ast.Import([ast.alias("cython", "_omp_cython")])
        ast.fix_missing_locations(cython)
        node.body.insert(0, cython)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Simplify nested cast/typeof helper patterns.

        This transformation removes expressions matching the form:

            tp_cast(tp_typeof(x), y)

        and replaces them directly with `y`.

        The optimization exists as a temporary workaround for
        cython/issues/7683.

        Args:
            node (ast.Call):
                Call expression node.

        Returns:
            ast.expr:
                Transformed expression node.
        """
        if len(node.args) == 2:
            match node.func:
                case ast.Attribute(ast.Name("_omp"), "tp_cast"):
                    match node.args[0]:
                        case ast.Call(ast.Attribute(ast.Name("_omp"), "tp_typeof")):
                            return node.args[1]

        return typing.cast("ast.expr", self.generic_visit(node))

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Rewrite generic typing helpers into native Cython APIs.

        The following replacements are applied:

        - `_omp.tp_typeof` -> `_omp_cython.typeof`
        - `_omp.tp_cast` -> `_omp_cython.cast`

        Args:
            node (ast.Attribute):
                Attribute access node.

        Returns:
            ast.Attribute:
                Transformed attribute node.
        """
        match node:
            case ast.Attribute(ast.Name("_omp"), "tp_typeof"):
                node.attr = "typeof"
                node.value = ast.Name("_omp_cython")
            case ast.Attribute(ast.Name("_omp"), "tp_cast"):
                node.attr = "cast"
                node.value = ast.Name("_omp_cython")
            case _:
                return node
        return ast.fix_missing_locations(node)


@modifier
class PurePythonTypes(Modifier, name="omp_pure_types", default=True):
    """Modifier that simplifies typing helpers in pure-Python mode.

    This modifier is enabled when code is executed without compilation.

    Since runtime typing helpers such as `tp_cast` only exist to support
    compiled modes, they introduce unnecessary overhead during normal
    Python execution. This modifier removes those abstractions and
    simplifies the generated AST accordingly.
    """

    @classmethod
    def should_run(cls, options: Options) -> bool:
        """Determine whether the modifier should be applied.

        The modifier runs only when compilation mode is disabled
        and the modifier itself is not disabled.

        Args:
            options (Options):
                Active preprocessing configuration.

        Returns:
            bool:
                `True` if the modifier should run, otherwise `False`.
        """
        return super().should_run(options) and not options.compile

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Simplify generic typing helper calls in pure-Python mode.

        The following transformations are applied:

        - `_omp.tp_cast(type, value)` -> `value`
        - `_omp.tp_typeof(value)` -> `object`

        Since Python uses dynamic typing, explicit runtime casting and
        specialized type queries are unnecessary during interpreted
        execution.

        Args:
            node (ast.Call):
                Call expression node.

        Returns:
            ast.expr:
                Simplified expression node.
        """
        match node.func:
            case ast.Attribute(ast.Name("_omp"), "tp_cast"):
                return node.args[1]
            case ast.Attribute(ast.Name("_omp"), "tp_typeof"):
                return ast.copy_location(ast.Name("object"), node)
        return node
