"""Reduction operation handling for the `omp4py` preprocessor.

This module implements OpenMP-like reduction semantics within the
`omp4py` transformation pipeline.

It provides mechanisms to define, register, and resolve reduction
operations, including both built-in and user-defined reductions.

Reduction operations are expressed as AST templates using special
placeholder variables (`omp_priv`, `omp_orig`, `omp_out`, `omp_in`)
that are later resolved into concrete variable names during
transformation.

The module also defines default reduction operators and initialization
rules, including a generic `__new__` operation for object creation and
copy semantics.
"""

from __future__ import annotations

import ast
import copy
import typing

from omp4py.core.preprocessor.transformers.symtable import runtime_ast

if typing.TYPE_CHECKING:
    from omp4py.core.preprocessor.transformers.transformer import Context

__all__ = ["get_reduction", "new_reduction", "resolve_names"]


# Default reduction definitions.

# Each entry maps a reduction name to a tuple:
# (initialization statement, combination statement)

# The initialization defines how the private variable (`omp_priv`)
# is initialized, while the combination defines how partial results
# (`omp_in`) are merged into the final result (`omp_out`).
_default_reductions: dict[str, tuple[ast.stmt, ast.stmt]] = {}


def _get_reductions(ctx: Context | None) -> dict[str, tuple[ast.stmt, ast.stmt]]:
    """Retrieve the active reduction registry.

    This function returns the reduction mapping associated with the
    current module context. If no reductions have been registered yet,
    default reductions are initialized.

    Args:
        ctx (Context | None): Active transformation context.

    Returns:
        dict[str, tuple[ast.stmt, ast.stmt]]: Mapping from reduction
        keys to initialization and combination statements.
    """
    if ctx is not None:
        reductions = ctx.module_storage.reductions
        if len(reductions) == 0:
            reductions |= _default_reductions
        return reductions
    return _default_reductions


def new_reduction(ctx: Context | None, name: str, ann: ast.expr, comb: ast.stmt, init: ast.stmt | None = None) -> bool:
    """Register a new reduction operation.

    This function allows defining custom reductions by specifying
    initialization and combination AST templates.

    Reductions may be specialized by type annotation. If no explicit
    initialization is provided, a default `__new__`-based initialization
    is used.

    Args:
        ctx (Context | None): Active transformation context.
        name (str): Reduction name (e.g., '+', 'max').
        ann (ast.expr): Type annotation associated with the reduction.
        comb (ast.stmt): Combination operation template.
        init (ast.stmt | None): Initialization template.

    Returns:
        bool: True if the reduction was successfully registered, False
        if a reduction with the same key already exists.
    """
    reductions = _get_reductions(ctx)

    match ann:
        case ast.Name(id="object"):
            key = name
        case _:
            ann_name: str = ast.unparse(ann)
            key = name + "-" + ann_name
            if init is None and "__new__-" + ann_name in _default_reductions:
                init = _default_reductions["__new__-" + ann_name][0]

    if init is None:
        init = _default_reductions["__new__"][0]

    if key in reductions:
        return False
    reductions[key] = (init, comb)
    return True


def get_reduction(ctx: Context, name: str, ann: ast.expr | None = None) -> tuple[ast.stmt, ast.stmt] | None:
    """Retrieve a reduction operation.

    This function resolves a reduction by name and optional type
    annotation. If a specialized version is not found, a generic
    version is returned if available.

    Args:
        ctx (Context): Active transformation context.
        name (str): Reduction name.
        ann (ast.expr | None): Optional type annotation.

    Returns:
        tuple[ast.stmt, ast.stmt] | None: Initialization and combination
        templates, or None if not found.
    """
    reductions = _get_reductions(ctx)

    match ann:
        case ast.Name(id="object") | None:
            key = name
        case _:
            key = name + "-" + ast.unparse(ann)

    if key in reductions:
        return reductions[key]
    return reductions.get(name, None)


class _ReductionVars(typing.TypedDict, total=False):
    """Mapping of placeholder variables used in reduction templates.

    These variables are used inside AST templates and later replaced
    with concrete variable names during transformation.

    Attributes:
        omp_orig (str): Original variable.
        omp_priv (str): Private copy of the variable.
        omp_out (str): Output variable for reduction.
        omp_in (str): Input variable for reduction.
    """
    omp_orig: str
    omp_priv: str
    omp_out: str
    omp_in: str


def resolve_names(stmt: ast.stmt, **r_vars: typing.Unpack[_ReductionVars]) -> ast.stmt:
    """Resolve placeholder variable names in a reduction template.

    This function replaces occurrences of placeholder identifiers
    (`omp_*`) in an AST statement with concrete variable names.

    A deep copy of the statement is created to preserve the original
    template.

    Args:
        stmt (ast.stmt): Template AST statement.
        **r_vars: Mapping from placeholder names to actual identifiers.

    Returns:
        ast.stmt: Transformed AST statement with resolved names.
    """
    new_stmt = copy.deepcopy(stmt)
    for node in ast.walk(new_stmt):
        value: object
        for field, value in ast.iter_fields(node):
            if isinstance(value, str) and value in r_vars:
                setattr(node, field, r_vars[value])
    return new_stmt

# Mapping of arithmetic operators to their AST representation and
# neutral element. These are used to define default reductions.
_op: dict[str, tuple[ast.operator, int]] = {
    "+": (ast.Add(), 0),
    "-": (ast.Sub(), 0),
    "*": (ast.Mult(), 1),
    "&": (ast.BitAnd(), ~0),
    "|": (ast.BitOr(), 0),
    "^": (ast.BitXor(), 0),
}

for name, (op, neutral) in _op.items():
    _default_reductions[name] = (
        ast.Assign([ast.Name("omp_priv", ast.Store())], ast.Constant(neutral)),
        ast.AugAssign(ast.Name("omp_out", ast.Store()), op, ast.Name("omp_in")),
    )


# Mapping of boolean operators to their AST representation and
# neutral element, used for logical reductions.
_bop: dict[str, tuple[ast.boolop, bool]] = {
    "and": (ast.And(), True),
    "or": (ast.Or(), False),
}

for name, (op, neutral) in _bop.items():
    _default_reductions[name] = (
        ast.Assign([ast.Name("omp_priv", ast.Store())], ast.Constant(neutral)),
        ast.Assign([ast.Name("omp_out", ast.Store())], ast.BoolOp(op, [ast.Name("omp_out"), ast.Name("omp_in")])),
    )

# Special reduction used for object creation and copying.

# This reduction defines how a private copy is created from the
# original variable and how results are propagated back. It is
# used as a fallback for types without explicit reduction rules.
_default_reductions["__new__"] = (
    ast.Assign(
        [ast.Name("omp_priv", ast.Store())],
        ast.Call(
            runtime_ast("new_var"),
            [ast.Name("omp_orig")],
        ),
    ),
    ast.Assign(
        [ast.Name("omp_out", ast.Store())],
        ast.Call(
            runtime_ast("copy_var"),
            [ast.Name("omp_in")],
        ),
    ),
)


# List of immutable types with specialized `__new__` behavior.

# For these types, copying is replaced with direct assignment,
# avoiding unnecessary runtime calls.
_inmutables: list[str] = ["int", "float", "complex", "str", "bytes"]

for type_ in _inmutables:
    new_reduction(
        None,
        "__new__",
        ast.Constant(type_),
        ast.Assign([ast.Name("omp_priv", ast.Store())], ast.Name("omp_orig")),
        ast.Assign([ast.Name("omp_out", ast.Store())], ast.Name("omp_in")),
    )
