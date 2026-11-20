"""OpenMP-like data scoping and variable transformation utilities.

This module implements the handling of data-sharing attributes in the
`omp4py` preprocessor, such as `private`, `firstprivate`, `shared`, and
`reduction`.

It is responsible for validating scope clauses, creating transformed
execution scopes, and generating the necessary AST constructs to enforce
OpenMP-like semantics.

The module also provides utilities to resolve variable initialization,
handle reductions, and propagate type annotations across transformed
variables.
"""

from __future__ import annotations

import ast
import copy
import functools
import operator
import typing

from omp4py.core.preprocessor.transformers.operators import get_reduction, resolve_names
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, syntax_error_ctx, team_lock
from omp4py.core.preprocessor.transformers.utils import scope_if

if typing.TYPE_CHECKING:
    from omp4py.core.parser import (
        Construct,
        DataScope,
        Default,
        FirstPrivate,
        Private,
        PyName,
        Reduction,
        ReductionOp,
        Shared,
    )
    from omp4py.core.preprocessor.transformers.symtable import SymbolEntry, SymbolTable

__all__ = ["check_scopes", "create_scope", "dec_annotation", "dec_cast", "scope_names"]


def check_scopes(ctx: Context, *scopes: DataScope, allow_threadprivate: bool = False) -> dict[str, PyName]:
    """Validate variables used in data scope clauses.

    This function ensures that all variables referenced in scope clauses:
    - Exist in the current or enclosing scopes
    - Are not duplicated across clauses
    - Respect thread-private restrictions

    Args:
        ctx (Context): Active transformation context.
        *scopes (DataScope): Data scope clauses to validate.
        allow_threadprivate (bool): Whether threadprivate variables are allowed.

    Returns:
        dict[str, PyName]: Dict of validated variable names.

    Raises:
        SyntaxError: If any validation rule is violated.
    """
    used: dict[str, PyName] = {}

    for scope in scopes:
        for var in scope.targets:
            if not (s := ctx.symtable.get(var.string, True, True)):
                msg = f"name '{var.string}' is not defined"
                raise syntax_error_ctx(msg, var.span, ctx)

            if not allow_threadprivate and s.threadprivate:
                msg = f"'{var.string}' is predetermined 'threadprivate'"
                raise syntax_error_ctx(msg, var.span, ctx)

            if var.string in used:
                msg = f"'{var}' appears more than once in data clauses"
                raise syntax_error_ctx(msg, var.span, ctx)
            used[var.string] = var
    return used


def create_scope(  # ruff: ignore[complex-structure, too-many-arguments, too-many-branches, too-many-positional-arguments]
    ctr: Construct,
    ctx: Context,
    body: list[ast.stmt],
    default: Default | None = None,
    shared_ds: list[Shared] | None = None,
    private_ds: list[Private] | None = None,
    first_private_ds: list[FirstPrivate] | None = None,
    reduction_ds: list[Reduction] | None = None,
    new_python_scope: ast.FunctionDef | None = None,
) -> SymbolTable:
    """Create a new variable scope for a construct.

    This function initializes a new scope based on OpenMP-like data-sharing
    clauses and applies the corresponding transformations to the function
    body.

    This function applies OpenMP-like data-sharing semantics by:
    - Validating scope clauses
    - Determining variable classification (shared, private, etc.)
    - Renaming variables to avoid conflicts
    - Generating initialization code for private and reduction variables
    - Injecting nonlocal/global declarations when required

    It transforms the given function body in-place by inserting
    initialization statements and updating variable references.

    Args:
        ctr (Construct): Construct being processed.
        ctx (Context): Active transformation context.
        body (list[ast.stmt]): AST body representing the scope.
        default (Default | None): Default data-sharing clause.
        shared_ds (list[Shared] | None): Shared clauses.
        private_ds (list[Private] | None): Private clauses.
        first_private_ds (list[FirstPrivate] | None): Firstprivate clauses.
        reduction_ds (list[Reduction] | None): Reduction clauses.
        new_python_scope (ast.FunctionDef | None): Function scope must create
          a new python scope.

    Returns:
        SymbolTable: Symbol table associated with the transformed scope.

    Raises:
        SyntaxError: If scope rules are violated or reductions are invalid.
    """
    shared_ds = shared_ds or []
    private_ds = private_ds or []
    first_private_ds = first_private_ds or []
    reduction_ds = reduction_ds or []

    used_vars = check_scopes(ctx, *(shared_ds or []), *private_ds, *first_private_ds, *reduction_ds)
    required = False
    shared = scope_names(*shared_ds)
    private = scope_names(*private_ds)
    first_private = scope_names(*first_private_ds)
    reduction = scope_names(*reduction_ds)
    reduction_op: dict[str, ReductionOp] = {}

    for scope in reduction_ds:
        for var in scope.str_targets:
            reduction_op[var] = scope.op

    if default is not None:
        match default.data.choice:
            case default.data.Choice.PRIVATE:
                private.extend(ctx.symtable.identifiers())
            case default.data.Choice.FIRST_PRIVATE:
                first_private.extend(ctx.symtable.identifiers())
            case default.data.Choice.NONE:
                required = True

    set_private = {*private, *first_private, *reduction}
    scope_ast = scope_if(body, sorted(set_private))
    f_symbols: SymbolTable = ctx.symtable.rename(set_private, scope_ast)

    if required:
        defined = ctx.symtable.identifiers()
        used = set(shared + private + first_private + reduction)
        for name in f_symbols.identifiers():
            if name not in used and name in defined:
                msg = f"'{name}' not specified in enclosing '{ctr.name.string.lower()}'"
                raise syntax_error_ctx(msg, ctr.span, ctx)

    f_inits: list[ast.stmt] = []
    if new_python_scope:
        scope_table = ctx.symtable

        def create_new_python_scope() -> None:
            freevars: list[SymbolEntry] = [
                symbol
                for symbol in scope_table.check_namespace(scope_ast).symbols()
                if symbol.real_name not in ctx.module_storage.threadprivate
                and (
                    symbol.real_name in reduction_op
                    or (
                        symbol.real_name not in set_private
                        and symbol.assigned
                        and scope_table.get(symbol.old_name, True)
                    )
                )
            ]
            if freevars:
                var_inits: list[ast.stmt] = []
                non_local_vars: list[str] = [symbol.old_name for symbol in freevars if not symbol.global_]
                global_vars: list[str] = [symbol.old_name for symbol in freevars if symbol.global_]

                if non_local_vars:
                    var_inits.append(ctr.span.to_ast(ast.Nonlocal(non_local_vars)))
                if global_vars:
                    var_inits.append(ctr.span.to_ast(ast.Global(global_vars)))
                new_python_scope.body = var_inits + new_python_scope.body

        ctx.node_finalizers.append(create_new_python_scope)

    for name in sorted(set_private):
        if s := f_symbols.get(name):
            ann: ast.expr | None = s.annotation
            f_inits.append(dec_annotation(ctx, s))

            if name in first_private:
                stmt = get_reduction(ctx, "__new__", ann)
                if stmt is None:  # never for __new__
                    continue
                span = used_vars[s.real_name].span
                f_inits.append(resolve_names(span, stmt[1], omp_out=s.scope_name, omp_in=s.old_name))
            elif name in reduction_op:
                if s.private:
                    msg = f"variable '{s.real_name}' is private in outer context"
                    raise syntax_error_ctx(msg, used_vars[s.real_name].span, ctx)

                stmt = get_reduction(ctx, reduction_op[name].op.string, ann)
                span = reduction_op[name].span
                if stmt is None:  # never for __new__
                    msg = f"user defined reduction not found for '{name}'"
                    raise syntax_error_ctx(msg, span, ctx)

                f_inits.append(resolve_names(span, stmt[0], omp_priv=s.scope_name, omp_orig=s.old_name))
                scope_ast.body.append(
                    team_lock(span, [resolve_names(span, stmt[1], omp_out=s.old_name, omp_in=s.scope_name)]),
                )

    scope_ast.body = f_inits + scope_ast.body
    body[:] = [scope_ast]
    return f_symbols


def scope_names(*scopes: DataScope) -> list[str]:
    """Flatten variable names from multiple data scopes.

    Args:
        *scopes (DataScope): Data scope clauses.

    Returns:
        list[str]: List of variable names in declaration order.
    """
    return functools.reduce(operator.iadd, [x.str_targets for x in scopes], [])


def dec_annotation(ctx: Context, s: SymbolEntry, new_name: str | None = None) -> ast.AnnAssign:
    """Generate an annotated assignment for a variable declaration.

    This function creates an `AnnAssign` node that declares a variable
    with the same type as an existing symbol. If no explicit annotation
    is available, a runtime type inference call is used.

    Args:
        ctx (Context): Active transformation context.
        s (SymbolEntry): Symbol entry to replicate.
        new_name (str | None): Optional new variable name.

    Returns:
        ast.AnnAssign: Annotated assignment node.
    """
    ann = s.annotation
    ann = ast.Call(runtime_ast("tp_typeof"), [ast.Name(s.old_name)]) if ann is None else copy.deepcopy(ann)
    return ast.AnnAssign(ast.Name(new_name or s.scope_name, ast.Store()), ann, simple=1)


def dec_cast(ctx: Context, origin: ast.expr, s: SymbolEntry, new_name: str | None = None) -> ast.AnnAssign:
    """Generate a cast assignment for a variable declaration.

    This function creates an `AnnAssign` node that cast a variable
    to the same type as an existing symbol. If no explicit annotation
    is available, a runtime type inference call is used.

    Args:
        ctx (Context): Active transformation context.
        origin (ast.expr): Expression to be cast.
        s (SymbolEntry): Symbol entry to replicate.
        new_name (str | None): Optional new variable name.

    Returns:
        ast.AnnAssign: Annotated assignment node.
    """
    ann = s.annotation
    ann = ast.Call(runtime_ast("tp_typeof"), [ast.Name(s.old_name)]) if ann is None else copy.deepcopy(ann)
    value = ast.Call(runtime_ast("tp_cast"), [ann, origin])
    ann = copy.deepcopy(ann)
    return ast.AnnAssign(ast.Name(new_name or s.scope_name, ast.Store()), ann, value, simple=1)
