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
import functools
import operator
import typing

from omp4py.core.preprocessor.transformers.operators import get_reduction, resolve_names
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, syntax_error_ctx

if typing.TYPE_CHECKING:
    from omp4py.core.parser.tree import (
        Construct,
        DataScope,
        Default,
        FirstPrivate,
        Private,
        Reduction,
        ReductionOp,
        Shared,
    )
    from omp4py.core.preprocessor.transformers.symtable import SymbolEntry, SymbolTable

__all__ = ["check_scopes", "create_scope", "dec_annotation", "get_scopes", "modify_scope", "scope_names"]


def get_scopes(*scopes: DataScope) -> set[str]:
    """Extract variable names from a collection of data scopes.

    Args:
        *scopes (DataScope): Data scope clauses.

    Returns:
        set[str]: Set of variable names referenced in the scopes.
    """
    return {name.string for scope in scopes for name in scope.targets}


def check_scopes(ctx: Context, *scopes: DataScope, allow_threadprivate: bool = False) -> set[str]:
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
        set[str]: Set of validated variable names.

    Raises:
        SyntaxError: If any validation rule is violated.
    """
    used: set[str] = set()

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
            used.add(var.string)
    return used


def create_scope( # noqa: PLR0913
    ctr: Construct,
    ctx: Context,
    f_ast: ast.FunctionDef,
    default: Default | None,
    shared_ds: list[Shared],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
    """Create a new variable scope for a construct.

    This function initializes a new scope based on OpenMP-like data-sharing
    clauses and applies the corresponding transformations to the function
    body.

    Args:
        ctr (Construct): Construct being processed.
        ctx (Context): Active transformation context.
        f_ast (ast.FunctionDef): Function node representing the new scope.
        default (Default | None): Default data-sharing clause.
        shared_ds (list[Shared]): Shared variable clauses.
        private_ds (list[Private]): Private variable clauses.
        first_private_ds (list[FirstPrivate]): Firstprivate clauses.
        reduction_ds (list[Reduction]): Reduction clauses.

    Returns:
        SymbolTable: Symbol table associated with the new scope.
    """
    return _variable_scope(ctr, ctx, f_ast, default, shared_ds, private_ds, first_private_ds, reduction_ds)


def modify_scope( # noqa: PLR0913
    ctr: Construct,
    ctx: Context,
    body: list[ast.stmt],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
    """Modify an existing scope with additional data-sharing clauses.

    Unlike `create_scope`, this function operates on an existing body,
    applying private, firstprivate, and reduction semantics without
    introducing a full new function scope.

    Args:
        ctr (Construct): Construct being processed.
        ctx (Context): Active transformation context.
        body (list[ast.stmt]): AST body to transform.
        private_ds (list[Private]): Private clauses.
        first_private_ds (list[FirstPrivate]): Firstprivate clauses.
        reduction_ds (list[Reduction]): Reduction clauses.

    Returns:
        SymbolTable: Updated symbol table for the modified scope.
    """
    f_ast = ast.FunctionDef("", ast.arguments(), body)
    return _variable_scope(ctr, ctx, f_ast, None, [], private_ds, first_private_ds, reduction_ds)


def _variable_scope( # noqa: C901 PLR0912 PLR0913
    ctr: Construct,
    ctx: Context,
    f_ast: ast.FunctionDef,
    default: Default | None,
    shared_ds: list[Shared],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
    """Core implementation for scope creation and transformation.

    This internal function applies OpenMP-like data-sharing semantics by:
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
        f_ast (ast.FunctionDef): Function node representing the scope.
        default (Default | None): Default data-sharing clause.
        shared_ds (list[Shared]): Shared clauses.
        private_ds (list[Private]): Private clauses.
        first_private_ds (list[FirstPrivate]): Firstprivate clauses.
        reduction_ds (list[Reduction]): Reduction clauses.

    Returns:
        SymbolTable: Symbol table associated with the transformed scope.

    Raises:
        SyntaxError: If scope rules are violated or reductions are invalid.
    """
    new_scope = f_ast.name != ""
    check_scopes(ctx, *shared_ds, *private_ds, *first_private_ds, *reduction_ds)
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
        match default.type:
            case default.Type.PRIVATE:
                private.extend(ctx.symtable.identifiers())
            case default.Type.FIRST_PRIVATE:
                first_private.extend(ctx.symtable.identifiers())
            case default.Type.NONE:
                required = True

    set_private = {*private, *first_private, *reduction}
    f_symbols: SymbolTable = ctx.symtable.rename(set_private, f_ast)

    if required:
        defined = ctx.symtable.identifiers()
        used = set(shared + private + first_private + reduction)
        for name in f_symbols.identifiers():
            if name not in used and name in defined:
                msg = f"'{name}' not specified in enclosing '{ctr.name.string.lower()}'"
                raise syntax_error_ctx(msg, ctr.span, ctx)

    f_inits: list[ast.stmt] = []
    if new_scope:
        freevars: list[SymbolEntry] = [
            symbol
            for symbol in ctx.symtable.check_namespace(f_ast).symbols()
            if symbol.real_name not in ctx.module_storage.threadprivate
            and (
                symbol.real_name in reduction_op
                or (symbol.real_name not in set_private and symbol.assigned and ctx.symtable.get(symbol.old_name, True))
            )
        ]
        if freevars:
            non_local_vars: list[str] = [symbol.old_name for symbol in freevars if not symbol.global_]
            global_vars: list[str] = [symbol.old_name for symbol in freevars if symbol.global_]

            if non_local_vars:
                f_inits.append(ctr.span.to_ast(ast.Nonlocal(non_local_vars)))
            if global_vars:
                f_inits.append(ctr.span.to_ast(ast.Global(global_vars)))

    for name in sorted(set_private):
        if s := f_symbols.get(name, ann=True):
            ann: ast.expr | None = None
            f_inits.append(dec_annotation(ctx, s))

            if name in first_private:
                stmt = get_reduction(ctx, "__new__", ann)
                if stmt is None:  # never for __new__
                    continue
                f_inits.append(resolve_names(stmt[1], omp_out=s.scope_name, omp_in=s.old_name))
            elif name in reduction_op:
                stmt = get_reduction(ctx, reduction_op[name].value, ann)
                if stmt is None:  # never for __new__
                    msg = f"user defined reduction not found for '{name}'"
                    raise syntax_error_ctx(msg, reduction_op[name].span, ctx)
                f_inits.append(resolve_names(stmt[0], omp_priv=s.scope_name, omp_orig=s.old_name))
                f_ast.body.append(resolve_names(stmt[1], omp_out=s.old_name, omp_in=s.scope_name))

    f_ast.body[:] = f_inits + f_ast.body
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
    if ann is None:
        ann = ast.Call(runtime_ast("cy_typeof"), [ast.Name(s.old_name)])
    return ast.AnnAssign(ast.Name(new_name or s.scope_name, ast.Store()), ann, simple=1)
