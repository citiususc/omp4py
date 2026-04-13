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
    return {name.string for scope in scopes for name in scope.targets}


def check_scopes(ctx: Context, *scopes: DataScope, allow_threadprivate: bool = False) -> set[str]:
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


def create_scope(
    ctr: Construct,
    ctx: Context,
    f_ast: ast.FunctionDef,
    default: Default | None,
    shared_ds: list[Shared],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
    return _variable_scope(ctr, ctx, f_ast, default, shared_ds, private_ds, first_private_ds, reduction_ds)


def modify_scope(
    ctr: Construct,
    ctx: Context,
    body: list[ast.stmt],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
    f_ast = ast.FunctionDef("", ast.arguments(), body)
    return _variable_scope(ctr, ctx, f_ast, None, [], private_ds, first_private_ds, reduction_ds)


def _variable_scope(
    ctr: Construct,
    ctx: Context,
    f_ast: ast.FunctionDef,
    default: Default | None,
    shared_ds: list[Shared],
    private_ds: list[Private],
    first_private_ds: list[FirstPrivate],
    reduction_ds: list[Reduction],
) -> SymbolTable:
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
    return functools.reduce(operator.iadd, [x.str_targets for x in scopes], [])


def dec_annotation(ctx: Context, s: SymbolEntry, new_name: str | None = None) -> ast.AnnAssign:
    ann = s.annotation
    if ann is None:
        ann = ast.Call(runtime_ast("cy_typeof"), [ast.Name(s.old_name)])
    return ast.AnnAssign(ast.Name(new_name if new_name else s.scope_name, ast.Store()), ann, simple=1)
