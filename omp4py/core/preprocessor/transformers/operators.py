import ast
import copy
import typing

from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context

__all__ = ["get_reduction", "new_reduction", "resolve_names"]

_default_reductions: dict[str, tuple[ast.stmt, ast.stmt]] = {}


def _set_reductions(ctx: Context | None) -> dict[str, tuple[ast.stmt, ast.stmt]]:
    if ctx is not None:
        reductions = ctx.module_storage.reductions
        if len(reductions) == 0:
            reductions |= _default_reductions
        return reductions
    return _default_reductions


def new_reduction(ctx: Context | None, name: str, ann: ast.expr, comb: ast.stmt, init: ast.stmt | None = None) -> bool:
    reductions = _set_reductions(ctx)

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
    reductions = _set_reductions(ctx)

    match ann:
        case ast.Name(id="object") | None:
            key = name
        case _:
            key = name + "-" + ast.unparse(ann)

    if key in reductions:
        return reductions[key]
    return reductions.get(name, None)


class _ReductionVars(typing.TypedDict, total=False):
    omp_orig: str
    omp_priv: str
    omp_out: str
    omp_in: str


def resolve_names(stmt: ast.stmt, **r_vars: typing.Unpack[_ReductionVars]) -> ast.stmt:
    new_stmt = copy.deepcopy(stmt)
    for node in ast.walk(new_stmt):
        value: object
        for field, value in ast.iter_fields(node):
            if isinstance(value, str) and value in r_vars:
                setattr(node, field, r_vars[value])
    return new_stmt


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


_bop: dict[str, tuple[ast.boolop, bool]] = {
    "and": (ast.And(), True),
    "or": (ast.Or(), False),
}

for name, (op, neutral) in _bop.items():
    _default_reductions[name] = (
        ast.Assign([ast.Name("omp_priv", ast.Store())], ast.Constant(neutral)),
        ast.Assign([ast.Name("omp_out", ast.Store())], ast.BoolOp(op, [ast.Name("omp_out"), ast.Name("omp_in")])),
    )

_default_reductions["__new__"] = (
    ast.Assign(
        [ast.Name("omp_priv", ast.Store())],
        ast.Call(
            runtime_ast("__new__"),
            [ast.Name("omp_orig")],
        ),
    ),
    ast.Assign(
        [ast.Name("omp_out", ast.Store())],
        ast.Call(
            runtime_ast("__copy__"),
            [ast.Name("omp_in")],
        ),
    ),
)

_inmutables: list[str] = ["int", "float", "complex", "str", "bytes"]

for type_ in _inmutables:
    new_reduction(
        None,
        "__new__",
        ast.Constant(type_),
        ast.Assign([ast.Name("omp_priv", ast.Store())], ast.Name("omp_orig")),
        ast.Assign([ast.Name("omp_out", ast.Store())], ast.Name("omp_in")),
    )
