from __future__ import annotations
import ast
import typing
import symtable

import omp4py.core.processor.nodes as nodes
from omp4py.core.directive import OmpItem


def get_item(elems: typing.Iterable[OmpItem], name: str) -> OmpItem | None:
    item: OmpItem
    for item in elems:
        if item.name == name:
            return item
    return None


def name_array(elems: typing.Iterable[OmpItem]) -> list[str]:
    node: OmpItem
    return [nodes.node_name(node.value) for node in elems]


def is_constant(expr: ast.expr) -> bool:
    node: ast.AST
    for node in ast.walk(expr):
        if isinstance(node, (ast.Name, ast.NamedExpr, ast.Call)):
            return False
    return True


def code_to_function(ctx: nodes.NodeContext, fname: str, body: list[ast.stmt]) -> (ast.FunctionDef, list[str]):
    maybe_ref: list[str] = list(ctx.variables.names)

    block_func: ast.FunctionDef = ctx.new_function(fname)
    block_func.body.append(ast.Nonlocal(names=maybe_ref))
    block_func.body.extend(body)

    fake_func: ast.FunctionDef = ctx.new_function("fake")
    fake_func.args.args = [ast.arg(n) for n in maybe_ref]
    fake_func.body.append(block_func)

    table = symtable.symtable(ast.unparse(fake_func), "string", "exec")

    be_ref: list[str] = list()
    be_decl: list[str] = list()
    be_global: list[str] = list()
    s: symtable.Symbol
    for s in table.get_children()[0].get_children()[0].get_symbols():
        sname: str = s.get_name()
        if sname in ctx.variables.globals:
            be_global.append(sname)
            continue
        if s.is_free():
            if s.is_assigned():
                be_decl.append(sname)
            elif s.is_referenced():
                be_ref.append(sname)
    if len(be_decl) == 0:
        block_func.body = block_func.body[1:]
    else:
        block_func.body[0] = ast.Nonlocal(names=be_decl)
    return block_func, be_ref + be_decl


def barrier(ctx: nodes.NodeContext) -> ast.stmt:
    return ctx.copy_pos(ast.Expr(ctx.new_call(f'{ctx.r}.barrier')))


def mutex(ctx: nodes.NodeContext, body: list[ast.stmt]) -> [ast.stmt]:
    mutex_body: list[ast.stmt] = list()

    lock: ast.Call = ctx.new_call(f'{ctx.r}.mutex_lock')
    unlock: ast.Call = ctx.new_call(f'{ctx.r}.mutex_unlock')

    mutex_body.append(ctx.copy_pos(ast.Expr(lock)))
    mutex_body.extend(body)

    return [ctx.new_try(mutex_body, [ctx.copy_pos(ast.Expr(unlock))])]


def no_wait(ctx: nodes.NodeContext, expr: ast.expr) -> ast.stmt:
    if isinstance(expr, ast.Constant):
        if expr.value:
            return ctx.copy_pos(ast.Pass())
        return barrier(ctx)

    return ctx.copy_pos(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=expr), body=[barrier(ctx)], orelse=[]))


class OmpItemError(ValueError):
    value: OmpItem

    def __init__(self, value: OmpItem, message: str):
        super().__init__(self, message)
        self.value = value
