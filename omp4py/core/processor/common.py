import ast
import typing
import symtable
import dataclasses

from omp4py.core.processor.processor import NodeContext
from omp4py.core.directive import OmpItem
from omp4py.core.processor.nodes import VariableRenaming, node_name, Variables


def get_item(elems: typing.Iterable[OmpItem], name: str) -> OmpItem | None:
    item: OmpItem
    for item in elems:
        if item.name == name:
            return item
    return None


def name_array(elems: typing.Iterable[OmpItem]) -> list[str]:
    node: OmpItem
    return [node_name(node.value) for node in elems]


def is_constant(expr: ast.expr) -> bool:
    node: ast.AST
    for node in ast.walk(expr):
        if isinstance(node, (ast.Name, ast.NamedExpr, ast.Call)):
            return False
    return True


def code_to_function(ctx: NodeContext, fname: str, body: list[ast.stmt]) -> (ast.FunctionDef, list[str]):
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
    return block_func, be_ref


def data_add(ctx: NodeContext, data_scope: set[str], new_vars: typing.Iterable[OmpItem]):
    item: OmpItem
    for item in new_vars:
        var_name: str = node_name(item.value)
        if ctx.variables.final_name(var_name) not in ctx.variables:
            raise item.tokens[0].make_error(f"'{var_name}' undeclared (first use in this function)")
        if var_name in data_scope:
            raise item.tokens[0].make_error(f"'{var_name}' appears more than once in data clauses")


def data_rename(ctx: NodeContext, body: list[ast.stmt], new_vars: list[str], init: str) -> list[ast.stmt]:
    renaming: dict[str, str] = dict()
    result: list[ast.stmt] = []
    var_name: str
    for var_name in new_vars:
        old_name: str = ctx.variables.final_name(var_name)
        new_name: str = ctx.new_variable(var_name)
        new_value: ast.Call = ctx.new_call(init)
        new_value.args.append(ast.Name(id=old_name, ctx=ast.Load()))
        result.append(ctx.copy_pos(
            ast.Assign(targets=[ast.Name(id=new_name, ctx=ast.Store())], value=new_value)
        ))
        renaming[old_name] = new_name

    VariableRenaming.rename(body, renaming)

    return result


def data_update(ctx: NodeContext, new_vars: typing.Iterable[OmpItem], op: str) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    item: OmpItem
    for item in new_vars:
        var_name: str = node_name(item.value)
        new_name: str = ctx.variables.final_name(var_name)
        old_name: str = ctx.variables.renaming[new_name]

        reduction_value: ast.Call = ctx.new_call(op)
        reduction_value.args.append(ast.Name(id=old_name, ctx=ast.Load()))
        reduction_value.args.append(ast.Name(id=new_name, ctx=ast.Load()))

        result.append(ctx.copy_pos(
            ast.Assign(targets=[ast.Name(id=old_name, ctx=ast.Store())], value=reduction_value)
        ))

    return result


def data_delete(ctx: NodeContext, old_variables: Variables) -> list[ast.stmt]:
    name: str
    del_vars: ast.Delete = ctx.copy_pos(ast.Delete(targets=[]))
    for name in ctx.variables.renaming:
        if name not in old_variables.renaming:
            del_vars.targets.append(ast.Name(id=name, ctx=ast.Del()))
    if len(del_vars.targets) > 0:
        return [del_vars]
    return []


def barrier(ctx: NodeContext) -> ast.stmt:
    return ctx.copy_pos(ast.Expr(ctx.new_call(f'{ctx.r}.barrier')))


def no_wait(ctx: NodeContext, expr: ast.expr) -> ast.stmt:
    if isinstance(expr, ast.Constant):
        if expr.value:
            return ctx.copy_pos(ast.Pass())
        return barrier(ctx)

    return ctx.copy_pos(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=expr), body=[barrier(ctx)], orelse=[]))
