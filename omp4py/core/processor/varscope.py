from __future__ import annotations
import ast
import copy
import typing
import dataclasses

from omp4py.core.directive import OmpItem
import omp4py.core.processor.common as common
import omp4py.core.processor.nodes as nodes


def _const_var() -> list[ast.stmt]:
    return [ast.Assign(
        targets=[ast.Name(id='omp_priv', ctx=ast.Store())],
        value=ast.Name(id='omp_orig', ctx=ast.Load()))]


def _call_op(op: str) -> list[ast.stmt]:
    return [ast.Assign(
        targets=[ast.Name(id='omp_priv', ctx=ast.Store())],
        value=ast.Call(func=ast.Attribute(value=ast.Name(id='__omp', ctx=ast.Load()), attr=op, ctx=ast.Load()),
                       args=[ast.Name(id='omp_orig', ctx=ast.Load())],
                       keywords=[]))]


def _symbol_op(op: ast.operator, mode: int) -> tuple[list[ast.stmt], list[ast.stmt]]:
    init: list[ast.stmt] = _call_op('init_var')
    init[0].value.args.append(ast.Constant(mode))
    return (init,
            [ast.AugAssign(target=ast.Name(id='omp_out', ctx=ast.Store()),
                           op=op,
                           value=ast.Name(id='omp_in', ctx=ast.Load()))])


def _bool_op(op: ast.boolop, mode: int) -> tuple[list[ast.stmt], list[ast.stmt]]:
    init: list[ast.stmt] = _call_op('init_var')
    init[0].value.args.append(ast.Constant(mode))
    return (init,
            [ast.Assign(targets=[ast.Name(id='omp_out', ctx=ast.Store())],
                        value=ast.BoolOp(
                            op=op,
                            values=[ast.Name(id='omp_out', ctx=ast.Load()),
                                    ast.Name(id='omp_in', ctx=ast.Load())]))])


_storage: dict[str, tuple[list[ast.stmt], list[ast.stmt]]] = {
    '__new__': (_call_op('new_var'), []),
    '__new__-int': (_const_var(), []),
    '__new__-float': (_const_var(), []),
    '__new__-tuple': (_const_var(), []),
    '__new__-complex': (_const_var(), []),
    '__new__-str': (_const_var(), []),
    '__new__-bytes': (_const_var(), []),
    '__copy__': (_call_op('cp_var'), []),
    '+': _symbol_op(ast.Add(), 0),
    '-': _symbol_op(ast.Sub(), 0),
    '*': _symbol_op(ast.Mult(), 1),
    '&': _symbol_op(ast.BitAnd(), 2),
    '|': _symbol_op(ast.BitOr(), 0),
    '^': _symbol_op(ast.BitXor(), 0),
    'and': _bool_op(ast.And(), 1),
    'or': _bool_op(ast.Or(), 1),
    # max TODO
    # min TODO
}


@dataclasses.dataclass
class Variables:
    names: set[str] = dataclasses.field(default_factory=set)
    globals: set[str] = dataclasses.field(default_factory=set)
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    shared: set[str] = dataclasses.field(default_factory=set)
    type_comments: dict[str, str] = dataclasses.field(default_factory=dict)
    _storage: dict[str, tuple[list[ast.stmt], list[ast.stmt]]] = \
        dataclasses.field(default_factory=lambda: _storage.copy())

    def final_name(self, name: str) -> str:
        new_name: str = name
        while new_name in self.renaming:
            old_name: str = new_name
            new_name = self.renaming[new_name]
            if name == new_name:
                return old_name
        return new_name

    def add(self, name: str, type_comment: str | None = None):
        if name not in self.globals:
            self.names.add(name)
            self.renaming[name] = name
        if type_comment is not None:
            self.type_comments[name] = type_comment

    def gadd(self, name: str, type_comment: str | None = None):
        self.globals.add(name)
        if type_comment is not None:
            self.type_comments[name] = type_comment

    def add_multiple(self, names: typing.Iterable[str]):
        [self.add(name) for name in names]

    def gadd_multiple(self, names: typing.Iterable[str]):
        [self.gadd(name) for name in names]

    def __contains__(self, item: str) -> bool:
        return item in self.names

    def new_scope(self) -> 'Variables':
        other: Variables = Variables.__new__(Variables)
        other.names = self.names.copy()
        other.globals = self.globals.copy()
        other.renaming = self.renaming.copy()
        other._storage = self._storage.copy()
        return other


@dataclasses.dataclass
class VariableVisitor(ast.NodeVisitor):
    _store: Variables = dataclasses.field(default_factory=Variables)
    _load: Variables = dataclasses.field(default_factory=Variables)

    def visit_declare(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        self._store.add(node.name)

    visit_FunctionDef = visit_declare
    visit_AsyncFunctionDef = visit_declare
    visit_ClassDef = visit_declare

    def visit_ignore(self, node: ast.AST):
        pass

    visit_GeneratorExp = visit_ignore
    visit_Attribute = visit_ignore

    def visit_Global(self, node: ast.Global):
        self._store.gadd_multiple(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self._store.add_multiple(node.names)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._load.add(node.id)
        else:
            self._store.add(node.id)

    def visit_alias(self, node: ast.alias):
        if node.asname is not None:
            self._store.add(node.asname)
        else:
            self._store.add(node.name)

    @classmethod
    def search(cls, nodes: list[ast.stmt]) -> tuple[set[str], set[str]]:
        v: VariableVisitor = VariableVisitor()
        v._load.globals = v._store.globals
        [v.visit(node) for node in nodes]
        return v._store.names, v._load.names


@dataclasses.dataclass
class VariableRenaming(ast.NodeVisitor):
    _names: dict[str, str]

    def visit_Nonlocal(self, node: ast.Nonlocal):
        node._names = [self._names.get(name, name) for name in node._names]

    def visit_Name(self, node: ast.Name):
        node.id = self._names.get(node.id, node.id)

    def visit_alias(self, node: ast.alias):
        if node.asname is not None:
            node.asname = self._names.get(node.asname, node.asname)
        elif node.name in self._names:
            node.asname = self._names[node.name]

    def visit_arg(self, node: ast.arg):
        node.arg = self._names.get(node.arg, node.arg)

    def visit_Attribute(self, node: ast.Attribute):
        pass

    @classmethod
    def rename(cls, nodes: list[ast.stmt], names: dict[str, str]):
        v: VariableRenaming = VariableRenaming(names)
        [v.visit(node) for node in nodes]

    @classmethod
    def copy_rename(cls, nodes: list[ast.stmt], names: dict[str, str]) -> list[ast.stmt]:
        nodes: list[ast.stmt] = copy.deepcopy(nodes)
        cls.rename(nodes, names)
        return nodes


def var_add(ctx: nodes.NodeContext, var_scope: set[str], new_vars: typing.Iterable[OmpItem]):
    item: OmpItem
    for item in new_vars:
        var_name: str = nodes.node_name(item.value)
        if ctx.variables.final_name(var_name) not in ctx.variables:
            raise item.tokens[0].make_error(f"'{var_name}' undeclared (first use in this function)")
        if var_name in var_scope:
            raise item.tokens[0].make_error(f"'{var_name}' appears more than once in data clauses")


def var_rename(ctx: nodes.NodeContext, body: list[ast.stmt], new_vars: list[str], init: str | OmpItem) -> list[
    ast.stmt]:
    renaming: dict[str, str] = dict()
    result: list[ast.stmt] = []
    var_name: str
    for var_name in new_vars:
        old_name: str = ctx.variables.final_name(var_name)
        new_name: str = ctx.new_variable(var_name)

        result.extend(_create_var(ctx, init, old_name, new_name, ctx.variables.type_comments.get(old_name, None), 0))
        # new_value.args.append(ast.Name(id=old_name, ctx=ast.Load()))
        # result.append(ctx.copy_pos(
        #     ast.Assign(targets=[ast.Name(id=new_name, ctx=ast.Store())], value=new_value)
        # ))
        renaming[old_name] = new_name

    VariableRenaming.rename(body, renaming)

    return result


def var_update(ctx: nodes.NodeContext, new_vars: typing.Iterable[OmpItem], op: str | OmpItem) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    item: OmpItem
    for item in new_vars:
        var_name: str = nodes.node_name(item.value)
        new_name: str = ctx.variables.final_name(var_name)
        old_name: str = ctx.variables.renaming[new_name]

        result.extend(_create_var(ctx, op, old_name, new_name, ctx.variables.type_comments.get(old_name, None), 1))

        # reduction_value: ast.Call = ctx.new_call(op)
        # reduction_value.args.append(ast.Name(id=old_name, ctx=ast.Load()))
        # reduction_value.args.append(ast.Name(id=new_name, ctx=ast.Load()))
        #
        # result.append(ctx.copy_pos(
        #     ast.Assign(targets=[ast.Name(id=old_name, ctx=ast.Store())], value=reduction_value)
        # ))

    return result


def var_delete(ctx: nodes.NodeContext, old_variables: Variables) -> list[ast.stmt]:
    name: str
    del_vars: ast.Delete = ctx.copy_pos(ast.Delete(targets=[]))
    for name in ctx.variables.renaming:
        if name not in old_variables.renaming:
            del_vars.targets.append(ast.Name(id=name, ctx=ast.Del()))
    if len(del_vars.targets) > 0:
        return [del_vars]
    return []


def _create_var(ctx: nodes.NodeContext, op: str | OmpItem, old_name: str, new_name: str,
                type_comment: str | None, mode: int) -> list[ast.stmt]:
    op_type: str = str(op) + '-' + str(type_comment)
    template: tuple[list[ast.stmt], list[ast.stmt]] | None = None
    if type_comment is not None:
        template = ctx.variables._storage.get(op_type, None)
    if template is None:
        template = ctx.variables._storage.get(str(op), None)
    if template is None:
        if isinstance(op, str):
            raise ValueError(f"{op} is not defined")
        raise op.tokens[0].make_error(f"'{op.tokens[0]}' is not defined")

    if mode == 0:
        names: dict[str, str] = {
            'omp_orig': old_name,
            'omp_priv': new_name,
            '__omp': ctx.r,
        }
        return VariableRenaming.copy_rename(template[mode], names)
    else:
        names: dict[str, str] = {
            'omp_in': new_name,
            'omp_out': old_name,
            '__omp': ctx.r,
        }
        return common.mutex(ctx, VariableRenaming.copy_rename(template[mode], names))
