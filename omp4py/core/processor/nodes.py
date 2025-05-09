from __future__ import annotations
import ast
import typing
import dataclasses
import copy
from os import rename

from omp4py.core.directive import OmpDirective, OmpClause, parse_line
from omp4py.core.processor.varscope import Variables


@dataclasses.dataclass(frozen=True)
class ParserArgs:
    alias: str
    cache: bool
    dump: bool
    debug: bool
    pure: bool
    compile: bool
    compiler_args: dict
    force: bool
    cache_dir: str


@dataclasses.dataclass
class NodeContext:
    filename: str
    src_lines: list[str]
    parser_args: ParserArgs
    runtime: str
    global_parse: bool
    directive_callback: typing.Callable[['NodeContext', OmpDirective], None] | None = None
    directive: ast.Constant | None = None
    stack: list[ast.AST] = dataclasses.field(default_factory=list)
    variables: Variables = dataclasses.field(default_factory=Variables)
    id_gen: dict[str, int] = dataclasses.field(default_factory=dict)

    def error(self, msg: str, node: ast.stmt | ast.expr) -> SyntaxError:
        ast.fix_missing_locations(self.stack[0])
        lines: list[str] = self.src_lines[node.lineno - 1:node.end_lineno]
        if len(lines) > 1:
            return SyntaxError(msg, (self.filename, node.lineno, node.col_offset + 1, lines[0]))

        return SyntaxError(
            msg, (self.filename, node.lineno, node.col_offset + 1, lines[0], node.end_lineno, node.end_col_offset + 1)
        )

    @property
    def r(self):
        return self.runtime

    def is_omp(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            return self.is_omp(node.func)
        elif isinstance(node, ast.Name):
            return node.id in ("omp", self.parser_args.alias)
        elif isinstance(node, ast.Attribute):
            return node.attr in ("omp", self.parser_args.alias)
        return False

    def new_variable(self, name: str) -> str:
        new_name: str = self.new_id(name)
        self.variables.renaming[new_name] = name
        self.variables.renaming[name] = new_name
        return new_name

    def cast_expression(self, target: str, expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Constant) and type(expr.value).__name__ == target:
            return expr
        to_target = self.new_call(target)
        to_target.args.append(expr)
        return to_target

    def new_id(self, name: str) -> str:
        if name not in self.id_gen:
            self.id_gen[name] = 1
            return f'__omp_{name}'
        n: int = self.id_gen[name]
        self.id_gen[name] += 1
        return f'__omp_{name}_{n}'

    def new_function(self, name: str) -> ast.FunctionDef:
        return self.copy_pos(ast.FunctionDef(name=name, body=[], decorator_list=[],
                                             args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                                kw_defaults=[], defaults=[])
                                             ))

    def array_pos(self, id: str, pos: int, ctx: ast.expr_context):
        return self.copy_pos(ast.Subscript(value=ast.Name(id=id, ctx=ast.Load()), slice=ast.Constant(pos), ctx=ctx))

    def new_call(self, name: str) -> ast.Call:
        func: ast.Name | ast.Attribute
        if "." in name:
            attrs: list[str] = name.split(".")
            func = ast.Name(id=attrs[0], ctx=ast.Load())

            for attr in attrs[1:]:
                func = ast.Attribute(value=func, attr=attr, ctx=ast.Load())
        else:
            func = ast.Name(id=name, ctx=ast.Load())

        return self.copy_pos(ast.Call(func=func, args=[], keywords=[]))

    def new_try(self, body: list[ast.stmt], finalbody: list[ast.stmt]) -> ast.Try:
        return self.copy_pos(ast.Try(body=body, handlers=[], orelse=[], finalbody=finalbody))

    def copy_pos[T](self, node: T) -> T:
        return ast.copy_location(node, self.directive)

    def clone[T](self, node: T) -> T:
        return copy.deepcopy(node)


def check_body(ctx: NodeContext, body: list[ast.stmt]):
    if len(body) == 0:
        raise ctx.error("directive requires statements body", ctx.directive)


def check_nobody(ctx: NodeContext, body: list[ast.stmt]):
    s: ast.stmt
    for s in body:
        if isinstance(s, ast.Pass) or (isinstance(s, ast.Constant) and s.value is ...):
            continue
        raise ctx.error("no statements body allowed", body[0])


def clause_not_implemented(clause: OmpClause) -> SyntaxError:
    return clause.token.make_error("clause is not implemented yet")


def directive_node(ctx: NodeContext, node: ast.Call) -> OmpDirective:
    func: ast.Name | ast.Attribute = typing.cast(ast.Name | ast.Attribute, node.func)
    if len(node.args) != 1:
        raise ctx.error(f"{node_name(func)}() takes exactly one argument", func)

    if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        raise ctx.error(f"{node_name(func)} argument needs constant string expression", func)
    ctx.directive = node.args[0]

    start: int = node.args[0].col_offset
    end: int = node.args[0].end_col_offset
    str_lines: list[str] = ctx.src_lines[node.args[0].lineno - 1:node.args[0].end_lineno]
    str_lines[0] = ' ' * start + str_lines[0][start:]
    str_lines[-1] = str_lines[-1][:end]

    srt_value: str = ''.join(str_lines)

    return parse_line(ctx.filename, srt_value, node.args[0].lineno, True)


def node_name(node: ast.expr | str) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        return node_name(node.func)
    elif isinstance(node, ast.Attribute):
        return node.attr + '.' + node_name(node.value)
    elif isinstance(node, ast.Subscript):
        return node_name(node.value)
    return str(node)


def ast_search(name: str, root: ast.AST | list[ast.AST]) -> ast.expr | ast.stmt | None:
    node: ast.AST
    nroot: ast.AST
    field: typing.Any
    for nroot in root if isinstance(root, list) else [root]:
        for node in ast.walk(nroot):
            for field in ast.iter_fields(node):
                if field[1] == name:
                    return node
        return None
