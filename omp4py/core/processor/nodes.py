import ast
import typing
import dataclasses
import copy

from omp4py.core.directive import OmpDirective, OmpClause, tokenizer, parse_line


@dataclasses.dataclass()
class Variables:
    names: set[str] = dataclasses.field(default_factory=set)
    globals: set[str] = dataclasses.field(default_factory=set)
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    shared: set[str] = dataclasses.field(default_factory=set)

    def final_name(self, name: str) -> str:
        new_name: str = name
        while new_name in self.renaming:
            old_name: str = new_name
            new_name = self.renaming[new_name]
            if name == new_name:
                return old_name
        return new_name

    def add(self, name: str):
        if name not in self.globals:
            self.names.add(name)
            self.renaming[name] = name

    def gadd(self, name: str):
        self.globals.add(name)

    def add_multiple(self, names: typing.Iterable[str]):
        [self.add(name) for name in names]

    def gadd_multiple(self, names: typing.Iterable[str]):
        [self.gadd(name) for name in names]

    def __contains__(self, item: str) -> bool:
        return item in self.names

    def new_scope(self) -> 'Variables':
        copy: Variables = Variables()
        copy.names = self.names.copy()
        copy.globals = self.globals.copy()
        copy.renaming = self.renaming.copy()
        return copy


@dataclasses.dataclass
class NodeContext:
    filename: str
    src_lines: list[str]
    omp_alias: str
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

    def is_omp(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            return self.is_omp(node.func)
        elif isinstance(node, ast.Name):
            return node.id in ("omp", self.omp_alias)
        elif isinstance(node, ast.Attribute):
            return self.is_omp(node.value)
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

    def array_pos(self, id:str, pos:int, ctx:ast.expr_context):
        return  self.copy_pos(ast.Subscript(value=ast.Name(id=id, ctx=ast.Load()), slice=ast.Constant(pos), ctx=ctx))

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


def check_body(body: list[ast.stmt]):
    if len(body) == 0:
        raise ValueError("directive requires statements body")


def check_nobody(body: list[ast.stmt]):
    if len(body) > 0:
        raise ValueError("no statements body allowed")


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


@dataclasses.dataclass
class VariableVisitor(ast.NodeVisitor):
    store: Variables = dataclasses.field(default_factory=Variables)
    load: Variables = dataclasses.field(default_factory=Variables)

    def visit_declare(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        self.store.add(node.name)

    visit_FunctionDef = visit_declare
    visit_AsyncFunctionDef = visit_declare
    visit_ClassDef = visit_declare

    def visit_ignore(self, node: ast.AST):
        pass

    visit_GeneratorExp = visit_ignore
    visit_Attribute = visit_ignore

    def visit_Global(self, node: ast.Global):
        self.store.gadd_multiple(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.store.add_multiple(node.names)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.load.add(node.id)
        else:
            self.store.add(node.id)

    def visit_alias(self, node: ast.alias):
        if node.asname is not None:
            self.store.add(node.asname)
        else:
            self.store.add(node.name)

    @classmethod
    def search(cls, nodes: list[ast.stmt]) -> tuple[set[str], set[str]]:
        v: VariableVisitor = VariableVisitor()
        v.load.globals = v.store.globals
        [v.visit(node) for node in nodes]
        return v.store.names, v.load.names


@dataclasses.dataclass
class VariableRenaming(ast.NodeVisitor):
    names: dict[str, str]

    def visit_Nonlocal(self, node: ast.Nonlocal):
        node.names = [self.names.get(name, name) for name in node.names]

    def visit_Name(self, node: ast.Name):
        node.id = self.names.get(node.id, node.id)

    def visit_alias(self, node: ast.alias):
        if node.asname is not None:
            node.asname = self.names.get(node.asname, node.asname)
        elif node.name in self.names:
            node.asname = self.names[node.name]

    def visit_arg(self, node: ast.arg):
        node.arg = self.names.get(node.arg, node.arg)

    def visit_Attribute(self, node: ast.Attribute):
        pass

    @classmethod
    def rename(cls, nodes: list[ast.stmt], names: dict[str, str]):
        v: VariableRenaming = VariableRenaming(names)
        [v.visit(node) for node in nodes]
